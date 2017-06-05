import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from language_model import LM
from common import CheckpointLoader
from utils import get_topk_index

def run_train(dataset, hps, logdir, ps_device, task=0, master=""):
    with tf.variable_scope("model"):
        model = LM(hps, "train", ps_device)
    print("ALL VARIABLES")
    for v in tf.all_variables():
        print("%s %s %s" % (v.name, v.get_shape(), v.device))
    print("TRAINABLE VARIABLES")
    for v in tf.trainable_variables():
        print("%s %s %s" % (v.name, v.get_shape(), v.device))
    print("LOCAL VARIABLES")
    for v in tf.local_variables():
        print("%s %s %s" % (v.name, v.get_shape(), v.device))

    sv = tf.train.Supervisor(is_chief=(task == 0),
                             logdir=logdir,
                             summary_op=None,  # Automatic summaries don't work with placeholders.
                             global_step=model.global_step,
                             save_summaries_secs=30,
                             save_model_secs=120 * 5)

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=2,
                            inter_op_parallelism_threads=20)
    with sv.managed_session(master, config=config) as sess:
        # Slowly increase the number of workers during beginning of the training.
        while not sv.should_stop():
            step = int(sess.run(model.global_step))
            print("step:%d" % (step))
            waiting_until_step = task * hps.num_delayed_steps
            if step >= waiting_until_step:
                break
            else:
                print("Current step is %d. Waiting until: %d" % (step, waiting_until_step))
            time.sleep(10.0)

        local_step = 0
        prev_global_step = sess.run(model.global_step)
        prev_time = time.time()
        data_iterator = dataset.iterate_forever(hps.batch_size * hps.num_gpus, hps.num_steps)
        while not sv.should_stop():
            fetches = [model.global_step, model.loss, model.train_op]
            # Chief worker computes summaries every 20 steps.
            should_compute_summary = (task == 0 and local_step > 0 and local_step % 20 == 0)
            if should_compute_summary:
                fetches += [model.summary_op]

            x, y, w = next(data_iterator)
            should_run_profiler = (hps.run_profiler and task == 0 and local_step % 1000 == 13)    #False
            if should_run_profiler:                 #False
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                fetched = sess.run(fetches, {model.x: x, model.y: y, model.w: w},
                                   options=run_options, run_metadata=run_metadata)
                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                print("Running profiler")
                with open(logdir + "/timeline.json", 'w') as f:
                    f.write(ctf)
                print("Finished profiling!")
            else:
                fetched = sess.run(fetches, {model.x: x, model.y: y, model.w: w})

            local_step += 1
            if should_compute_summary:
                sv.summary_computed(sess, fetched[-1])

            if local_step < 10 or local_step % 20 == 0:
                cur_time = time.time()
                num_words = hps.batch_size * hps.num_gpus * hps.num_steps
                wps = (fetched[0] - prev_global_step) * num_words / (cur_time - prev_time)
                prev_global_step = fetched[0]
                print("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f, ppl = %.4f" % (
                    fetched[0], cur_time - prev_time, wps, fetched[1],np.exp(fetched[1])))
                prev_time = cur_time
    sv.stop()


def run_eval(dataset, hps, logdir, mode, num_eval_steps):
    with tf.variable_scope("model"):
        hps.num_sampled = 0  # Always using full softmax at evaluation.   run out of memory
        hps.keep_prob = 1.0
        model = LM(hps, "eval", "/cpu:0")

    if hps.average_params:
        print("Averaging parameters for evaluation.")
        saver = tf.train.Saver(model.avg_dict)
    else:
        saver = tf.train.Saver()

    # Use only 4 threads for the evaluation.
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=20,
                            inter_op_parallelism_threads=1)
    sess = tf.Session(config=config)
    sw = tf.summary.FileWriter(logdir + "/" + mode, sess.graph)
    ckpt_loader = CheckpointLoader(saver, model.global_step, logdir + "/train")
    with sess.as_default():
            ckpt_loader.load_checkpoint()  #  FOR ONLY ONE CHECKPOINT
            global_step = ckpt_loader.last_global_step
            data_iterator = dataset.iterate_once(hps.batch_size * hps.num_gpus, hps.num_steps)
            sess.run(tf.local_variables_initializer())
            print("global_step:",global_step)
            loss_nom = 0.0
            loss_den = 0.0
            for i, (x, y, w) in enumerate(data_iterator):
                if i >= num_eval_steps:
                    break
                loss = sess.run(model.loss, {model.x: x, model.y: y, model.w: w})
                loss_nom += loss
                loss_den += w.mean()
                loss = loss_nom / loss_den
                sys.stdout.write("%d: %.3f (%.3f) ... " % (i, loss, np.exp(loss)))
                sys.stdout.flush()
            sys.stdout.write("\n")

            log_perplexity = loss_nom / loss_den
            print("Results at %d: log_perplexity = %.3f perplexity = %.3f" % (
                global_step, log_perplexity, np.exp(log_perplexity)))

            summary = tf.Summary()
            summary.value.add(tag='eval/log_perplexity', simple_value=log_perplexity)
            summary.value.add(tag='eval/perplexity', simple_value=np.exp(log_perplexity))
            sw.add_summary(summary, global_step)
            sw.flush()


def predict_next(dataset, hps, logdir, mode, num_eval_steps,vocab):
    with tf.variable_scope("model"):
        hps.num_sampled = 0  # Always using full softmax at evaluation.   run out of memory
        hps.keep_prob = 1.0
        model = LM(hps,"predict_next", "/cpu:0")

    if hps.average_params:
        print("Averaging parameters for evaluation.")
        saver = tf.train.Saver(model.avg_dict)
    else:
        saver = tf.train.Saver()

    # Use only 4 threads for the evaluation.
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=20,
                            inter_op_parallelism_threads=1)
    sess = tf.Session(config=config)
    sw = tf.summary.FileWriter(logdir + "/" + mode, sess.graph)
    ckpt_loader = CheckpointLoader(saver, model.global_step, logdir + "/train")
    with sess.as_default():
            ckpt_loader.load_checkpoint()  #  FOR ONLY ONE CHECKPOINT
            global_step = ckpt_loader.last_global_step
            data_iterator = dataset.iterate_once(hps.batch_size * hps.num_gpus, hps.num_steps)
            sess.run(tf.local_variables_initializer())
            print("global_step:",global_step)
            loss_nom = 0.0
            loss_den = 0.0
            cur_time = time.time()
            savedKey = 0
            totalKey = 0
            '''
            text = open("data/news.en.heldout-00001-of-00050","r")
            for kk,line in enumerate(text):
                totalKey += len(line.strip())
                if kk==0:
                    print len(line)
            print "totalKey:",totalKey
            '''
            predicted_words = []
            for i, (x, y, w) in enumerate(data_iterator):
                #if i >= num_eval_steps:
                #    break
                '''
                print "i",i
                print "x",x
                
                for j in x[:]:
                    print j
                    for jj in j:
                        print vocab.get_token(jj)
                '''
                #print "x:",[vocab.get_token(ix) for ix in x[0]]
                #print "y:",[vocab.get_token(ix) for ix in y[0]]
                inputs = [vocab.get_token(ix) for ix in x[0]]
                labels = [vocab.get_token(ix) for ix in y[0]]
                loss,logits,indexes = sess.run([model.loss,model.logits,model.index], {model.x: x, model.y: y, model.w: w})
                #print logits.shape,indexes
                #print indexes[0]
                tmpKS = 0
                tmpAllKey = 0

                for step in range(hps.num_steps):
                    words = []
                    totalKey += len(inputs[step])
                    tmpAllKey += len(inputs[step])
                    if step >0 :
                        totalKey += 1  # for space between two keys 
                        tmpAllKey += 1    
                    for j in range(hps.arg_max):
                        word = vocab.get_token(indexes[0][step][j])
                        words += [word]
                        if word == labels[step]:
                            predicted_words += [word]
                            tmpKS +=  len(labels[step])
                            savedKey += len(labels[step])
                    #print "predict: ", words
                # print "x:",x
                print("i:%6d,  savedKey:%d , totalKey:%d,  ksr : %.3f "%(i, tmpKS, tmpAllKey, tmpKS*1.0/tmpAllKey))
            print("savedKey:%d , totalKey:%d,  ksr : %.3f "%(savedKey, totalKey, savedKey*1.0/totalKey))
            print("predicted_words:")
            print(predicted_words)
            now = time.time()
            print "time:",now-cur_time
