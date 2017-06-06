import tensorflow as tf
from utils import get_topk_index

from model_utils import sharded_variable, LSTMCell
from common import assign_to_gpu, average_grads, find_trainable_variables
from hparams import HParams


class LM(object):
    def __init__(self, hps,mode="train", ps_device="/gpu:0"):
        self.hps = hps
        data_size = hps.batch_size * hps.num_gpus
        self.x = tf.placeholder(tf.int32, [data_size, hps.num_steps])
        self.y = tf.placeholder(tf.int32, [data_size, hps.num_steps])
        self.w = tf.placeholder(tf.int32, [data_size, hps.num_steps])
        losses = []
        tower_grads = []
        logitses = []
        indexes = []
        if mode == "predict_next":
            self.ind = tf.placeholder(tf.int32,name="ind")
            self.ind_len = tf.placeholder(tf.int32,name="ind_len")

        xs = tf.split(self.x, hps.num_gpus, 0)
        ys = tf.split(self.y, hps.num_gpus, 0)
        ws = tf.split(self.w, hps.num_gpus, 0)
        print ("ngpus:",hps.num_gpus)
        for i in range(hps.num_gpus):
            with tf.device(assign_to_gpu(i, ps_device)), tf.variable_scope(tf.get_variable_scope(),
                                                                           reuse=True if i > 0 else None):
                if mode == "predict_next":
                    loss, logits, index = self._forward(i, xs[i], ys[i], ws[i], mode=mode)
                    logitses += [logits]
                    indexes += [index]
                    #self.logits = logits
                else:
                    loss = self._forward(i, xs[i], ys[i], ws[i])
                #self.logits = logits
                losses += [loss]
                if mode == "train":
                    cur_grads = self._backward(loss, summaries=(i == hps.num_gpus - 1))
                    tower_grads += [cur_grads]
        if mode == "predict_next":   # ngpus = 1, nlayers = 1, nums_step =1
            self.logits = logitses
            self.index = indexes
            ind_logits =  tf.reshape(logitses[0], [hps.vocab_size, -1])
            ind_logits =  tf.gather(ind_logits, self.ind)
            print  "ind_logits:", logitses, ind_logits
            ind_logits =  tf.reshape(ind_logits, [-1, self.ind_len])
            self.top_k = tf.minimum(self.ind_len, hps.arg_max) 
            _, self.ind_index    =  tf.nn.top_k(ind_logits, self.top_k)
            print  "ind_index:",  self.ind_index


        self.loss = tf.add_n(losses) / len(losses)  # total loss 
        tf.summary.scalar("model/loss", self.loss)

        self.global_step = tf.get_variable("global_step", [], tf.int32,trainable=False)

        if mode == "train":
            grads = average_grads(tower_grads)
            optimizer = tf.train.AdagradOptimizer(hps.learning_rate, initial_accumulator_value=1.0)
            self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
            self.summary_op = tf.summary.merge_all()
            #print self.summary_op
        else:
            self.train_op = tf.no_op()

        if mode in ["train", "eval","predict_next"] and hps.average_params:
            with tf.name_scope(None):  # This is needed due to EMA implementation silliness.
                # Keep track of moving average of LSTM variables.
                ema = tf.train.ExponentialMovingAverage(decay=0.999)
                variables_to_average = find_trainable_variables("LSTM")
                self.train_op = tf.group(*[self.train_op, ema.apply(variables_to_average)])
                self.avg_dict = ema.variables_to_restore(variables_to_average)

    def _forward(self, gpu, x, y, w, mode="train"):
        hps = self.hps
        w = tf.to_float(w)
        self.initial_states = []
        #every layer has a initial_state :tf.zero([hps.batch_size, hps.state_size + hps.projected_size])
        for i in range(hps.num_layers):
            with tf.device("/gpu:%d" % gpu):
                v = tf.Variable(tf.zeros([hps.batch_size, hps.state_size + hps.projected_size]), trainable=False,
                                collections=[tf.GraphKeys.LOCAL_VARIABLES], name="state_%d_%d" % (gpu, i))
                #self.initial_states += [v]
                self.initial_states = v   # for layers = 1

        emb_vars = sharded_variable("emb", [hps.vocab_size, hps.emb_size], hps.num_shards)  #vocab_size is too big for this model
        x = tf.nn.embedding_lookup(emb_vars, x)  # [batch_size, steps, emb_size]
        if hps.keep_prob < 1.0:
            x = tf.nn.dropout(x, hps.keep_prob)
        
        #  [batch_size,emb_size]*steps        
        inputs = [tf.squeeze(v, [1]) for v in tf.split(x, hps.num_steps, 1)]
        
        for i in range(hps.num_layers):
            with tf.variable_scope("lstm_%d" % i):
                cell = LSTMCell(hps.state_size, hps.emb_size, num_proj=hps.projected_size)

            state = self.initial_states
            for t in range(hps.num_steps):  # step 
                inputs[t], state = cell(inputs[t], state)  #inputs[t] is h{t}??  state is h(t) ?? result of inputs[t] is project ??
                if hps.keep_prob < 1.0:
                    inputs[t] = tf.nn.dropout(inputs[t], hps.keep_prob)

            #with tf.control_dependencies([self.initial_states.assign(state)]):   # for bi-lstm? or two layer lstm 
            #    inputs[t] = tf.identity(inputs[t])
       
        self.final_state = state
        # [batch_size*steps,projected_size]
        inputs = tf.reshape(tf.concat(inputs,1), [-1, hps.projected_size])

        # Initialization ignores the fact that softmax_w is transposed. That worked slightly better.
        softmax_w = sharded_variable("softmax_w", [hps.vocab_size, hps.projected_size], hps.num_shards)
        softmax_b = tf.get_variable("softmax_b", [hps.vocab_size])

        if hps.num_sampled == 0:
            full_softmax_w = tf.reshape(tf.concat(softmax_w,1), [-1, hps.projected_size])
            full_softmax_w = full_softmax_w[:hps.vocab_size, :]
            
            # [batch_size*steps,vocab]
            logits = tf.matmul(inputs, full_softmax_w, transpose_b=True) + softmax_b
            print("train log logits:", logits)
            # targets = tf.reshape(tf.transpose(self.y), [-1])
            # index = tf.argmax(logits,axis=1)
            
            _, index = tf.nn.top_k(logits, hps.arg_max)
            print "index:",index 
            
            targets = tf.reshape(y, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,logits=logits)
            loss = tf.reduce_mean(loss * tf.reshape(w, [-1]))

            if mode == "predict_next":
                return loss, logits, index

        else:
            targets = tf.reshape(y, [-1, 1])
            
            loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, targets, tf.to_float(inputs),
                                              hps.num_sampled, hps.vocab_size)
            
        loss = tf.reduce_mean(loss * tf.reshape(w, [-1]))
        return loss

    def _backward(self, loss, summaries=False):
        hps = self.hps

        loss = loss * hps.num_steps

        emb_vars = find_trainable_variables("emb")
        lstm_vars = find_trainable_variables("LSTM")
        softmax_vars = find_trainable_variables("softmax")
        all_vars = emb_vars + lstm_vars + softmax_vars
        grads = tf.gradients(loss, all_vars)
        orig_grads = grads[:]
        emb_grads = grads[:len(emb_vars)]
        grads = grads[len(emb_vars):]
        for i in range(len(emb_grads)):
            #print "emb_grads:",emb_grads[i]
            assert isinstance(emb_grads[i], tf.IndexedSlices)
            emb_grads[i] = tf.IndexedSlices(emb_grads[i].values * hps.batch_size, emb_grads[i].indices,
                                            emb_grads[i].dense_shape)
         

        lstm_grads = grads[:len(lstm_vars)]
        softmax_grads = grads[len(lstm_vars):]
        lstm_grads, lstm_norm = tf.clip_by_global_norm(lstm_grads, hps.max_grad_norm)
        clipped_grads = emb_grads + lstm_grads + softmax_grads
        assert len(clipped_grads) == len(orig_grads)

        if summaries:
            tf.summary.scalar("model/lstm_grad_norm", lstm_norm)
            tf.summary.scalar("model/lstm_grad_scale", tf.minimum(hps.max_grad_norm / lstm_norm, 1.0))
            tf.summary.scalar("model/lstm_weight_norm", tf.global_norm(lstm_vars))
            # for v, g, cg in zip(all_vars, orig_grads, clipped_grads):
            #     name = v.name.lstrip("model/")
            #     tf.histogram_summary(name + "/var", v)
            #     tf.histogram_summary(name + "/grad", g)
            #     tf.histogram_summary(name + "/clipped_grad", cg)

        return list(zip(clipped_grads, all_vars))

    @staticmethod
    def get_default_hparams():
        return HParams(
            batch_size= 1,  # 256 
            num_steps = 1,  # 20
            num_shards= 8,
            num_layers=1,
            learning_rate=0.2,
            max_grad_norm=10.0,
            num_delayed_steps=150,
            keep_prob=0.9,
            vocab_size=80000,
            emb_size=512,
            state_size=2048,
            projected_size=512,
            num_sampled=8192,
            num_gpus=1,
            arg_max=10,
            average_params=True,
            run_profiler=False,
        )
