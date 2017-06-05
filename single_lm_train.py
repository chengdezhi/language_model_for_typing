import tensorflow as tf

from data_utils import Vocabulary, Dataset
from language_model import LM
from run_utils import run_train, run_eval, predict_next

flags = tf.flags
flags.DEFINE_string("logdir", "cv_v", "Logging directory.")
flags.DEFINE_string("datadir", "/data1/cdz/1b_data/", "Logging directory.")
flags.DEFINE_string("mode", "eval_test", "Whether to run 'train' or 'eval' model.")
flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
flags.DEFINE_integer("num_gpus", 2, "Number of GPUs used.")
flags.DEFINE_integer("eval_steps", 10000, "Number of eval steps.")
FLAGS = flags.FLAGS

def main(_):
    hps = LM.get_default_hparams().parse(FLAGS.hpconfig)
    hps.num_gpus = FLAGS.num_gpus
    
    vocab = Vocabulary.from_file(FLAGS.datadir + "/lm_vocab.txt", hps.vocab_size)

    if FLAGS.mode == "train":
        hps.batch_size = 256  # reset batchsize
        dataset = Dataset(vocab, FLAGS.datadir + "/train/*")
        run_train(dataset, hps, FLAGS.logdir + "/train", ps_device="/gpu:0")
    elif FLAGS.mode.startswith("eval_"):
        if FLAGS.mode.startswith("eval_train"):
            data_dir = FLAGS.datadir + "/train/*"
        elif FLAGS.mode.startswith("eval_test"):
            data_dir = FLAGS.datadir + "/heldout/*"
        print("data_dir:",data_dir)
        dataset = Dataset(vocab, data_dir, deterministic=True)
        run_eval(dataset, hps, FLAGS.logdir, FLAGS.mode, FLAGS.eval_steps)
    elif  FLAGS.mode.startswith("predict_next"):
        data_dir = "data/news.en.heldout-00001-of-00050"
        dataset = Dataset(vocab, data_dir)
        predict_next(dataset, hps, FLAGS.logdir, FLAGS.mode, FLAGS.eval_steps,vocab) 
 

if __name__ == "__main__":
    tf.app.run()
