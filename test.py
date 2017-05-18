import tensorflow as tf
from data_utils import Vocabulary, Dataset
from language_model import LM
from common import CheckpointLoader


BATCH_SIZE = 1
NUM_TIMESTEPS = 1
MAX_WORD_LEN = 50

UPLOAD_FOLDER = '/data/ngramTest/uploads'
UPLOAD_FOLDER = './'


hps = LM.get_default_hparams()
vocab = Vocabulary.from_file("1b_word_vocab.txt")
with tf.variable_scope("model"):                                                        
    hps.num_sampled = 0  # Always using full softmax at evaluation.   run out of memory 
    hps.keep_prob = 1.0       
    hps.num_gpus = 1
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
ckpt_loader = CheckpointLoader(saver, model.global_step,  "log.txt/train")             
saver.restore(sess,"log.txt/train/model.ckpt-742996")

