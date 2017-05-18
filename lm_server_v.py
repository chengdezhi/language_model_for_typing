# -*- coding: utf-8 -*-
import datrie, heapq
import thriftpy
import time
import re,json
import urllib2
from thriftpy.rpc import make_server
import tensorflow as tf
from data_utils import Vocabulary, Dataset
from language_model import LM
from common import CheckpointLoader
import numpy as np

interface_thrift = thriftpy.load("interface.thrift",
                                module_name="interface_thrift")
#import pdb
#pdb.set_trace()
top_k = 3
pattern = re.compile('[\w+]')
p_punc  = re.compile('(\.|\"|,|\?|\!)')
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
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)                                                        
saver.restore(sess,"log.txt/train/model.ckpt-742996")
#sess.run(tf.local_variables_initializer())
trie = datrie.Trie.load("data/vocab_trie")
init = tf.local_variables_initializer()

class PredictHandler(object):
    def __init__(self):
        try:
            self.log = {}
        except Exception as e:
            print e
    
    def getPrediction(self,sWord,sLocale,sAppName):
        #def getPrediction(self,sWord):
        #import pdb
        #pdb.set_trace()
        try :
            start = time.time()
            input_words = sWord.decode("utf-8")
            if input_words.find('<S>')!=0:
                input_words = '<S> ' + input_words
            isCompletion = False
            if input_words[-1] == ' ':
                #print "Predict:"
                prefix_input = [vocab.get_id(w) for w in input_words.split()[:20]]
            else:
                #print "Compeletion:"
                isCompletion = True
                prefix_input = [vocab.get_id(w) for w in input_words.split()[:-1][:20]]
                prefix = input_words.split()[-1]
                #print "prefix:",prefix,type(prefix)
            print("input:",sWord,"pre:",prefix_input,"len:",len(prefix_input))
            w = np.zeros([1, len(prefix_input)], np.uint8)
            w[:] =1 
            inputs = np.zeros([hps.batch_size*hps.num_gpus,hps.num_steps])
            weights = np.zeros([hps.batch_size*hps.num_gpus,hps.num_steps])
            inputs[0,:len(prefix_input)] = prefix_input[:]
            weights[0,:len(prefix_input)] = w[:]
            with sess.as_default():
                #ckpt_loader.load_checkpoint()  #  FOR ONLY ONE CHECKPOINT 
                #sess.run(tf.local_variables_initializer())
                #sess.run(tf.initialize_all_variables())
                #sess.run(tf.local_variables_initializer())
                words = []
                w_prob = []
                if not isCompletion:
                    _, indexes = sess.run([init, model.index],{model.x:inputs, model.w:weights})
                    indexes = np.reshape(indexes,[hps.num_steps,hps.arg_max])
                    for j in range(hps.arg_max):
                        word = vocab.get_token(indexes[len(prefix_input)-1][j])
                        if not p_punc.match(word)==None:
                            words += [word]
                            continue
                        if pattern.match(word)==None:
                            continue
                        words += [word]
                else:   
                    _, prob = sess.run([init, model.logits],{model.x:inputs, model.w:weights})
                    prob = np.reshape(prob,[hps.num_steps,hps.vocab_size])
                    prob = prob[len(prefix_input)-1]   # the last prefix_input step prob is the predict one 
                    #print "prob:", len(prob)
                    #print "prefix:",trie.keys(prefix)
                    cand = [trie[cand_index] for cand_index in trie.keys(prefix)] 
                    #print "cand:", cand
                    #print "prefix:", prefix
                    cand_prob = [prob[pb] for pb in cand]
                    ins = heapq.nlargest(top_k, range(len(cand_prob)), cand_prob.__getitem__)
                    for j in ins:
                        word = vocab.get_token(cand[j])
                        words += [word]
                        w_prob += [cand_prob[j]]
            #print words
            print words[:top_k],w_prob[:top_k]
            #TODO: ADD LSTM PREDICT
            
            result = interface_thrift.Result()
            result.timeUsed = time.time()-start
            print result.timeUsed
            result.sEngineTimeInfo = "1:0,3:0"
            #result.listWords = self.ltmClient.get(sWord)
            result.listWords = words[:top_k] 
            
            #result.listWords = ['word']
            return result
        except Exception as  e:  
            print e
        #return result


def main():
    ip = "0"
    port = 9090
    lstm = PredictHandler()
    server = make_server(interface_thrift.Suggestion, lstm,
                         ip, port)
    print("serving...",ip, port)
    server.serve()


if __name__ == '__main__':
    main()
    lstm = PredictHandler()
    res = lstm.getPrediction("as soon as p","","")
    print res 
    #init = tf.local_variables_initializer()
    res = lstm.getPrediction("how are you ","","")
    print res 
    res = lstm.getPrediction("as soon as p","","")
    print res 
