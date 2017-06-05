# -*- coding: utf-8 -*-
import thriftpy
import time
import re,json
import urllib2
from thriftpy.rpc import make_server
#import tensorflow as tf
#from data_utils import Vocabulary, Dataset
#from language_model import LM
#from common import CheckpointLoader
import numpy as np
interface_thrift = thriftpy.load("interface.thrift",
                                module_name="interface_thrift")
'''
hps = LM.get_default_hparams()
vocab = Vocabulary.from_file("1b_word_vocab.txt")
with tf.variable_scope("model"):                                                        
    hps.num_sampled = 0  # Always using full softmax at evaluation.   run out of memory 
    hps.keep_prob = 1.0       
    hps.num_gpus = 1
    model = LM(hps,"predict_next","/cpu:0")                                            

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
'''
class PredictHandler(object):
    def __init__(self):
        try:
            '''
            self.hps = LM.get_default_hparams()
            self.vocab = Vocabulary.from_file("1b_word_vocab.txt")
            hps = self.hps
            with tf.variable_scope("model"):                                    
                hps.num_sampled = 0  # Always using full softmax at evaluation.   run out of memory 
                hps.keep_prob = 1.0       
                hps.num_gpus = 1
                self.model = LM(hps,"predict_next","/cpu:0")                                            
                
            if hps.average_params:                                                                  
                print("Averaging parameters for evaluation.")                                       
                saver = tf.train.Saver(self.model.avg_dict)                                              
            else:                                                                                   
                saver = tf.train.Saver()                                                                
                                                                                    
            # Use only 4 threads for the evaluation.                                                
            
            config = tf.ConfigProto(allow_soft_placement=True,                                      
                            intra_op_parallelism_threads=200,                                
                            inter_op_parallelism_threads=100)                                 
            self.sess = tf.Session(config=config)                                                        
            ckpt_loader = CheckpointLoader(saver, self.model.global_step,  "log.txt/train")             
            saver.restore(self.sess,"log.txt/train/model.ckpt-742996")
            self.log = {}
            print("LOCAL VARIABLES")
            for v in tf.local_variables():
                print("%s %s %s" % (v.name, v.get_shape(), v.device))
            '''
            self.log = {}
        except Exception as e:
            print e
    
    def getPrediction(self,sWord,sLocale,sAppName):
        #def getPrediction(self,sWord):
        #import pdb
        #pdb.set_trace()
        try :
            '''
            hps = LM.get_default_hparams()
            vocab = Vocabulary.from_file("1b_word_vocab.txt")
            #hps = self.hps
            with tf.variable_scope("model"):                                    
                hps.num_sampled = 0  # Always using full softmax at evaluation.   run out of memory 
                hps.keep_prob = 1.0       
                hps.num_gpus = 1
                model = LM(hps,"predict_next","/cpu:0")                                            
                
            if hps.average_params:                                                                  
                print("Averaging parameters for evaluation.")                                       
                saver = tf.train.Saver(model.avg_dict)                                              
            else:                                                                                   
                saver = tf.train.Saver()                                                                
                                                                                    
            # Use only 4 threads for the evaluation.                                                
            
            config = tf.ConfigProto(allow_soft_placement=True,                                      
                            intra_op_parallelism_threads=200,                                
                            inter_op_parallelism_threads=100)                                 
            sess = tf.Session(config=config)                                                        
            ckpt_loader = CheckpointLoader(saver, model.global_step,  "log.txt/train")             
            saver.restore(sess,"log.txt/train/model.ckpt-742996")
            #self.log = {}
            print("LOCAL VARIABLES")
            for v in tf.local_variables():
                print("%s %s %s" % (v.name, v.get_shape(), v.device))
            '''
            
            '''
            #load model
            #vocab = self.vocab
            #hps = self.hps
            #sess = self.sess
            #model = self.model
            print "word test:", vocab.get_token(0)
            input_words = sWord
            if input_words.find('<S>')!=0:
                input_words = '<S> ' + input_words
            prefix_input = [vocab.get_id(w) for w in input_words.split()]
            #print("input:",input,"pre:",prefix_input,"len:",len(prefix_input))
            print hps.num_gpus,"gpus"
            inputs = np.zeros([hps.batch_size*hps.num_gpus,hps.num_steps])
            weights = np.zeros([hps.batch_size*hps.num_gpus,hps.num_steps])
            inputs[0,:len(prefix_input)] = prefix_input[:]
            weights[0,:] = 1
            words = []
            print("LOCAL VARIABLES")
            for v in tf.local_variables():
                print("%s %s %s" % (v.name, v.get_shape(), v.device))
            with sess.as_default():
                #ckpt_loader.load_checkpoint()  #  FOR ONLY ONE CHECKPOINT 
                sess.run(tf.local_variables_initializer())
                #self.sess.run(tf.global_variables_initializer())
                indexes = sess.run([model.index],{model.x:inputs, model.w:weights})
                indexes = np.reshape(indexes,[hps.num_steps,hps.arg_max])
                words = []
                for j in range(hps.arg_max):
                    #print j
                    word = vocab.get_token(indexes[len(prefix_input)-1][j])
                    words += [word]
                #print words
            print("words:",words)
            '''
            #TODO: ADD LSTM PREDICT
            start = time.time() 
            pre = ""
            plen = len(sWord.split())
            for i,word in enumerate(sWord.split()):
                if len(word)>50:
                    break
                pre += word
                if i>=18:
                    continue
                pre += "%20"
            #print 'http://10.60.118.70:5000/ngram/'+pre
            if not sWord[-1]==" ":
                pre = pre[:-3]
            ret = urllib2.urlopen('http://10.60.118.70:9898/ngram/'+pre)
            #import pdb
            #pdb.set_trace()
            #print ret
            ret= ret.read()
            print ret
            #ret =  strhtml.read()
            '''
            pattern = re.compile('\s')
            res = []
            for predict in  ret.split(','):
                res += [pattern.sub("",predict)]
            '''
            
            result = interface_thrift.Result()
            result.timeUsed = time.time()-start
            print result.timeUsed
            result.sEngineTimeInfo = "1:0,3:0"
            #result.listWords = self.ltmClient.get(sWord)
            result.listWords = json.loads(ret)
            
            #result.listWords = ['word']
            return result
        except Exception as  e:  
            print e
        #return result


def main():
    ip = "0"
    port = 9090
    server = make_server(interface_thrift.Suggestion, PredictHandler(),
                         ip, port)
    print("serving...",ip, port)
    server.serve()


if __name__ == '__main__':
    main()
    lstm = PredictHandler()
    res = lstm.getPrediction("how are you","","")
    print res 
    res = lstm.getPrediction("as soon as","","")
    print res 
