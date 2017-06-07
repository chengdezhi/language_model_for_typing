# -*- coding: utf-8 -*-
import datrie, heapq
import thriftpy
import time
import re,json
import urllib2
from thriftpy.rpc import make_server
import tensorflow as tf
from data_utils import Vocabulary, Dataset
from language_model import LM, inference_graph
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
st  =  hps.num_steps


with tf.variable_scope("model"):                                                        
    hps.vocab_size = 793470
    hps.num_sampled = 0  # Always using full softmax at evaluation.   run out of memory 
    hps.keep_prob = 1.0       
    hps.num_gpus = 1 
    model = inference_graph(hps)
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
sw = tf.summary.FileWriter("cv", sess.graph)
saver.restore(sess,"log.txt/train/model.ckpt-742996")
#sess.run(tf.local_variables_initializer())
trie = datrie.Trie.load("data/vocab_trie")
init = tf.local_variables_initializer()
sess.run(init)


# additional graph
logits_cache = tf.get_variable("logits_cache", hps.vocab_size)
logits_bos = tf.get_variable("logits_bos", hps.vocab_size)
tf_inds = tf.placeholder(tf.int32)
tf_ind_len = tf.placeholder(tf.int32, name="ind_len")
tf_top_k = tf.minimum(tf_ind_len, hps.arg_max)
assign_cache = tf.assign(logits_cache, model.logits)
assign_bos = tf.assign(logits_cache, logits_bos)
_,partial_top_k = tf.nn.top_k(tf.gather(logits_cache, tf_inds), tf_top_k)

import pylru
size = 100
cache = pylru.lrucache(size)

# warm up 
with sess.as_default():
    x = vocab.s_id
    #x = np.array(x)
    x = np.reshape(x,[1])
    print "x:",x
    init_rnn_state = sess.run(model.initial_state)
    init_rnn_state = sess.run(model.final_state, {model.x:x, model.initial_state:init_rnn_state})
    print "init_rnn_state:", init_rnn_state

def  word2id(word):
    return vocab.get_id(word)

def  predict(instr):
    words = instr.split()
    if " ".join(words[:-1]) in cache:
        state = cache[" ".join(words[:-1])]
        x_words = [words[-1]]
    else:
        state = init_rnn_state
        x_words = words
    print "x_words:",x_words
    x_len = len(x_words)
    for i,word in enumerate(x_words):
        x = word2id(word)
        x = np.reshape(x,[1])
        if i!=x_len-1:
            with sess.as_default():
                state, _ = sess.run([model.final_state,assign_cache],{model.x:x,model.initial_state:state})
        else:
            with sess.as_default():
                state, _, top_k_index =  sess.run([model.final_state,assign_cache,model.index], {model.x:x, model.initial_state:state})
    cache[" ".join(words)] = state
    preds = [vocab.get_token(ix) for ix in top_k_index]
    return preds

def  complete(instr):
    words = instr.strip().split()
    compl = words[-1]
    compl_ind = [trie[cand_index] for cand_index in trie.keys(compl)]
    compl_ind_len = np.array(len(compl_ind))
    compl_ind = np.array(compl_ind)
    words = words[:-1]

    if " ".join(words) in cache:
        state = cache[" ".join(words)]
    else:
        predict(" ".join(words))

    with sess.as_default():
        top_k_index = sess.run(partial_top_k,{tf_inds:compl_ind, tf_ind_len:compl_ind_len})
    preds = [vocab.get_token(compl_ind[ix]) for ix in top_k_index]
    return preds


def  query_string(instr):
    print "instr:", instr, type(instr)
    if  type(instr) != unicode or instr.strip() == "":
        return
    print "instr:",instr
    if instr[-1]==" ":
        print "instr:", instr
        return predict(instr)
    else:
        return complete(instr)

class PredictHandler(object):
    def __init__(self):
        try:
            self.log = {}
        except Exception as e:
            print e

    def getPrediction(self, sWord, sLocale, sAppName):
        try:
            start = time.time()
            sWord = sWord.decode("utf-8")
            words = query_string(sWord)
            if not words:
                words = []
            result = interface_thrift.Result()
            result.timeUsed = time.time() - start
            print "result_time_used:",result.timeUsed
            print "words:",words
            result.sEngineTimeInfo = "1:0,3:0"
            # result.listWords = self.ltmClient.get(sWord)
            result.listWords = words
            return result
        except Exception as e:
            print e

    def getPrediction_old(self,sWord,sLocale,sAppName):
        #def getPrediction(self,sWord):
        #import pdb
        #pdb.set_trace()
        try :
            print "sWord:",sWord
            start = time.time()
            input_words = sWord.decode("utf-8")
            if input_words.find('<S>')!=0:
                input_words = '<S> ' + input_words
            isCompletion = False
            if input_words[-1] == ' ':
                print "Predict:"
                prefix_input = [vocab.get_id(w) for w in input_words.split()]
            else:
                print "Compeletion:"
                isCompletion = True
                prefix_input = [vocab.get_id(w) for w in input_words.split()[:-1]]
                prefix = input_words.split()[-1]
                print "prefix:",prefix,type(prefix)
            print("input:",sWord,"pre:",prefix_input,"len:",len(prefix_input))
            input_len = len(prefix_input)
            inputs_v = np.zeros([hps.batch_size*hps.num_gpus, hps.num_steps])
            ''' 
            inputs = np.zeros([hps.batch_size*hps.num_gpus, input_len])
            weights = np.zeros([hps.batch_size*hps.num_gpus, input_len])
            inputs[0,:len(prefix_input)] = prefix_input[:]
            weights[0,:len(prefix_input)] = w[:]
            print "input_len:", input_len
            '''
            with sess.as_default():
                #ckpt_loader.load_checkpoint()  #  FOR ONLY ONE CHECKPOINT 
                #sess.run(tf.local_variables_initializer())
                #sess.run(tf.initialize_all_variables())
                #sess.run(tf.local_variables_initializer())
                words = []
                w_prob = []

                
                init_rnn_state = sess.run(model.initial_states)
                step = 0
                while step + st < input_len:
                    #inputs_v[0,:]  =  inputs[0,step:step+st]
                    inputs_v[0,:] =   prefix_input[step:step+st]
                    #index = sess.run([model.index],{model.x:inputs_v, model.w:weights_v})
                    init_rnn_state = sess.run(model.final_state,{model.x:inputs_v, model.initial_states:init_rnn_state})
                    step += st
                #print type(state),state[-1][-1][-1][-1]
                
                inputs_v[0,:input_len-step] = prefix_input[step:input_len]
                if not isCompletion:
                    #TODO  optimizeir
                    start = time.time()
                    t_logits = sess.run(model.logits, {model.x:inputs_v, model.initial_states:init_rnn_state})
                    end = time.time()
                    print "logits_time:", end -start
                    start = time.time()
                    indexes = sess.run(model.index, {model.x:inputs_v, model.initial_states:init_rnn_state})
                    end = time.time()
                    print "index_time:", end - start
                    indexes = np.reshape(indexes,[hps.num_steps,hps.arg_max])
                    for j in range(hps.arg_max):
                        word = vocab.get_token(indexes[len(prefix_input)-1-step][j])
                        if not p_punc.match(word)==None:
                            words += [word]
                            continue
                        if pattern.match(word)==None:
                            continue
                        words += [word]
                    """
                    prob = sess.run([model.logits],{model.x:inputs_v, model.w:weights_v})                                            
                    prob = np.reshape(prob,[hps.num_steps, hps.vocab_size])                                        
                    prob = prob[len(prefix_input)-1-step]   # the last prefix_input step prob is the predict one  
                    ins  = heapq.nlargest(top_k+3, range(len(prob)), prob.__getitem__)
                    print "ins:",ins
                    for j in xrange(top_k+3):
                        word = vocab.get_token(ins[j])
                        if not p_punc.match(word)==None:    # reserve punc for next word
                            words += [word]
                            continue
                        if pattern.match(word)==None:
                            continue
                        words += [word]
                    """
                else:   
                    start = time.time() 
                    cand = [trie[cand_index] for cand_index in trie.keys(prefix)] 
                    ind_len = np.array(len(cand))
                    if ind_len <= 1:
                        words += [prefix]
                    else:
                        cand  = np.array(cand)
                        #print ind_len, "ind_len"
                        #topk = sess.run(model.top_k,{model.ind_len:ind_len})
                        #print topk ,"top_k"
                        index, topk = sess.run([model.ind_index,model.top_k],{model.x:inputs_v,  model.initial_states:init_rnn_state, model.ind: cand, model.ind_len:ind_len})
                        index = np.reshape(index,[hps.num_steps, topk])
                        print "index:",index
                        words = []
                        for j in index[0] :
                            word = vocab.get_token(cand[j])
                            words += [word]
                        """
                        # not add ind in the graph
                        prob = sess.run([model.logits],{model.x:inputs_v, model.w:weights_v, model.initial_states:init_rnn_state, })
                        prob = np.reshape(prob,[hps.num_steps,hps.vocab_size])
                        prob = prob[len(prefix_input)-1-step]   # the last prefix_input step prob is the predict one 
                        #print "prob:", len(prob)
                        #print "prefix:",trie.keys(prefix)
                        cand = [trie[cand_index] for cand_index in trie.keys(prefix)] 
                        #print "cand:", cand
                        #print "prefix:", prefix
                        cand_prob = [prob[pb] for pb in cand]
                    
                        #cand_prob = np.array(cand_prob)
                        #ind = np.argpartition(cand_prob,-top_k)[-top_k:]
                        #ins = ind[np.argsort(cand_prob[ind])][::-1]
                        
                        ins = heapq.nlargest(top_k, range(len(cand_prob)), cand_prob.__getitem__)
                        for j in ins:
                            word = vocab.get_token(cand[j])
                            words += [word]
                            w_prob += [cand_prob[j]]
                        """
            #print words
            print words[:top_k] # ,w_prob[:top_k]
            #TODO: ADD LSTM PREDICT
            
            result = interface_thrift.Result()
            result.timeUsed = time.time()-start
            print result.timeUsed
            result.sEngineTimeInfo = "1:0,3:0"
            #result.listWords = self.ltmClient.get(sWord)
            result.listWords = words[:top_k] 
            
            #result.listWords = ['word']
            return result
        except Exception as e:  
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
    """
    # for test 
    lstm = PredictHandler()
    res = lstm.getPrediction("long before the advent of e-commerce Wal-Mart's founder Sam Walton set out his vision for a successful retail operation We let folks know we're interested in them and that they're vital to us-- cause they are he said","","")
    res = lstm.getPrediction("Having a little flexibility on that issue would go a long way to putting together a final  once upon a  ","","")
    print res 
    #init = tf.local_variables_initializer()
    res = lstm.getPrediction("once upon a  ","","")
    print res 
    res = lstm.getPrediction("as ? soon as p","","")
    res = lstm.getPrediction("as ? soon as p","","")
    res = lstm.getPrediction("as ? soon as p","","")
    res = lstm.getPrediction("as ? soon as p","","")
    res = lstm.getPrediction("as ? soon as a","","")
    print res 
    """
