import datrie 
from data_utils import Vocabulary, Dataset
import string
import re 
from flask import Flask
from flask_restful import Resource, Api
import traceback
import time
import sys
#import thriftpy
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from newPyClient import computeKSR
import json
import numpy as np
import time
import tensorflow as tf
from data_utils import Vocabulary, Dataset
from language_model import LM
from common import CheckpointLoader
import heapq

UPLOAD_FOLDER = '/data/ngramTest/uploads'
UPLOAD_FOLDER = './'

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
ckpt_loader = CheckpointLoader(saver, model.global_step,  "log.txt/train")             
saver.restore(sess,"log.txt/train/model.ckpt-742996")
app = Flask(__name__)
api = Api(app)


'''
#build vocab  trie
trie = datrie.new(string.printable)

cnt = 0
vocab_size = 140000
for i in range(vocab_size):
    word = vocab.get_token(i)
    trie[word] = i

for key in trie.keys(u"pre"):
    print key,trie[key]
trie.save("data/vocab_trie")                                                        
'''
trie = datrie.Trie.load("data/vocab_trie")



class ngramPredict(Resource):
    def get(self,input):
        input = input.decode("utf-8")
        #print "input:",input
        input_words = input
        if input_words.find('<S>')!=0:
            input_words = '<S> ' + input
        isCompletion = False
        if input_words[-1] == ' ':
            #print "Predict:"
            prefix_input = [vocab.get_id(w) for w in input_words.split()]
        else:
            #print "Compeletion:"
            isCompletion = True
            prefix_input = [vocab.get_id(w) for w in input_words.split()[:-1]]
            prefix = input_words.split()[-1]
            #print "prefix:",prefix,type(prefix)
        #print("input:",input,"pre:",prefix_input,"len:",len(prefix_input))
        w = np.zeros([1, len(prefix_input)], np.uint8)
        w[:] =1 
        inputs = np.zeros([hps.batch_size*hps.num_gpus,hps.num_steps])
        weights = np.zeros([hps.batch_size*hps.num_gpus,hps.num_steps])
        inputs[0,:len(prefix_input)] = prefix_input[:]
        weights[0,:len(prefix_input)] = w[:]
        words = []
        with sess.as_default():
            #ckpt_loader.load_checkpoint()  #  FOR ONLY ONE CHECKPOINT 
            sess.run(tf.local_variables_initializer())
            words = []
            if not isCompletion:
                indexes = sess.run([model.index],{model.x:inputs, model.w:weights})
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
                prob = sess.run([model.logits],{model.x:inputs, model.w:weights})
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
        #print words
        return words[:top_k]

@app.route('/ngramfile/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        doc = request.json
        if doc:
            doc = doc['text']
            ngramClient = ngramPredict()
            res = computeKSR(ngramClient,doc) 
            return json.dumps(res)

        if 'text' not in request.files:
            return "{\"ret\":-1}"
        file = request.files['text']
        if file.filename == '':
            return "{\"ret\":-2}"
        filename = secure_filename(file.filename)
        uploadFilePath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(uploadFilePath)
        doc = ""
        with open(uploadFilePath, 'rb') as textFile:
            doc = textFile.read()
        ngramClient = ngramPredict()
        res = computeKSR(ngramClient,doc) 
        #print("res:",res)
        #TODO
        #return json.dumps(res)



api.add_resource(ngramPredict, '/ngram/<input>')
#predictClient = PredictClient()


if __name__ == '__main__':
    '''
    ngrampredict  =  ngramPredict()
    ngrampredict.get("how are")
    ngrampredict.get("what the")
    ngrampredict.get("i am") 
    ngrampredict.get("how do") 
    '''
    #print('test for grep ksr')
    app.run(host = "0",port=9898)





