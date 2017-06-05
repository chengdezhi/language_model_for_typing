from data_utils import Vocabulary, Dataset
import datrie,string
vocab = Vocabulary.from_file("1b_word_vocab.txt")
#build vocab  trie                                                                   
trie = datrie.new(string.ascii_lowercase)                                            
vocab_size = 100001                                                                  
cnt = 0                                                                              
for i in range(vocab_size):                                                          
    word = vocab.get_token(i)                                                        
    if word[0]=='<':                                                                 
        continue                                                                     
    #if pattern.match(word)==None:                                                   
    #    continue                                                                    
    trie[word] = i                                                                   
                                                                                     
for key in trie.keys(u"pre"):                                                        
    print key,trie[key]                                                              
trie.save("data/vocab_trie")                                                         
assert u"china" in trie
