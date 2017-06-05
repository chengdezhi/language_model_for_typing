from data_utils import Vocabulary
import datrie
vocab = Vocabulary.from_file("1b_word_vocab.txt")
trie = datrie.Trie.load("data/vocab_trie")
print vocab.get_token(238)

print trie.keys(u'F')
