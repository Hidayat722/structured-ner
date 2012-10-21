import codecs
import pickle
from nltk.corpus.reader.conll import ConllChunkCorpusReader
from nltk.corpus.util import LazyCorpusLoader
from nltk.stem.wordnet import WordNetLemmatizer
from sentence import load_conll
from case.MosesTrueCaser import MosesTrueCaser

lemmatizer = WordNetLemmatizer()
truecaser  = MosesTrueCaser(open('models/truecase/truecase-model.en'))

m = pickle.load(open("models/_eng_gaz.pickle"))

corpus = LazyCorpusLoader('conll2003', ConllChunkCorpusReader, '.*\.(test|train).*', ('LOC', 'PER', 'ORG', 'MISC'), encoding='utf-8')
test    = load_conll(corpus.chunked_sents('eng.testa'), lemmatizer, truecaser)

out, acc = m.test(test)

codecs.open('test.txt', 'w', encoding='utf-8').write(out)
