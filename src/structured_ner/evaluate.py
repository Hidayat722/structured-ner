from nltk.corpus import conll2002
from features import ExtendedFeatureGenerator
from sentence import sentence_from_conll
from train_supervised import train_ner

conll2002_labels = ['LOC', 'PER', 'ORG', 'MISC', 'O']

dutch_train     = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('ned.train')]
dutch_test      = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('ned.testb')]
dutch_ner       = train_ner(conll2002_labels, dutch_train, ExtendedFeatureGenerator)

spanish_train   = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('esp.train')]
spanish_test    = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('esp.testb')]
spanish_ner     = train_ner(conll2002_labels, spanish_train, ExtendedFeatureGenerator)

dutch_ner.test(dutch_test)
spanish_ner.test(spanish_test)