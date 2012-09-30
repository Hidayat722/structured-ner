from nltk.corpus import conll2002, conll2000
from feature_generator import SimpleFeatureGenerator
from sentence import Sentence, sentence_from_conll
from structured_perceptron import StructuredPerceptron

#We are using simple features:
feature_generator = SimpleFeatureGenerator()

#Create the Perceptron
perceptron = StructuredPerceptron(conll2002._chunk_types, feature_generator)

#Train it!
perceptron.train([sentence_from_conll(sent) for sent in conll2002.chunked_sents('ned.train')])

