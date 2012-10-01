from nltk.corpus import conll2002, conll2000
import sys
from feature_generator import SimpleFeatureGenerator
from sentence import sentence_from_conll
from structured_perceptron import StructuredPerceptron

train = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('ned.train')]

#We are using simple features:
print >>sys.stderr, "Extracting features from corpus..."
feature_generator = SimpleFeatureGenerator(train)

#Create the Perceptron
print >>sys.stderr, "Creating the Perceptron..."
perceptron = StructuredPerceptron(['LOC', 'PER', 'ORG', 'MISC', 'O'], feature_generator)

#Train it!
print >>sys.stderr, "Training the Perceptron..."
perceptron.train(train)

#Test it!
test = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('ned.testb')]
perceptron.test(test)