import pickle
from nltk.corpus import conll2002, conll2000
import sys
from sentence import sentence_from_conll
from features import ExtendedFeatureGenerator
from structured_perceptron import StructuredPerceptron

train = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('ned.train')]

#We are using simple features:
print >>sys.stderr, "Extracting features from corpus..."
feature_generator = ExtendedFeatureGenerator(train)

#Create the Perceptron
print >>sys.stderr, "Creating the Perceptron..."
perceptron = StructuredPerceptron(['LOC', 'PER', 'ORG', 'MISC', 'O'], feature_generator, epochs=5)

#Train it!
print >>sys.stderr, "Training the Perceptron..."
perceptron.train(train)

pickle.dump(perceptron, open("ner_model.pickle", 'w'))

#Test it!
test = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('ned.testb')]
perceptron.test(test)