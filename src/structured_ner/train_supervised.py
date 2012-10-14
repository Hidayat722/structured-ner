import random
import sys
from structured_perceptron import StructuredPerceptron

def train_ner(labels, data, feature_generator):

    print >>sys.stderr, "Extracting features from corpus..."
    feature_generator = feature_generator(data)

    #Create the Perceptron
    print >>sys.stderr, "Creating the Perceptron..."
    perceptron = StructuredPerceptron(labels, feature_generator, epochs=1)

    #Train it!
    print >>sys.stderr, "Training the Perceptron..."
    random.shuffle(data)
    perceptron.train(data)

    return perceptron