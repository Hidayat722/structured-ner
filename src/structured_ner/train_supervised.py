import codecs
import random
import sys

from features.feature_generator import FeatureGenerator
from structured_perceptron import StructuredPerceptron

def train_ner(lang, labels, data, heldout, feature_sets, verbose=False):

    print >>sys.stderr, "Extracting features from corpus..."
    feature_generator = FeatureGenerator(data, feature_sets)

    #Create the Perceptron
    print >>sys.stderr, "Creating the Perceptron..."
    perceptron = StructuredPerceptron(labels, feature_generator, epochs=2)

    #Train it!
    print >>sys.stderr, "Training the Perceptron..."
    random.shuffle(data)

    run_label = "%s_%s" % (lang, '+'.join(map(lambda fs: fs.__class__.__name__, feature_sets)))
    perceptron.train(data, heldout, verbose=verbose, run_label=run_label)

    if verbose:
        codecs.open('../eval/%s_features.csv' % run_label, 'w', encoding='utf-8').write('\n'.join(perceptron.evaluate_features()))

    return perceptron