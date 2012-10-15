import codecs
import random
import sys

from features.feature_generator import FeatureGenerator
from structured_perceptron import StructuredPerceptron

def train_ner(lang, labels, data, heldout, test, feature_sets, verbose=False, additional_run_label=None):

    print >>sys.stderr, '\n' * 2

    print >>sys.stderr, '-' * 60
    print >>sys.stderr, "Training: %s (%s)" % (lang, '+'.join(map(lambda fs: fs.__class__.__name__, feature_sets)))
    print >>sys.stderr, '-' * 60

    print >>sys.stderr, "Extracting features from corpus..."
    feature_generator = FeatureGenerator(data, feature_sets)

    #Create the Perceptron
    print >>sys.stderr, "Creating the Perceptron..."
    perceptron = StructuredPerceptron(labels, feature_generator, epochs=10)

    #Train it!
    print >>sys.stderr, "Training the Perceptron..."
    random.shuffle(data)

    run_label = "%s_%s" % (lang, '+'.join(map(lambda fs: fs.__class__.__name__, feature_sets)))

    if additional_run_label:
        run_label += "_" + additional_run_label

    perceptron.train(data, heldout, verbose=verbose, run_label=run_label)

    test_result, _ = perceptron.test(test)
    codecs.open('../eval/%s_test.txt' % run_label, 'w', encoding='utf-8').write(test_result)

    if verbose:
        codecs.open('../eval/%s_features.csv' % run_label, 'w', encoding='utf-8').write('\n'.join(perceptron.evaluate_features()))

    return perceptron