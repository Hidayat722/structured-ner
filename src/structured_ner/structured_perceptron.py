from __future__ import division
import numpy as np
import sys
from linear_classifier import LinearClassifier

class StructuredPerceptron(LinearClassifier):

    parameters_for_epoch = []

    def __init__(self, labels, feature_generator, epochs=10, eta=1.):
        LinearClassifier.__init__(self, labels, feature_generator)

        self.n_epochs = epochs
        self.n_features = feature_generator.n_features()
        self.eta = eta


    def train(self, sentences):

        self.w = np.zeros(self.n_features, dtype=float)

        for i_epoch in xrange(self.n_epochs):

            incorrect = 0.
            total     = 0.

            for sentence in sentences:
                total, incorrect = self.train_one(sentence, total, incorrect)

            self.parameters_for_epoch.append(self.w.copy())

            accuracy = 1.0 - (incorrect/total)

            print >>sys.stderr, "Epoch %i, Accuracy: %f" % (i_epoch, accuracy)

        #Average!
        averaged_parameters = 0
        for epoch_parameters in self.parameters_for_epoch:
            averaged_parameters += epoch_parameters
        averaged_parameters /= len(self.parameters_for_epoch)

        self.w = averaged_parameters

        #Finished training
        self.trained = True


    def train_one(self, sentence, total, incorrect):

        z = self.viterbi_decode(sentence)

        for i in xrange(len(sentence.x)):

            total += 1.

            y_i = sentence.y[i]
            z_i = z[i]

            if i == 0:
                y_prev = None
                z_prev = None
            else:
                y_prev = sentence.y[i-1]
                z_prev = z[i-1]

            if y_i != z_i:

                #The predicted tag was not the correct tag
                incorrect += 1.

                correct_token_features   = self.feature_generator.generate(sentence, i, y_prev, y_i)
                self.w[correct_token_features]   += self.eta

                predicted_token_features = self.feature_generator.generate(sentence, i, z_prev, z_i)
                self.w[predicted_token_features] -= self.eta

        return total, incorrect

    def prune(self):
        self.feature_generator.prune(self.w)
