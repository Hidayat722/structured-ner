from __future__ import division
import codecs
import numpy as np
import sys
from linear_classifier import LinearClassifier
import matplotlib.pyplot as plt


class StructuredPerceptron(LinearClassifier):

    def __init__(self, labels, feature_generator, epochs=10, eta=1.):
        LinearClassifier.__init__(self, labels, feature_generator)

        self.parameters_for_epoch = []

        self.n_epochs = epochs
        self.n_features = feature_generator.n_features()
        self.eta = eta


    def train(self, data, heldout, verbose=False, run_label=None):

        self.w = np.zeros(self.n_features, dtype=float)

        training_accuracy = [0.0]
        heldout_accuracy = [0.0]

        for i_epoch in xrange(self.n_epochs):

            incorrect = 0.
            total     = 0.

            for sentence in data:
                total, incorrect = self.train_one(sentence, total, incorrect)

            self.parameters_for_epoch.append(self.w.copy())

            accuracy = 1.0 - (incorrect/total)
            training_accuracy.append(accuracy)

            #Stop if the error on the training data does not decrease
            if training_accuracy[-1] <= training_accuracy[-2]:
                break

            if verbose:
                _, acc = self.test(heldout)
                heldout_accuracy.append(acc)

            print >>sys.stderr, "Epoch %i, Accuracy: %f" % (i_epoch, accuracy)

        #Average!
        averaged_parameters = 0
        for epoch_parameters in self.parameters_for_epoch:
            averaged_parameters += epoch_parameters
        averaged_parameters /= len(self.parameters_for_epoch)

        self.w = averaged_parameters

        #Finished training
        self.trained = True

        #Export training info in verbose mode:
        if verbose:
            x = np.arange(0, self.n_epochs+1, 1.0)
            plt.plot(x, training_accuracy, marker='o', linestyle='--', color='r', label='Training')
            plt.plot(x, heldout_accuracy,  marker='o', linestyle='--', color='b', label='Heldout')

            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Heldout Accuracy')

            plt.ylim([0.9, 1.0])

            plt.legend(bbox_to_anchor=(1., 0.2))

            codecs.open('../eval/%s_training.csv' % run_label, 'w', encoding='utf-8').write('\n'.join(map(lambda x: ', '.join(map(str, x)), zip(x, training_accuracy, heldout_accuracy))))
            plt.savefig('../eval/%s_training.png' % run_label)

            plt.close()



    def train_one(self, sentence, total, incorrect):

        z = self.viterbi_decode(sentence)

        for i in xrange(len(sentence.x)):

            total += 1.

            y_i = sentence.y[i]
            z_i = z[i]

            if i == 0:
                y_prev = '<S>'
                z_prev = '<S>'
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
