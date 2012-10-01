from __future__ import division
import numpy as np
import sys

class StructuredPerceptron:

    w = None
    parameters_for_epoch = []

    def __init__(self, labels, feature_generator, epochs=10, eta=1.):
        self.n_epochs = epochs
        self.labels = labels
        self.feature_generator = feature_generator
        self.n_features = feature_generator.n_features()
        self.eta = eta

    def test(self, sentences):
        total = 0
        incorrect = 0

        tp_for_label = dict([ (l, 0) for l in self.labels ])
        fp_for_label = dict([ (l, 0) for l in self.labels ])
        fn_for_label = dict([ (l, 0) for l in self.labels ])

        for sentence in sentences:
            z = self.viterbi_decode(sentence)

            for i in xrange(len(sentence.x)):
                total += 1.

                if sentence.y[i] != z[i]:
                    incorrect += 1

                    if sentence.y[i] == 'O':
                        fp_for_label[ z[i] ] += 1

                    if z[i] == 'O':
                        fn_for_label[ sentence.y[i] ] += 1

                else:
                    tp_for_label[ z[i] ] += 1

        print >>sys.stderr, "Test Accuracy: %f" % (1.0 - (incorrect/total))

        print >>sys.stderr,     "Label,  Precision,  Recall"
        print >>sys.stderr,     "--------------------------"
        for l in self.labels:
            if l != 'O':
                print >>sys.stderr, l, str(tp_for_label[l]/(tp_for_label[l] + fp_for_label[l])), str(tp_for_label[l]/(tp_for_label[l] + fn_for_label[l]))



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

    def fs(self, sentence, i, last_label, current_label):
        return self.feature_generator.generate(sentence, i, last_label, current_label)

    def viterbi_decode(self, sentence):
        alphas = np.zeros((sentence.size(), len(self.labels)), dtype=float)
        path = {}

        for y in xrange(len(self.labels)):
            path[y] = [y]
            alphas[0][y] = np.sum(self.w[self.fs(sentence, 0, None, self.labels[y])])

        for i in xrange(1, sentence.size()):
            new_path = {}

            for y in xrange(len(self.labels)):
                (alphas[i][y], b) = max([(alphas[i-1][y_last] + np.sum(self.w[self.fs(sentence, i, self.labels[y_last], self.labels[y])] ), y_last) for y_last in xrange(len(self.labels))])
                new_path[y] = path[b] + [y]

            path = new_path

        return map(lambda i: self.labels[i], path[ alphas[-1:,].argmax() ])
