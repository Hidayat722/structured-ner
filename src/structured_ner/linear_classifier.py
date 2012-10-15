from __future__ import division
import numpy as np
import sys

class LinearClassifier(object):

    def __init__(self, labels, feature_generator):
        self.w = None
        self.feature_generator = feature_generator
        self.labels = labels

    def fs(self, sentence, i, last_label, current_label):
        return filter(lambda f: f != -1, self.feature_generator.generate(sentence, i, last_label, current_label))

    def viterbi_decode(self, sentence, k=1):
        alphas = np.zeros((len(sentence), len(self.labels)), dtype=float)
        path = {}

        for y in xrange(len(self.labels)):
            path[y] = [y]
            alphas[0][y] = np.sum(self.w[self.fs(sentence, 0, '<S>', self.labels[y])])

        for i in xrange(1, len(sentence)):
            new_path = {}

            for y in xrange(len(self.labels)):
                (alphas[i][y], b) = max([(alphas[i-1][y_last] + np.sum(self.w[self.fs(sentence, i, self.labels[y_last], self.labels[y])] ), y_last) for y_last in xrange(len(self.labels))])
                new_path[y] = path[b] + [y]

            path = new_path

        return map(lambda i: self.labels[i], path[ alphas[-1:,].argmax() ])


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

                elif z[i] != 'O':
                    tp_for_label[ z[i] ] += 1

        out = "Test Accuracy: %f\n" % (1.0 - (incorrect/total))

        out += "Label,  Precision,  Recall\n"
        out += "--------------------------\n"

        for l in self.labels:
            if l != 'O':

                try:
                    p = tp_for_label[l]/(tp_for_label[l] + fp_for_label[l])
                except ZeroDivisionError:
                    p = 0.

                try:
                    r = tp_for_label[l]/(tp_for_label[l] + fn_for_label[l])
                except ZeroDivisionError:
                    r = 0.

                out += l + ' ' + str(p) + ' ' + str(r) + "\n"


        return out, 1.0 - (incorrect/total)


    def evaluate_features(self):
        return [ ("%s\t%.2f" % (self.feature_generator.get_feature_by_id(top_feature), self.w[top_feature])) for top_feature in reversed(np.argsort(self.w)[-100:])]

    def prune(self):
        self.feature_generator.prune(self.w)
