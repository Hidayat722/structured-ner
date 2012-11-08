from __future__ import division
import numpy as np
import sys
from setuptools.command.sdist import entities
from util import entities_from_list

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

        errors = []
        true_and_predicted = []

        gold_count_for_label      = dict([ (l, 0) for l in self.labels ])
        predicted_count_for_label = dict([ (l, 0) for l in self.labels ])
        correct_count_for_label   = dict([ (l, 0) for l in self.labels ])

        for sentence in sentences:
            z = self.viterbi_decode(sentence)

            for i in range(len(z)):
                if z[i] != sentence.y[i]:
                    errors.append( sentence.x[i][0] + ' ' + sentence.x[i][1] + ' ' + sentence.true_case[i] + ' ' + sentence.y[i] + ' ' + z[i] )

                true_and_predicted.append([sentence.x[i][0], sentence.y[i], z[i]])

            total += len(sentence.y)
            incorrect += len( [i for i in range(len(sentence.y)) if sentence.y[i] != z[i]])

            gold_entities      = entities_from_list(sentence.y)
            predicted_entities = entities_from_list(z)

            for l in self.labels:
                if l != 'O':
                    gold_count_for_label[l]      += len( [ e for e in gold_entities      if e[2] == l] )
                    predicted_count_for_label[l] += len( [ e for e in predicted_entities if e[2] == l] )
                    correct_count_for_label[l]   += len( [ e for e in gold_entities      if e[2] == l and e in predicted_entities] )

        out = "Test Accuracy: %f\n" % (1.0 - (incorrect/total))

        out += "Label,  Precision,  Recall\n"
        out += "--------------------------\n"

        for l in self.labels:
            if l != 'O':

                try:
                    p = correct_count_for_label[l] / predicted_count_for_label[l]
                except ZeroDivisionError:
                    p = 0.

                try:
                    r = correct_count_for_label[l] / gold_count_for_label[l]
                except ZeroDivisionError:
                    r = 0.

                out += l + ' ' + str(p) + ' ' + str(r) + "\n"

        total_p = sum([ correct_count_for_label[l] for l in self.labels if l != 'O' ]) / sum([ predicted_count_for_label[l] for l in self.labels if l != 'O' ])
        total_r = sum([ correct_count_for_label[l] for l in self.labels if l != 'O' ]) / sum([ gold_count_for_label[l] for l in self.labels if l != 'O' ])

        f1 = 2 * ((total_p * total_r)/(total_p + total_r))
        out += "F1: " + str(f1)

        out += "\n"*5
        out += "Errors:\n"
        out += '\n'.join(errors)

        return out, true_and_predicted, 1.0 - (incorrect/total)


    def evaluate_features(self):
        return [ ("%s\t%.2f" % (self.feature_generator.get_feature_by_id(top_feature), self.w[top_feature])) for top_feature in reversed(np.argsort(self.w)[-100:])]

    def prune(self):
        self.feature_generator.prune(self.w)
