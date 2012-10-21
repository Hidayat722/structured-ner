from __future__ import division
import codecs
from collections import defaultdict
import sys
from feature_generator import FeatureSet

class GazetteerFeatures(FeatureSet):

    def __init__(self, gazetteer_files):

        self.gazetteer_tokens = {}

        for g_file in gazetteer_files:
            for entry in codecs.open(g_file, encoding='utf-8', mode='r'):
                for token in entry.strip().split(' '):

                    if token.startswith('('):
                        continue

                    if token not in self.gazetteer_tokens:
                        self.gazetteer_tokens[token] = []

                    if g_file not in self.gazetteer_tokens[token]:
                        self.gazetteer_tokens[token].append(g_file)


    def filter(self, corpus):

        print >>sys.stderr, "Filtering %d Gaz. tokens" % len(self.gazetteer_tokens)

        token_NE_count = defaultdict(int)
        token_O_count  = defaultdict(int)

        for sent in corpus:
            for (token_tag, label) in zip(sent.x, sent.y):
                token, tag = token_tag
                if label == 'O':
                    token_O_count[token]  += 1
                else:
                    token_NE_count[token] += 1

        N = len(set(token_O_count.keys() + token_NE_count.keys()))

        for token in self.gazetteer_tokens.keys():

            pNE = (token_NE_count[token] + 1) / (token_NE_count[token] + token_O_count[token] + N)
            pO  = (token_O_count[token] + 1)  / (token_NE_count[token] + token_O_count[token] + N)

            #if P(NE|token) < P(O|token): remove it
            if pNE < pO:
                del self.gazetteer_tokens[token]

        print >>sys.stderr, "Tokens after filtering: %d" % len(self.gazetteer_tokens)


    def generate_features(self, feature_storage, sequence, i, last_label, label):
        _, pos_tag = sequence.x[i]
        token = sequence.true_case[i]
        features_fired = []

        if token in self.gazetteer_tokens:
            for g_file in self.gazetteer_tokens[token]:
                feature_id = feature_storage.add_feature("gazetteer:%s-%s" % (g_file, label))
                features_fired.append(feature_id)

        return features_fired
