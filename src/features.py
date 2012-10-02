import re
import sys
from repoze.lru import lru_cache


class FeatureGenerator:

    feature_ids = {}

    trained = False

    def __init__(self, sentences):

        for sentence in sentences:
            for i in xrange(sentence.size()):
                if i == 0:
                    last_label = None
                else:
                    last_label = sentence.y[i-1]
                self.generate(sentence, i, last_label, sentence.y[i])
        self.trained = True
        print >>sys.stderr, "# Features: %d" % (self.n_features())

    def n_features(self):
        return len(self.feature_ids)

    def generate(self, sequence, i, last_label, label):
        pass

    def add_feature(self, feature):
        """
        Remember the id of the feature.
        """

        if feature in self.feature_ids:
            return self.feature_ids[feature]
        elif self.trained:
            return -1
        else:
            id = len(self.feature_ids)
            self.feature_ids[feature] = id
            return id


class SimpleNodeFeatureGenerator(FeatureGenerator):

    @lru_cache(maxsize=100)
    def generate_node_features(self, token, pos_tag, label):
        features_fired = []

        #Label prior:
        feature_id = self.add_feature("label:%s" % label)
        features_fired.append(feature_id)

        #Uppercase?:
        if token[0].isupper():
            feature_id = self.add_feature("capital_first-%s" % label)
            features_fired.append(feature_id)

        #All uppercase?:
        if token.isupper():
            feature_id = self.add_feature("all_upper-%s" % label)
            features_fired.append(feature_id)

        #Token:
        feature_id = self.add_feature("token:%s-%s" % (token, label))
        features_fired.append(feature_id)

        #Tag:
        feature_id = self.add_feature("tag:%s-%s" % (pos_tag, label))
        features_fired.append(feature_id)

        #Token:
        if '-' in token:
            feature_id = self.add_feature("contains_dash:%s" % (label))
            features_fired.append(feature_id)

        #Digits:
        if re.search("\d", token):
            feature_id = self.add_feature("contains_digit:%s" % label)
            features_fired.append(feature_id)

        return features_fired

    def generate(self, sequence, i, last_label, label):
        x_token, x_tag = sequence.x[i]
        return self.generate_node_features(x_token, x_tag, label)


class ExtendedFeatureGenerator(SimpleNodeFeatureGenerator):

    def generate(self, sequence, i, last_label, label):

        x_token, x_tag = sequence.x[i]
        features_fired = self.generate_node_features(x_token, x_tag, label)

        #Label bigram:
        feature_id = self.add_feature("label_bigram:%s-%s" % (last_label, label))
        features_fired.append(feature_id)

        #Token+label bigram:
        #feature_id = self.add_feature("token:%s-%s-%s" % (x_token, label, last_label))
        #features_fired.append(feature_id)


        return features_fired