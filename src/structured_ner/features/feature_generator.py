import sys

class FeatureGenerator(object):

    def __init__(self, sentences, feature_sets):
        self.trained = False
        self.feature_ids = {}
        self.feature_sets = feature_sets

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
        features_fired = []

        for feature_set in self.feature_sets:
            features_fired.extend( feature_set.generate_features(self, sequence, i, last_label, label) )

        return features_fired


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

    def prune(self, w):
        remove_features = []
        for (feature, feature_id) in self.feature_ids.items():
            if w[feature_id] == 0.0:
                remove_features.append(feature)

        for f in remove_features:
            del self.feature_ids[f]

    def get_feature_by_id(self, feature_id):
        for (feature, f_id) in self.feature_ids.items():
            if f_id == feature_id:
                return feature
        return None


class FeatureSet:

    def generate_features(self, feature_storage, sequence, i, last_label, label):
        pass


