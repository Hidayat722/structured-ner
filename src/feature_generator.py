
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

    def n_features(self):
        return len(self.feature_ids)

    def generate(self, sequence, i, last_label, label):
        pass

    def add_feature(self, feature):
        """
        Remember the id of the feature.
        """

        if self.trained:
            return -1

        if feature in self.feature_ids:
            return self.feature_ids[feature]
        else:
            id = len(self.feature_ids)
            self.feature_ids[feature] = id
            return id


class SimpleFeatureGenerator(FeatureGenerator):


    def generate(self, sequence, i, last_label, label):

        features_fired = []

        x_token, x_tag = sequence.x[i]

        #Token:
        feature_id = self.add_feature("token: %s-%s" % (x_token, label))
        features_fired.append(feature_id)

        #Tag:
        feature_id = self.add_feature("tag: %s-%s" % (x_tag, label))
        features_fired.append(feature_id)

        #Digits:
        if unicode.isdigit(x_token):
            feature_id = self.add_feature("digit: %s" % label)
            features_fired.append(feature_id)

        return features_fired