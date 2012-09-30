
class FeatureGenerator:

    feature_ids = {}

    def n_features(self):
        pass

    def generate(self, sequence, i, last_label, label):
        pass

    def add_feature(self, feature):
        """
        Remember the id of the feature.
        """

        if feature in self.feature_ids:
            return self.feature_ids[feature]
        else:
            id = len(self.feature_ids)
            self.feature_ids[feature] = id
            return id


class SimpleFeatureGenerator(FeatureGenerator):

    def n_features(self):
        return 0

    def generate(self, sequence, i, last_label, label):

        features_fired = []

        x_token, x_tag = sequence[i]

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