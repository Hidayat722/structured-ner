from feature_generator import FeatureSet

class LabelInteractionFeatures(FeatureSet):

    def generate_features(self, feature_storage, sequence, i, last_label, label):

        features_fired = []

        #Label bigram:
        feature_id = feature_storage.add_feature("label_bigram:%s-%s" % (last_label, label))
        features_fired.append(feature_id)

        return features_fired

