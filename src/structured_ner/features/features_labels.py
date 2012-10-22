from feature_generator import FeatureSet

class LabelInteractionFeatures(FeatureSet):

    def generate_features(self, feature_storage, sequence, i, last_label, label):

        features_fired = []

        #Label bigram:
        feature_id = feature_storage.add_feature("label_bigram:%s-%s" % (last_label, label))
        features_fired.append(feature_id)

        #Last label and current POS tag:
        if sequence.x[i][1] in ['IN', ')', '(', 'PRP', 'POS']:
            feature_id = feature_storage.add_feature("labels_and_pos_tag:%s-%s-%s" % (last_label, sequence.x[i][0], label))
            features_fired.append(feature_id)

        return features_fired

