from feature_generator import FeatureSet

class LabelInteractionFeatures(FeatureSet):
    """Set of features for interaction between NE labels."""

    def generate_features(self, feature_storage, sequence, i, last_label, label):

        features_fired = []

        #Label bigram:
        feature_id = feature_storage.add_feature("label_bigram:%s-%s" % (last_label, label))
        features_fired.append(feature_id)

        #" as part of the entity:
        if sequence.x[i][0] == '"':
            feature_id = feature_storage.add_feature("''_and_last_POS:%s-%s" % (sequence.x[i][1], label))
            features_fired.append(feature_id)


        #Last label and current POS tag:
        def attachment(j):
            return sequence.x[j][0] in ['&'] or sequence.x[j][1] in ['IN', ')', '(', 'PRP', 'POS', 'Conj', 'Prep', '&', 'Num', 'Fe', 'Fz', 'SP']

        if attachment(i):
            feature_id = feature_storage.add_feature("attachment_problem:%s-%s-%s-%s" % (last_label, sequence.x[i][0], str(attachment(i-1)), label))
            features_fired.append(feature_id)

        return features_fired


