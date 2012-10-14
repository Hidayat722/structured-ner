import re
from repoze.lru import lru_cache
from feature_generator import FeatureSet

class SimpleNodeFeatures(FeatureSet):

    @lru_cache(maxsize=100)
    def generate_features(self, feature_storage, sequence, i, last_label, label):
        token, pos_tag = sequence.x[i]

        features_fired = []

        #Label ID:
        feature_id = feature_storage.add_feature("label:%s" % label)
        features_fired.append(feature_id)

        #Uppercase?:
        if token[0].isupper():
            feature_id = feature_storage.add_feature("capital_first-%s" % label)
            features_fired.append(feature_id)

        #All uppercase?:
        if token.isupper():
            feature_id = feature_storage.add_feature("all_upper-%s" % label)
            features_fired.append(feature_id)

        #Token ID:
        feature_id = feature_storage.add_feature("token:%s-%s" % (token, label))
        features_fired.append(feature_id)

        #POS Tag:
        feature_id = feature_storage.add_feature("tag:%s-%s" % (pos_tag, label))
        features_fired.append(feature_id)

        #Contains dash:
        if '-' in token:
            feature_id = feature_storage.add_feature("contains_dash:%s" % (label))
            features_fired.append(feature_id)

        #Contains Digit:
        if re.search("\d", token):
            feature_id = feature_storage.add_feature("contains_digit:%s" % label)
            features_fired.append(feature_id)

        #Suffix (2-4 length):
        for i in range(2, 4):
            if len(token) > i+1:
                feature_id = feature_storage.add_feature("suffix:%s-%s" % (token[-i:], label))
                features_fired.append(feature_id)

        #Prefix (2-4 length):
        for i in range(2, 4):
            if len(token) > i+1:
                feature_id = feature_storage.add_feature("prefix:%s-%s" % (token[:i], label))
                features_fired.append(feature_id)

        return features_fired
