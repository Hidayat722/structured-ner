import re
from repoze.lru import lru_cache
from feature_generator import FeatureSet

class SimpleNodeFeatures(FeatureSet):

    initial_pattern = re.compile(r"[A-Z]\.")
    digit_pattern_2 = re.compile(r"[0-9]{2}")
    digit_pattern_4 = re.compile(r"[0-9]{4}")


    def __init__(self):
        self.lemmatizer = None

    @lru_cache(maxsize=100)
    def generate_features(self, feature_storage, sequence, i, last_label, label):
        _, pos_tag = sequence.x[i]
        token = sequence.true_case[i]

        features_fired = []

        #Label ID:
        feature_id = feature_storage.add_feature("label:%s" % label)
        features_fired.append(feature_id)

        #First word.
        if i == 0:
            feature_id = feature_storage.add_feature("first_word-%s" % label)
            features_fired.append(feature_id)

        #In quotes?
        if i < len(sequence)-1 and sequence.x[i-1] == '"' and sequence.x[i+1] == '"':
            feature_id = feature_storage.add_feature("in_quotes-%s" % label)
            features_fired.append(feature_id)

        #Numbers?
        if self.digit_pattern_2.match(token):
            feature_id = feature_storage.add_feature("2_digit_number-%s" % label)
            features_fired.append(feature_id)

        if self.digit_pattern_4.match(token):
            feature_id = feature_storage.add_feature("2_digit_number-%s" % label)
            features_fired.append(feature_id)


        #Initial of name?
        if self.initial_pattern.match(token):
            feature_id = feature_storage.add_feature("name_initial-%s" % label)
            features_fired.append(feature_id)

        #Uppercase at start of sentence?:
        if token[0].isupper() and i == 0:
            feature_id = feature_storage.add_feature("capital_first_sentence_initial-%s" % label)
            features_fired.append(feature_id)

        #Uppercase within sentence?:
        if token[0].isupper() and i > 0:
            feature_id = feature_storage.add_feature("capital_first_non_sentence_initial-%s" % label)
            features_fired.append(feature_id)

        #All uppercase?:
        if token.isupper():
            feature_id = feature_storage.add_feature("all_upper-%s" % label)
            features_fired.append(feature_id)

        #All lowercase?:
        if token.islower():
            feature_id = feature_storage.add_feature("all_lower-%s" % label)
            features_fired.append(feature_id)

        #Token ID:
        feature_id = feature_storage.add_feature("token:%s-%s" % (token, label))
        features_fired.append(feature_id)

        #Lowercased token ID:
        feature_id = feature_storage.add_feature("lower_token:%s-%s" % (token.lower(), label))
        features_fired.append(feature_id)

        #Lemma:
        if sequence.lemmas:
            lemma = sequence.lemmas[i]
            feature_id = feature_storage.add_feature("lemma:%s-%s" % (lemma, label))
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
        for i in range(2, 4+1):
            if len(token) > i+2:
                feature_id = feature_storage.add_feature("suffix:%s-%s" % (token[-i:], label))
                features_fired.append(feature_id)

        #Prefix (2-4 length):
        for i in range(2, 4+1):
            if len(token) > i+2:
                feature_id = feature_storage.add_feature("prefix:%s-%s" % (token[:i], label))
                features_fired.append(feature_id)

        return features_fired
