import codecs
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



    def generate_features(self, feature_storage, sequence, i, last_label, label):
        token, pos_tag = sequence.x[i]
        features_fired = []

        if token in self.gazetteer_tokens:
            for g_file in self.gazetteer_tokens[token]:
                feature_id = feature_storage.add_feature("gazetteer:%s-%s" % (g_file, label))
                features_fired.append(feature_id)

        return features_fired
