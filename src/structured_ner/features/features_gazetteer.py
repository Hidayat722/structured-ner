"""
>>> GazetteerFeatures([[u"The New York Times", u"New York"]]).find(u"The New York Times is a newspaper in New York .".split(' '))
['0', '0', '0', '0', 'O', 'O', 'O', 'O', '0', '0', 'O']

"""

from __future__ import division
import codecs
from collections import defaultdict
import sys
from feature_generator import FeatureSet
import dawg
import struct

class GazetteerFeatures(FeatureSet):

    def __init__(self, gazetteer_files, truecaser):

        data = []

        if type(gazetteer_files[0]) != list:
            for i in range(len(gazetteer_files)):
                gazetteer_files[i] = codecs.open(gazetteer_files[i], encoding='utf-8', mode='r')

        for i in range(len(gazetteer_files)):
            for entry in gazetteer_files[i]:
                e = entry.rstrip()
                data.append( (e, bytes(i)) )

                #Only store true-case versions of 2+ token entries if they are different from the original e
                if ' ' in entry:
                    e_true_case = u' '.join(truecaser.case(entry.rstrip().split(' '), []))

                    if e_true_case != e:
                        data.append( (e_true_case, bytes(i)) )

        self.gazetteer_dawg = dawg.BytesDAWG(data)

    def find(self, tokens):

        matches = []

        i = 0
        while i < len(tokens):

            j = i+1
            while j <= len(tokens) and len( self.gazetteer_dawg.keys( u' '.join(tokens[i:j]) ) ) > 0:
                j += 1

            match_found = False
            for z in reversed(range(i, j)):
                try:
                    matches.append( (i, z, self.gazetteer_dawg[ u' '.join(tokens[i:z]) ] ) )
                    match_found = True
                    break
                except KeyError:
                    pass

            if match_found:
                i = matches[-1][1] + 1
            else:
                i += 1


        annotations = map(lambda _: 'O', tokens)
        for (i, j, k) in matches:
            for p in range(i, j):
                annotations[ p ] = str(k[0])

        return annotations


    def filter(self, corpus):
        pass


    def generate_features(self, feature_storage, sequence, i, last_label, label):
        _, pos_tag = sequence.x[i]
        features_fired = []

        if sequence.gazetteer_entries[i] != 'O':
            feature_id = feature_storage.add_feature("gazetteer:%s-%s" % (sequence.gazetteer_entries[i], label))
            features_fired.append(feature_id)

        return features_fired
