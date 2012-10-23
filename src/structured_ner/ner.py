# -*- coding: utf-8 -*-

import pickle
from traits.trait_types import self
from sentence import Sentence
from sentence import all_uppercase, headline

class NamedEntityRecognizer:

    def recognize(self, text):
        pass


class StructuredNER(NamedEntityRecognizer):

    def __init__(self, perceptron_model, tokenizer, tagger, lemmatizer, truecaser):
        self.perceptron = pickle.load(perceptron_model)
        self.tokenizer = tokenizer
        self.tagger = tagger
        self.truecaser = truecaser
        self.lemmatizer = lemmatizer
        self.gazetteer = self.perceptron.feature_generator.feature_sets[-1]

        self.feature_by_id = {}

        for (feature, id) in self.perceptron.feature_generator.feature_ids.items():
            self.feature_by_id[id] = feature


    def recognize(self, sentence):
        tokens = self.tokenizer.tokenize( sentence )
        x = self.tagger.tag(tokens)
        _, tags = map(list, zip(*x))

        if self.truecaser is not None and all_uppercase(tokens, tags):
            true_case = self.truecaser.case(map(lambda x: x.lower(), tokens), map(lambda e: e[1], x))
        elif self.truecaser is not None and headline(tokens, tags):
            true_case = self.truecaser.case(tokens, map(lambda e: e[1], x))
        else:
            true_case = tokens

        if self.gazetteer:
            gazetteer_entries = self.gazetteer.find( true_case )
        else:
            gazetteer_entries = map(lambda _: 'O', true_case)

        lemmas = map(lambda tc: self.lemmatizer.lemmatize(tc), true_case)

        sentence = Sentence(x, [], lemmas, true_case, gazetteer_entries)

        labels = self.perceptron.viterbi_decode(sentence)

        features = []
        for i in range(len(tokens)):
            if i == 0:
                last_label = "<S>"
            else:
                last_label = labels[i-1]

            features.append( map(self.feature_by_id.get, self.perceptron.fs(sentence, i, last_label, labels[i])) )

        return zip(tokens, labels, features)