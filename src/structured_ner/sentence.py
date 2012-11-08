from __future__ import division
from nltk.stem.api import StemmerI

def all_uppercase(tokens, tags):
    return sum([ (tokens[i].isupper() or (tags[i] in ['CD', '.', ',', ':'])) for i in range(len(tokens))]) / float(len(tokens)) >= 0.95

def headline(tokens, tags):
    return sum([ not tokens[i].islower() for i in range(len(tokens))]) / float(len(tokens)) == 1.0 and tokens[-1] not in ('.', '?', '!') and len([ 1 for i in range(len(tokens)) if tags[i] not in ['CD', '.', ',', ':'] ]) >= 3

def sentence_from_conll(chunked_sent, lemmatizer, truecaser, gazetteer):
    x = []
    y = []

    for element in chunked_sent:
        if type(element) == tuple:
            x.append( (element[0], element[1]) )
            y.append('O')
        else:
            for ne_token in element.leaves():
                x.append( (ne_token[0], ne_token[1]) )
                y.append(element.node)

    tokens, tags = zip(*x)
    tokens, tags = list(tokens), list(tags)

    if truecaser is not None and all_uppercase(tokens, tags):
        true_case = truecaser.case(map(lambda x: x.lower(), tokens), map(lambda e: e[1], x))
    elif truecaser is not None and headline(tokens, tags):
        true_case = truecaser.case(tokens, map(lambda e: e[1], x))
    else:
        true_case = tokens

    if gazetteer:
        gazetteer_entries = gazetteer.find( true_case )
    else:
        gazetteer_entries = map(lambda _: 'O', true_case)

    if isinstance(lemmatizer, StemmerI):
        lemmas = map(lambda tc: lemmatizer.stem(tc), true_case)
    else:
        lemmas = map(lambda tc: lemmatizer.lemmatize(tc), true_case)

    return Sentence(x, y, lemmas, true_case, gazetteer_entries)

def load_conll(corpus, lemmatizer, truecaser, gazetteer):
    return [ sentence_from_conll(sent, lemmatizer, truecaser, gazetteer) for sent in corpus if len(sent) > 0 ]

class Sentence:

    def __init__(self, x, y, lemmas, true_case, gazetteer_entries):
        self.x         = x
        self.y         = y
        self.lemmas    = lemmas
        self.true_case = true_case
        self.gazetteer_entries = gazetteer_entries

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return " ".join(map(lambda (a,b): a[0]+"/"+a[1]+"/"+b, zip(self.x, self.y)))