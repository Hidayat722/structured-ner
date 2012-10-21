from __future__ import division

def true_case_sentence(tokens, tags):
    return sum([ (tokens[i].isupper() or (tags[i] in ['CD', '.', ',', ':'])) for i in range(len(tokens))]) / float(len(tokens)) >= 0.95

def sentence_from_conll(chunked_sent, lemmatizer, truecaser):
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
    if truecaser is not None and true_case_sentence(tokens, tags):
        true_case = truecaser.case(tokens, map(lambda e: e[1], x))
    else:
        true_case = tokens

    lemmas = map(lambda tc: lemmatizer.lemmatize(tc), true_case)

    return Sentence(x, y, lemmas, true_case)

def load_conll(corpus, lemmatizer, truecaser):
    return [ sentence_from_conll(sent, lemmatizer, truecaser) for sent in corpus if len(sent) > 0 ]

class Sentence:

    def __init__(self, x, y, lemmas, true_case):
        self.x         = x
        self.y         = y
        self.lemmas    = lemmas
        self.true_case = true_case

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return " ".join(map(lambda (a,b): a[0]+"/"+a[1]+"/"+b, zip(self.x, self.y)))