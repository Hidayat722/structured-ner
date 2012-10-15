def sentence_from_conll(chunked_sent):
    x = []
    y = []

    for element in chunked_sent:
        if type(element) == tuple:
            x.append(element)
            y.append('O')
        else:
            for ne_token in element.leaves():
                x.append(ne_token)
                y.append(element.node)

    return Sentence(x, y)

def load_conll(corpus):
    return [ sentence_from_conll(sent) for sent in corpus if len(sent) > 0 ]

class Sentence:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return " ".join(map(lambda (a,b): a[0]+"/"+a[1]+"/"+b, zip(self.x, self.y)))