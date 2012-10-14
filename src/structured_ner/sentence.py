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

class Sentence:

    def __init__(self, x, y):
        self.token_tags = x
        self.labels = y

    def size(self):
        return len(self.token_tags)

    def __str__(self):
        return " ".join(map(lambda (a,b): a[0]+"/"+a[1]+"/"+b, zip(self.token_tags, self.labels)))