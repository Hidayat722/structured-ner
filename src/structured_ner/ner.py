import pickle
from sentence import Sentence

class NamedEntityRecognizer:

    def recognize(self, text):
        pass


class StructuredNER(NamedEntityRecognizer):

    def __init__(self, perceptron_model, tokenizer, tagger):
        self.perceptron = pickle.load(perceptron_model)
        self.tokenizer = tokenizer
        self.tagger = tagger

    def recognize(self, tokens_and_tags):
        tokens = self.tokenizer.tokenize(tokens_and_tags)
        tags = self.tagger.tag(tokens)
        return zip(tokens, self.perceptron.viterbi_decode(Sentence(tags, [])))