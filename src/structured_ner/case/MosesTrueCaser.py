from copy import copy
import nltk
from TrueCaser import TrueCaser

class MosesTrueCaser(TrueCaser):

    """
    This is a Python port of the truecasing Perl script in the Moses SMT system:

    http://code.google.com/p/smt/source/browse/trunk/moses64/tools/moses-scripts/recaser/truecase.perl?r=32

    """

    def __init__(self, model_file):

        self.best  = {}
        self.known = {}

        for line in model_file:
            entry = line.split(" ")

            self.best[ entry[0].lower() ] = entry[0]
            self.known[ entry[0] ] = 1

            for i in range(2, len(entry), 2):
                self.known[ entry[i] ] = 1

        model_file.close()

    def case(self, tokens, tags):

        l_tokens = map(lambda s: s.lower(), tokens)
        cased_tokens = copy(l_tokens)

        for i in range(len(tokens)):
            word = tokens[i]

            if i == 0 and word.lower() in self.best:
                #Truecase sentence start
                cased_tokens[i] = self.best[ word.lower() ]
            elif word in self.known:
                #don't change known words
                cased_tokens[i] = word
            elif word.lower() in self.best:
                # truecase otherwise unknown words
                cased_tokens[i] = self.best[ word.lower() ]
            else:
                # unknown, nothing to do
                cased_tokens[i] = word

        return cased_tokens

t = MosesTrueCaser(open('/Users/jodaiber/Desktop/Groningen/structured-ner/src/structured_ner/models/truecase/truecase-model.en'))
print t.case_pairs( nltk.pos_tag( "New York".split(' ')) )