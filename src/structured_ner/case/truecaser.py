class TrueCaser:

    def case_pairs(self, token_tags):
        tokens, tags = zip(*token_tags)
        return self.case(list(tokens), list(tags))

    def case(self, tokens, pos_tags):
        pass
