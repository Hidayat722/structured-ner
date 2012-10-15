from nltk.corpus import conll2002
from sentence import sentence_from_conll
from features.features_labels import LabelInteractionFeatures
from features.features_node import SimpleNodeFeatures
from src.structured_ner.features.features_gazetteer import GazetteerFeatures
from train_supervised import train_ner

conll2002_labels = ['LOC', 'PER', 'ORG', 'MISC', 'O']

nl_gazetteer = GazetteerFeatures(['data/loc_nl.txt', 'data/person_nl.txt', 'data/org_nl.txt'])

dutch_train     = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('ned.train')]
dutch_heldout   = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('ned.testa')]
dutch_test      = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('ned.testb')]

dutch_ner        = train_ner('nl', conll2002_labels, dutch_train, dutch_heldout, [SimpleNodeFeatures(), LabelInteractionFeatures(), nl_gazetteer], verbose=True)
dutch_ner2       = train_ner('nl', conll2002_labels, dutch_train, dutch_heldout, [SimpleNodeFeatures(), LabelInteractionFeatures()], verbose=True)
dutch_ner1       = train_ner('nl', conll2002_labels, dutch_train, dutch_heldout, [SimpleNodeFeatures()], verbose=True)

spanish_train   = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('esp.train')]
spanish_heldout = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('esp.testa')]
spanish_test    = [sentence_from_conll(sent) for sent in conll2002.chunked_sents('esp.testb')]
spanish_ner     = train_ner('es', conll2002_labels, spanish_train, spanish_heldout, feature_sets)

dutch_ner.test(dutch_test)
spanish_ner.test(spanish_test)