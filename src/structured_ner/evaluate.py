import pickle
from nltk.corpus import conll2002
from nltk.corpus.reader.conll import ConllChunkCorpusReader
from nltk.corpus.util import LazyCorpusLoader
from sentence import load_conll
from features.features_labels import LabelInteractionFeatures
from features.features_node import SimpleNodeFeatures
from features.features_gazetteer import GazetteerFeatures
from train_supervised import train_ner

conll2002_labels = ['LOC', 'PER', 'ORG', 'MISC', 'O']
conll2003 = LazyCorpusLoader('conll2003', ConllChunkCorpusReader, '.*\.(test|train).*', ('LOC', 'PER', 'ORG', 'MISC'), encoding='utf-8')

########################################
# Feature sets and Gazetteers for Dutch
########################################

nl_gazetteer = GazetteerFeatures(['data/loc_nl.txt', 'data/person_nl.txt', 'data/org_nl.txt'])

dutch_train     = load_conll(conll2002.chunked_sents('ned.train'))
dutch_heldout   = load_conll(conll2002.chunked_sents('ned.testa'))
dutch_test      = load_conll(conll2002.chunked_sents('ned.testb'))

dutch_ner        = train_ner('nl', conll2002_labels, dutch_train, dutch_heldout, dutch_test, [SimpleNodeFeatures(), LabelInteractionFeatures(), nl_gazetteer], verbose=True)
dutch_ner2       = train_ner('nl', conll2002_labels, dutch_train, dutch_heldout, dutch_test, [SimpleNodeFeatures(), LabelInteractionFeatures()], verbose=True)
dutch_ner1       = train_ner('nl', conll2002_labels, dutch_train, dutch_heldout, dutch_test, [SimpleNodeFeatures()], verbose=True)

nlen_gazetteer   = GazetteerFeatures(['data/loc_nl.txt', 'data/person_nl.txt', 'data/org_nl.txt', 'data/loc_en.txt', 'data/person_en.txt', 'data/org_en.txt'])
dutch_ner_gaz_en = train_ner('nl', conll2002_labels, dutch_train, dutch_heldout, dutch_test, [SimpleNodeFeatures(), LabelInteractionFeatures(), nlen_gazetteer], verbose=True, additional_run_label='EnglishGazetteerEntries')


########################################
# Other languages
########################################

for (lang, corpus) in [('deu', conll2003), ('eng', conll2003), ('es', conll2002)]:

    train   = load_conll(corpus.chunked_sents('%s.train' % lang))
    heldout = load_conll(corpus.chunked_sents('%s.testa' % lang))
    test    = load_conll(corpus.chunked_sents('%s.testb' % lang))

    gazetteer = GazetteerFeatures(['data/loc_%s.txt' % lang, 'data/person_%s.txt' % lang, 'data/org_%s.txt' % lang])
    model = train_ner(lang, conll2002_labels, train, heldout, test, [SimpleNodeFeatures(), LabelInteractionFeatures(), gazetteer])

    pickle.dump(model, open("models/%s.pickle" % lang))
