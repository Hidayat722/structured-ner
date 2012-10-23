import pickle
from nltk.corpus import conll2002
from nltk.corpus.reader.conll import ConllChunkCorpusReader
from nltk.corpus.util import LazyCorpusLoader
from sentence import load_conll
from nltk.stem.wordnet import WordNetLemmatizer
from features.features_labels import LabelInteractionFeatures
from features.features_node import SimpleNodeFeatures
from features.features_gazetteer import GazetteerFeatures
from case.MosesTrueCaser import MosesTrueCaser
from train_supervised import train_ner

conll2002_labels = ['LOC', 'PER', 'ORG', 'MISC', 'O']
conll2003 = LazyCorpusLoader('conll2003', ConllChunkCorpusReader, '.*\.(test|train).*', ('LOC', 'PER', 'ORG', 'MISC'), encoding='utf-8')

########################################
# Feature sets and Gazetteers for Dutch
########################################

#nl_gazetteer = GazetteerFeatures(['data/loc_nl.txt', 'data/person_nl.txt', 'data/org_nl.txt'])

#dutch_train     = load_conll(conll2002.chunked_sents('ned.train'))[:6000]
#dutch_heldout   = load_conll(conll2002.chunked_sents('ned.testa'))[:1000]
#dutch_test      = load_conll(conll2002.chunked_sents('ned.testb'))

#dutch_ner        = train_ner('nl', conll2002_labels, dutch_train, dutch_heldout, dutch_test, [SimpleNodeFeatures(), LabelInteractionFeatures(), nl_gazetteer], verbose=True)
#pickle.dump(dutch_ner, open("models/%s.pickle" % 'nl', 'w'))

#dutch_ner2       = train_ner('nl', conll2002_labels, dutch_train, dutch_heldout, dutch_test, [SimpleNodeFeatures(), LabelInteractionFeatures()], verbose=True)
#dutch_ner1       = train_ner('nl', conll2002_labels, dutch_train, dutch_heldout, dutch_test, [SimpleNodeFeatures()], verbose=True)

#nlen_gazetteer   = GazetteerFeatures(['data/loc_nl.txt', 'data/person_nl.txt', 'data/org_nl.txt', 'data/loc_eng.txt', 'data/person_eng.txt', 'data/org_eng.txt'])
#dutch_ner_gaz_en = train_ner('nl', conll2002_labels, dutch_train, dutch_heldout, dutch_test, [SimpleNodeFeatures(), LabelInteractionFeatures(), nlen_gazetteer], verbose=True, additional_run_label='EnglishGazetteerEntries')


########################################
# Other languages
########################################

for (lang, corpus) in [
    ('eng', conll2003),
    #('deu', conll2003),
    #('esp', conll2002)
    ]:

    lemmatizer = WordNetLemmatizer()
    truecaser  = MosesTrueCaser(open('/Users/jodaiber/Desktop/Groningen/structured-ner/src/structured_ner/models/truecase/truecase-model.en'))

    gazetteer = GazetteerFeatures(['data/loc_%s.txt' % lang, 'data/person_%s.txt' % lang, 'data/org_%s.txt' % lang, 'data/misc_%s.txt' % lang], truecaser)

    train   = load_conll(corpus.chunked_sents('%s.train' % lang), lemmatizer, None,      gazetteer)
    heldout = load_conll(corpus.chunked_sents('%s.testa' % lang), lemmatizer, truecaser, gazetteer)
    test    = load_conll(corpus.chunked_sents('%s.testb' % lang), lemmatizer, truecaser, gazetteer)


    #gazetteer.filter(train)

    node_features = SimpleNodeFeatures()

    model_gaz = train_ner(lang, conll2002_labels, train, heldout, test, [node_features, LabelInteractionFeatures(), gazetteer], verbose=True)
    pickle.dump(model_gaz, open("models/%s_gaz.pickle" % lang, 'w'))

    model     = train_ner(lang, conll2002_labels, train, heldout, test, [node_features, LabelInteractionFeatures()])
    pickle.dump(model, open("models/%s.pickle" % lang, 'w'))

