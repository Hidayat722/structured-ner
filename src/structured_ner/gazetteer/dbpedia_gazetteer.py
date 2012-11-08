import codecs
from SPARQLWrapper import SPARQLWrapper2, JSON, XML
import sys

class GazetteerExtractor():

    """
    A GazetteerExtractor automatically retrieves lists of names from a public LOD endpoint.
    Currently the lists are for the classes Person, Organization, Location and SportsEvent.
    The list can easily be extended, since the mapping to labels is done automatically by
    the Perceptron learning algorithm.
    """

    dbp_classes = ['Person', 'Organisation', 'Place', 'SportsEvent']

    def __init__(self, lang):

        for dbp_class in GazetteerExtractor.dbp_classes:
            list = self.retrieve_name_list(lang, dbp_class)
            print >>sys.stderr, "Retrieved %d names: %s" % (len(list), dbp_class)

            codecs.open("../data/%s_%s.txt" % (dbp_class, lang), encoding='utf-8', mode='w').write('\n'.join(list))


    def retrieve_name_list(self, lang, dbp_class):

        repeat  = True
        entries = []
        offset  = 0
        limit   = 50000

        while repeat:
            sparql = SPARQLWrapper2("http://dbpedia.org/sparql")
            sparql.setQuery("""
            PREFIX dbp: <http://dbpedia.org/ontology/>
            SELECT ?label
            WHERE {
                ?entity rdf:type dbp:%s.
                ?entity rdfs:label ?label.
                FILTER (lang(?label) = '%s')
            }
            LIMIT %d
            OFFSET %d
            """ % (dbp_class, lang, limit, offset) )

            sparql.setReturnFormat(XML)
            results = sparql.query().convert()

            for binding in results.bindings:
                entries.append(binding["label"].value)

            if len(results.bindings) < limit:
                repeat = False
            else:
                offset += limit

            print '.'

        return entries

if __name__ == '__main__':
    GazetteerExtractor('en')
    GazetteerExtractor('nl')
    GazetteerExtractor('es')
    GazetteerExtractor('de')

