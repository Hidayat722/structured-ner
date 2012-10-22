import codecs
from SPARQLWrapper import SPARQLWrapper2, JSON, XML
import sys

class GazetteerExtractor():

    def __init__(self, lang):

        persons          = self.getGazetteerEntries(lang, 'Person')
        print >>sys.stderr, "Retrieved %d person names." % len(persons)
        codecs.open("data/person_%s.txt" % lang, encoding='utf-8', mode='w').write('\n'.join(persons))

        organizations    = self.getGazetteerEntries(lang, 'Organisation')
        print >>sys.stderr, "Retrieved %d organization names." % len(organizations)
        codecs.open("data/org_%s.txt" % lang, encoding='utf-8', mode='w').write('\n'.join(organizations))

        locations        = self.getGazetteerEntries(lang, 'Place')
        print >>sys.stderr, "Retrieved %d location names." % len(locations)
        codecs.open("data/loc_%s.txt" % lang, encoding='utf-8', mode='w').write('\n'.join(locations))



    def getGazetteerEntries(self, lang, dbp_class):

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


GazetteerExtractor('en')
GazetteerExtractor('nl')
GazetteerExtractor('es')
GazetteerExtractor('de')

