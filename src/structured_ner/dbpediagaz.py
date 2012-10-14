from SPARQLWrapper import SPARQLWrapper2, JSON,XML



class gazEntry():

    def __init__(self, lang):

        persons= self.getGazEntries(lang, 'Person')
        orgs= self.getGazEntries(lang, 'Organisation')
        locs= self.getGazEntries(lang, 'Place')
        comb_list=[];
        comb_list.append(persons)
        comb_list.append(orgs)
        comb_list.append(locs)
        print(comb_list)







    def getGazEntries(self, lang, entry):

       sparql = SPARQLWrapper2("http://dbpedia.org/sparql")
       sparql.setQuery("""
       PREFIX dbp: <http://dbpedia.org/ontology/>
       SELECT ?label
       WHERE {
            ?entity rdf:type dbp:Person.
            ?entity rdfs:label ?label.
           FILTER (lang(?label) = 'es')
        }
        """)
       sparql.setReturnFormat(XML)
       results = sparql.query().convert()

       entries=[];

       for binding in results.bindings :
           entries.append(binding["label"].value)


       return entries


comb_list=gazEntry('es')

