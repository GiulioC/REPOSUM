import requests

WIKIDATA_URL = 'https://query.wikidata.org/sparql'
explored = set()

def make_request(page_id, prev_page_name, type):
    if page_id not in explored or type == "philosopher":
        r = requests.get(
            WIKIDATA_URL,
            params = {
                'format': 'json',
                'query': wikidata_query%(page_id)
            }
        )
        explored.add(page_id)
        try:
            return parse_result(page_id, prev_page_name, r.json()['results']['bindings'], type)
        except:
            return None
    else:
        return None

def parse_result(page, prev_page_name, query_res, type):
    P = [t[0] for t in useful_props[type]]
    P_recurs = [t[2] for t in useful_props[type]]
    triples = []
    for row in query_res:
        prop_id = row['wd']['value'].split("/")[-1]
        if prop_id in P:
            recurs = P_recurs[P.index(prop_id)]
            if recurs is not None:
                recur_res = make_request(row['o']['value'].split("/")[-1], '_'.join(row['ooLabel']['value'].split()), recurs)
                if recur_res is None or recur_res == []:
                    triples.append((prev_page_name, '_'.join(row['wdLabel']['value'].lower().split()), '_'.join(row['ooLabel']['value'].split())))
                else:
                    triples.append((prev_page_name, '_'.join(row['wdLabel']['value'].lower().split()), '_'.join(row['ooLabel']['value'].split())))
                    triples.extend(recur_res)
            else:
                triples.append((prev_page_name, '_'.join(row['wdLabel']['value'].lower().split()), '_'.join(row['ooLabel']['value'].split())))
    return triples

wikidata_query = """
    SELECT ?wdLabel ?wd ?o ?ooLabel
    WHERE
    {
        VALUES (?s) {(wd:%s)}
        ?s ?wdt ?o .
        ?wd wikibase:directClaim ?wdt .
        ?wd rdfs:label ?wdLabel .
        OPTIONAL {
            ?o rdfs:label ?oLabel .
            FILTER (lang(?oLabel) = "en")
        }
        FILTER (lang(?wdLabel) = "en")
        BIND (COALESCE(?oLabel, ?o) AS ?ooLabel)
    }
    ORDER BY xsd:integer(STRAFTER(STR(?wd), "http://www.wikidata.org/entity/P"))
"""

useful_props = {
    "philosopher": [
        ("P39", False, "subclassof"),
        ("P69", False, None),
        ("P101", True, "fow"),
        ("P106", True, "occupation"),
        ("P108", False, None),
        ("P135", True, "subclassof"),
        ("P184", False, None),
        ("P185", False, None),
        ("P463", False, None),
        ("P737", False, None),
        ("P742", False, None),
        ("P793", False, None),
        ("P800", True, "works"),
        ("P802", False, None),
        ("P859", False, None),
        ("P937", False, None),
        ("P1066", False, None),
        ("P1441", False, None),
        ("P2650", False, "subclassof"),
    ],
    "occupation": [
        ("P279", True, "subclassof"),
        ("P425", False, None)
    ],
    "subclassof": [
        ("P279", False, "subclassof"),
        ("P361", True, "subclassof"),
        ("P461", True, "subclassof"),
        ("P1269", True, "subclassof")
    ],
    "fow": [
        ("P279", True, "subclassof"),
        ("P361", True, "subclassof"),
        ("P461", False, None),
        ("P2578", False, None),
        ("P2579", False, None),
        ("P138", False, None),
    ],
    "movement": [
        ("P61", False, None),
        ("P112", False, None),
        ("P138", False, None),
        ("P279", True, "subclassof"),
        ("P737", False, None),
        ("P1269", True, "facetof"),
        ("P2579", False, None),
        ("P2813", False, None),
        ("P3095", True, "subclassof"),
    ],
    "works": [
        ("P50", False, None),
        ("P156", True, "works"),
        ("P674", False, None),
        ("P921", False, None)
    ]
}
