import sparql_utils as spqlu
import pandas as pd

P_names = {}
Q_names = {}

wikidata_props = {}
errors = set()
df = pd.read_csv("../data/philosophers.csv")
for index, row in df.iterrows():

    wikiQ = row['item']
    Qname = row['itemLabel']
    wikiQ = wikiQ.split("/")[-1]

    print("[{} - {}] ({})".format(wikiQ, Qname, index))

    phil_triples = spqlu.make_request(wikiQ, '_'.join(Qname.split()), "philosopher")
    if phil_triples != "EXP":
        with open ("../data/wikidata_triples.csv", "a") as f:
            for triple in phil_triples:
                try:
                    a,b,c = triple
                    f.write("{}\t{}\t{}\n".format(a,b,c))
                except ValueError:
                    print("ValueError at triple {}".format(triple))
                    errors.add(wikiQ)

    print("saved {} triples".format(len(phil_triples)))

with open("../data/errors.csv", "w") as f:
    for err in errors:
        f.write(err+"\n")
