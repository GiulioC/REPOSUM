import networkx as nx
import pandas as pd
import pickle

# US or UK
theses_dataset = "US"

reverse_props = {
    "tmf_title": "in_title",
    "tmf_abstract": "in_abs"
}

G = nx.read_gpickle('../data/wikidata_graph.pkl')
data = pd.read_csv(
    "../data/tmf_entities.csv".format(theses_dataset)
    names=["id", "src", "entity", "score"]
)

new_nodes = 0
old_nodes = 0
last_id = None
for index, row in data.iterrows():
    print(index)

    try:
        id = row['id']
        if id != last_id:
            G.add_node(id, node_type='thesis')
            last_id = id
        src = row['src']
        entity = row['entity'].lower()
        score = row['score']
    except AttributeError:
        continue

    if entity in G.nodes:
        old_nodes += 1
    else:
        G.add_node(entity, node_type='tmf_ent')
        new_nodes += 1

    G.add_edge(id, entity, rel_type='tmf_'+src)
    G.add_edge(entity, id, rel_type=reverse_props['tmf_'+src])

nx.write_gpickle(G,'../data/complete_graph.pkl'.format(theses_dataset))

print("Nodes already present: {}".format(old_nodes))
print("Nodes added: {}".format(new_nodes))
