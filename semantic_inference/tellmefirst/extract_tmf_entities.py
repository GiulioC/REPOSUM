from elasticsearch import Elasticsearch
import dataset_utils as dsu
import tmf_utils as tmfu
import re, json, pickle
import elasticsearch
import pandas as pd

# US or UK
theses_dataset = "US"
index_name = 'wiki-en'
num_results_title = 10
num_results_abstract = 20
score_threshold = 30.0
out_file = "../data/tmf_entities_{}.csv".format(theses_dataset)
max_score = 0
min_score = 100

####################################################################

es = Elasticsearch('localhost', port=9200)

data_files = []
if theses_dataset == "US":
    data_files.append(dsu.read_dataset_US())
else:
    data_files.append(dsu.read_dataset_UK_ethos(True))
    data_files.append(dsu.read_dataset_UK_ethos(False))
print("data read")

tmf_entities = {}
for input_file in data_files:
    for index,row in input_file.iterrows():
    	print(index, end="\r")

    	title = row['title'].strip().replace("\"", "'")
    	doc_id = row['id']

    	if 'abstract' in data.columns:

    	if '***NO TITLE PROVIDED***' not in title:
    		tmf_entities[doc_id] = {
    			"title":{},
    			"abstract":{}
    		}

    		es_query = tmfu.es_query_template['query']['bool']['must'][0]['simple_query_string']['query'] = title
    		es_query['size'] = num_results_title
    		res_title = es.search(index=index_name, body=es_query)
    		tmf_entities[doc_id]['title'] = tmfu.parse_es_results(res_title)
    	else:
    		continue

    	if 'abstract' in data.columns:
        	abstract = row['abstract'].strip().lower().replace("\"", "'")
    		if abstract != 'nessun elemento disponibile.' and abstract != 'abstract not available.':
    			es_query['query']['bool']['must'][0]['simple_query_string']['query'] = abstract
    			es_query['size'] = num_results_abstract
    			res_abstract = es.search(index=index_name, body=es_query)
    			tmf_entities[doc_id]['abstract'] = tmfu.parse_es_results(res_abstract)

with open(out_file, "w") as f:
	for doc_id, inner_dict in tmf_entities.items():
		for entity, score in inner_dict['title'].items():
			f.write("{}\t{}\t{}\t{}\n".format(doc_id, "title", entity, score))

		for entity, score in inner_dict['abstract'].items():
			f.write("{}\t{}\t{}\t{}\n".format(doc_id, "abstract", entity, score))
