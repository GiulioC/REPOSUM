es_query_template = {
  "query": {
    "bool": {
      "must": [{
        "simple_query_string": {
          "query": "",
          "default_operator": "or",
          "fields": ["contesti"]
        }
      }]
    }
  },
  "_source": {
    "excludes": ["contesti"]
  },
  "size": None
}

def parse_es_results(es_dict):
	query_hits = {}
	for hit in es_dict['hits']['hits']:
		hit_score = hit['_score']
		query_hits[hit['_id']] = hit_score
	return query_hits
