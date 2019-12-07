import pandas as pd
import numpy as np
import pickle
import utils

ents_file = "../semantic_inference/tellmefirst/data/tmf_entities_UK.csv"

print("reading data")
tmf_entities = pd.read_csv(
	ents_file,
	delimiter="\t",
	names=['id', 'src', 'entity', 'score']
)

print("getting entities")
ents_df = utils.prepare_data(tmf_entities)
input(ents_df)

print("loading models")
cv = pickle.load(open("models/mapper_30000.pkl", "rb"))
svd = pickle.load(open("models/svd_30000.pkl", "rb"))
clf = pickle.load(open("models/clf_30000.pkl", "rb"))

print("transforming text")
ents_matrix = cv.transform(ents_df)

print("reducing dimensionality")
svd_matrix = svd.transform(ents_matrix)

print("classifying")
y_pred = clf.predict(svd_matrix)

num_phils = np.sum(y_pred)
print("\nDocuments classified as philosophical: {}".format(num_phils))
print("Documents Classified as non-philosophical: {}".format(len(y_pred)-num_phils))

ents_df.loc[:,'classification'] = y_pred
ents_df.to_csv("data/classification_results.csv", index=None)
