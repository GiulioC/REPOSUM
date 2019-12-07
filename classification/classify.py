import text_processing as tp
import dataset_utils as dsu
import pandas as pd
import numpy as np
import joblib

use_preprocessed = False

vectorizer = joblib.load("models/vectorizer.pkl")
analyzer = vectorizer.build_analyzer()
clf = joblib.load("models/classifier.pkl")

data_abs = dsu.read_dataset_UK_ethos(True)
data_nabs = dsu.read_dataset_UK_ethos(False)
data = data_abs[['id', 'titolo']].append(data_nabs[['id', 'titolo']])
print("data read")

if use_preprocessed:
    data_titles = pd.read_csv("data/preprocessed_titles.csv")['titolo']
else:
    data_titles = tp.lemmatize_data(data['titolo'])
    data_titles = tp.preprocess_text_data(data_titles, analyzer)
    pd.DataFrame({'titolo':data_titles}).to_csv("data/preprocessed_titles.csv", index=None)

print("samples:",len(data_titles))
TDmatrix = vectorizer.transform(data_titles)
y_pred = clf.predict(TDmatrix)
y_pred_probs = clf.predict_proba(TDmatrix)
data.loc[:,'classification'] = y_pred
data.loc[:,'prob_0'] = [y[0] for y in y_pred_probs]
data.loc[:,'prob_1'] = [y[1] for y in y_pred_probs]
data.to_csv("results/classification.csv",index=None)
print("results saved")
