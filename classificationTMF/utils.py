from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
import pandas as pd
import numpy as np

def read_phil_samples(sample_files):
    phils = pd.read_csv(sample_files[0])
    nphils = pd.read_csv(sample_files[1])
    return phils, nphils

def prepare_data(ents_file):
    last_id = ents_file.iloc[0]['id']
    text = {"id": [], "title": [], "abstract": []}
    t_temp = ""
    a_temp = ""
    for index,row in ents_file.iterrows():
        print(index, end="\r")
        if row['id'] != last_id:
            text['id'].append(last_id)
            text['title'].append(t_temp)
            text['abstract'].append(a_temp)
            last_id = int(row['id'])
            t_temp = ""
            a_temp = ""
        if row['src'] == 'title':
            t_temp = t_temp + " " + str(row['entity'])
        else:
            a_temp = a_temp + " " + str(row['entity'])
    print("")
    text['id'].append(int(last_id))
    text['title'].append(t_temp)
    text['abstract'].append(a_temp)
    return pd.DataFrame(text)

def custom_tokenizer(txt):
    return txt.lower().split()

def get_custom_vectorizer(custom_vocab):
    return CountVectorizer(
        vocabulary = custom_vocab,
        stop_words = None,
        tokenizer = custom_tokenizer,
        preprocessor = None,
    )

def get_dim_reduction_model(num_dimensions):
    return TruncatedSVD(
        n_components=num_dimensions,
        n_iter=20,
        random_state=420
    )

def get_classifier(clf_type):
    if clf_type == 'svc':
        return svm.SVC(
            gamma='auto'
        )
    elif clf_type == 'random_forest':
        return RandomForestClassifier(
            max_depth=None,
            random_state=None,
            n_estimators=100,
            max_features=0.25,
            n_jobs=6,
            verbose=0
        )

def compute_feature_importances(clf, cv_title, cv_abstract):
    num_features = effective_features
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
    indices = np.argsort(importances)[::-1][0:num_features]
    feature_names = cv_abstract.get_feature_names() + cv_title.get_feature_names()

    with open("data/importances.tsv", "w") as f:
        f.write("name\timportance\tstd\n")
        for ind in indices:
            f.write("{}\t{}\t{}\n".format(feature_names[ind],importances[ind],std[ind]))
