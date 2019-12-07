from sklearn.metrics import recall_score, precision_score,f1_score
from sklearn_pandas import DataFrameMapper
from random import shuffle
import pandas as pd
import numpy as np
import pickle
import utils

features_title = 10000
features_abstract = 20000
need_to_select_data = False
compute_importances = False
save_models = True
reduce_dimensionality = True
classifier_type = 'svc'

ents_file = "../semantic_inference/tellmefirst/data/tmf_entities_UK.csv"
freq_files = [
    "../semantic_inference/tellmefirst/data/entities_freq_title.tsv",
    "../semantic_inference/tellmefirst/data/entities_freq_abstract.tsv"
]
phil_files = [
    "../classification/classification_clean/data/philosophy_train.csv",
    "../classification/classification_clean/data/nophilosophy_train.csv"
]

print("reading data")
ents_title = pd.read_csv(
    freq_files[0],
    delimiter="\t",
    names=['entity','freq'],
    nrows=features_title
)['entity']

ents_abstract = pd.read_csv(
    freq_files[1],
    delimiter="\t",
    names=['entity','freq'],
    nrows=features_abstract
)['entity']

positive_samples_train, negative_samples_train = utils.read_phil_samples(phil_files)
test_samples = pd.read_csv(
    "../classification/classification_clean/data/test_set_1000.tsv",
    delimiter="\t",
    names=[
        'title','creator','university','publisher', 'year','abstract','type',
        'subject','id','philosophy'
    ]
)

if need_to_select_data:
    print("preparing train/test samples")
    tmf_entities = pd.read_csv(
        ents_file,
        delimiter="\t",
        names=['id', 'src', 'entity', 'score']
    )
    id_pos = positive_samples_train['id']
    id_neg = negative_samples_train['id']
    id_test = test_samples['id']
    data_df = utils.prepare_data(
        tmf_entities[tmf_entities['id'].isin(id_pos.append(id_neg).append(id_test))]
    )

    #train data
    data_train_pos = data_df[data_df['id'].isin(id_pos)]
    data_train_pos = data_train_pos.fillna("")

    data_train_neg = data_df[data_df['id'].isin(id_neg)]
    data_train_neg = data_train_neg.fillna("")

    data_train = data_train_pos.append(data_train_neg)
    labels_train = np.concatenate(
        (np.ones(len(id_pos),dtype=int),np.zeros(len(id_neg),dtype=int)),
        axis=0
    )
    data_train.loc[:,"philosophy"] = labels_train

    #test data
    data_test = data_df[data_df['id'].isin(id_test)]
    data_test.fillna("")
    labels_test = list(test_samples['philosophy'])
    data_test.loc[:,"philosophy"] = labels_test

    data_train.to_csv("data/data_train.csv", index=None)
    data_test.to_csv("data/data_test.csv", index=None)
else:
    #train data
    data_train = pd.read_csv("data/data_train.csv")
    data_train = data_train.fillna("")
    labels_train = list(data_train['philosophy'])

    #test data
    data_test = pd.read_csv("data/data_test.csv")
    data_test = data_test.fillna("")
    labels_test = list(data_test['philosophy'])

cv_title = utils.get_custom_vectorizer(ents_title)
cv_abstract = utils.get_custom_vectorizer(ents_abstract)

tuple_array = [
    ('title', cv_title),
    ('abstract', cv_abstract),
]
mapper = DataFrameMapper(tuple_array, sparse=True)
mapper.fit(data_train.append(data_test))

print("transforming data")
matrix_train = mapper.transform(data_train)
matrix_test = mapper.transform(data_test)
if save_models:
    pickle.dump(mapper, open("models/mapper_{}.pkl".format(features_title+features_abstract), "wb"))

indices = np.arange(matrix_train.shape[0])
shuffle(indices)
matrix_train = matrix_train[list(indices)]
labels_train = np.array(labels_train)[indices]

if reduce_dimensionality:
    print("computing SVD features")
    svd = utils.get_dim_reduction_model(2000)
    matrix_train = svd.fit_transform(matrix_train)
    matrix_test = svd.transform(matrix_test)
    if save_models:
    	pickle.dump(svd, open("models/svd_{}.pkl".format(features_title+features_abstract), "wb"))

print("training classifier")
clf = utils.get_classifier('random_forest')
clf.fit(matrix_train, labels_train)
if save_models:
    pickle.dump(clf, open("models/clf_{}.pkl".format(features_title+features_abstract), "wb"))

y_true = labels_test
y_pred = clf.predict(matrix_test)

acc = str(np.mean(np.equal(y_true, y_pred)))[0:6]
precision = str(precision_score(y_true, y_pred))[0:6]
recall = str(recall_score(y_true, y_pred))[0:6]
f1score = str(f1_score(y_true, y_pred))[0:6]
print("ACCURACY\tPRECISION\tRECALL\t\tF1")
print("%s\t\t%s\t\t%s\t\t%s"%(acc,precision,recall,f1score))

if classifier_type == 'random_forest' and compute_importances:
    utils.compute_features_importance(clf, cv_title, cv_abstract)
