from sklearn.metrics import recall_score, precision_score,f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import vstack
import text_processing as tp
import dataset_utils as dsu
import pandas as pd
import numpy as np
import joblib

#train options
cross_validation = False
cv_splits = 10
#train data options
min_df = 0.0
max_df = 0.8
n_features = 60000

cv = tp.build_vectorizer("cv", min_df, max_df, n_features)
analyzer = cv.build_analyzer()
vectorizer = tp.build_vectorizer("tfidf")

# read and preprocess text data
positive_samples_train, negative_samples_train = dsu.read_train_samples()
if not cross_validation:
    test_samples = dsu.read_test_samples()

print("Positive samples_train:",positive_samples_train.count()[0])
print("Negative samples_train:",negative_samples_train.count()[0])
if not cross_validation:
    print("Test samples:",test_samples.count()[0])

phil_titles = tp.lemmatize_data(positive_samples_train['title'])
phil_titles = tp.preprocess_text_data(phil_titles, analyzer)
nphil_titles = tp.lemmatize_data(negative_samples_train['title'])
nphil_titles = tp.preprocess_text_data(nphil_titles, analyzer)

if not cross_validation:
    test_titles = tp.lemmatize_data(test_samples['title'])
    test_titles = tp.preprocess_text_data(test_titles, analyzer)

# transform text data into vector space
vectorizer.fit(phil_titles.append(nphil_titles))
joblib.dump(vectorizer, "models/vectorizer.pkl")
TDmatrix_pos = vectorizer.transform(phil_titles)
TDmatrix_neg = vectorizer.transform(nphil_titles)
print("Vocabulary:",len(vectorizer.vocabulary_))
print("TDmatrix 1:",TDmatrix_pos.shape)
print("TDmatrix 2:",TDmatrix_neg.shape)
if not cross_validation:
    TDmatrix_test = vectorizer.transform(test_titles)
    labels_test = test_samples['philosophy']
    print("TDmatrix test:",TDmatrix_test.shape)

TDmatrix_train = vstack((TDmatrix_pos, TDmatrix_neg)).tocsr()
labels_train = np.concatenate(
    (np.ones(TDmatrix_pos.shape[0],dtype=int),np.zeros(TDmatrix_neg.shape[0],dtype=int)),
    axis=0
)

#shuffle data and labels
s = np.arange(TDmatrix_train.shape[0])
np.random.shuffle(s)
TDmatrix_train = TDmatrix_train[s]
labels_train = labels_train[s]

clf = RandomForestClassifier(
    max_depth=None,
    random_state=None,
    n_estimators=50,
    max_features=0.6,
    n_jobs=-1, #WARNING: consider changing it
    verbose=2
)

if cross_validation:
    #10-fold cross-validation
    k_fold = KFold(n_splits=cv_splits)
    it = 1
    accuracies = []
    precisions = []
    recalls = []
    f1scores = []
    for train, test in k_fold.split(train_data):
        y_true = labels_train[test]
        print("cv iteration {}...".format(it))
        clf.fit(TDmatrix_train[train], labels_train[train])
        y_pred = clf.predict(TDmatrix_train[test])

        accuracies.append(np.mean(np.equal(y_true, y_pred)))
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
        f1scores.append(f1_score(y_true, y_pred))

        print("accuracy at iteration {}: {}".format(it, np.mean(np.equal(y_true, y_pred))))
        print("precision at iteration {}: {}".format(it, precision_score(y_true, y_pred)))
        print("recall at iteration {}: {}".format(it, recall_score(y_true, y_pred)))
        print("f1 at iteration {}: {}".format(it, f1_score(y_true, y_pred)))
        it += 1
    print("\n\ncross-validation accuracy:",np.mean(np.array(accuracies)))
    print("cross-validation precision:",np.mean(np.array(precisions)))
    print("cross-validation recall:",np.mean(np.array(recalls)))
    print("cross-validation f1:",np.mean(np.array(f1scores)))

clf.fit(TDmatrix_train, labels_train)
y_pred = clf.predict(TDmatrix_test)
acc = np.mean(np.equal(labels_test, y_pred))
precision = precision_score(labels_test, y_pred)
recall = recall_score(labels_test, y_pred)
f1score = f1_score(labels_test, y_pred)
print("\n\ntest set Accuracy:",acc)
print("test set Precision:",precision)
print("test set Recall:",recall)
print("test set F1:",f1score)

joblib.dump(clf, 'models/classifier.pkl')
