from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import pandas as pd
import numpy as np
import random, os
import utils

base_dir = "../data/tesi_US/carriere/excel"
correct_threshold = 1

le_rank = LabelEncoder()
le_succ = LabelEncoder()
le_phil = LabelEncoder()
le_uni = LabelEncoder()

ranks_src = []
ranks_dst = []
succs = []
phils = []
unis_src = []
unis_dst = []

#data_df = None
for file in os.listdir(base_dir):
    print("reading {}...".format(file))
    phil_name = file.split()[0]
    df_columns = utils.get_columns_info(phil_name)
    df = pd.read_excel(
        os.path.join(base_dir,file),
        usecols = df_columns.values(),
        names = df_columns.keys()
    )
    for col in df_columns.keys():
        df = df[~df[col].isnull()]

    ranks_src.extend(list(df["rank-src"]))
    ranks_dst.extend(list(df["rank-dst"]))
    succs.extend(list(df["success"].astype(int)))
    phils.extend([phil_name for _ in range(df.count()[0])])
    unis_src.extend(df["uni-src"].apply(lambda x: x.strip().lower()))
    unis_dst.extend(df["uni-dst"].apply(lambda x: x.strip().lower()))

le_rank.fit(list(set(ranks_src+ranks_dst)))
le_succ.fit(list(set(succs)))
le_phil.fit(list(set(phils)))
le_uni.fit(list(set(unis_src+unis_dst)))

print("transforming data")
train_data = list(zip(
    le_uni.transform(unis_src),
    le_rank.transform(ranks_src),
    le_uni.transform(unis_dst),
    le_rank.transform(ranks_dst),
    le_phil.transform(phils)
))
train_labels = le_succ.transform(succs)

seed = random.randint(0,1000)
X_train, X_test, y_train, y_test = train_test_split(
    train_data,
    train_labels,
    test_size = 0.2,
    random_state = seed
)

print("training classifier")
clf = utils.load_estimator()
y_pred = clf.fit(X_train, y_train).predict(X_test)

y_pred_t = []
for i in range(len(y_pred)):
    if abs(int(y_pred[i])-int(y_test[i])) <= correct_threshold:
        y_pred_t.append(y_test[i])
    else:
        y_pred_t.append(y_pred[i])

acc = str(accuracy_score(y_test, y_pred_t))[0:6]
precision = precision_score(y_test, y_pred_t, average="weighted")
recall = recall_score(y_test, y_pred_t, average="weighted")
f1 = f1_score(y_test, y_pred_t, average="weighted")
print("\n\nShowing results for seed {}\n".format(seed))
print("ACCURACY\tPRECISION\tRECALL\t\tF1")
print("%s\t\t%s\t\t%s\t\t%s"%(acc,precision,recall,f1))
