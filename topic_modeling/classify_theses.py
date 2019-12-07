from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from treetagger import TreeTagger
import heapq, sys, joblib
import pandas as pd
import utils

################################################################################

n_features = 20000
n_topics = int(sys.argv[1]) if len(sys.argv) > 1 else 20
n_top_topics = 10
use_preprocessed_data = False

################################################################################

cv = joblib.load("models/cv_{}.pkl".format(n_features))
analyzer = cv.build_analyzer()
tt = TreeTagger(language='english', path_to_treetagger="tree-tagger-linux-3.2.2/")

if not use_preprocessed_data:
    print("reading data")
    df = pd.read_excel(
        file_path,
        usecols=[13,24],
        names=['id','abstract']
    )

    print("preprocessing data")
    df = utils.preprocess_data(df, analyzer, tt)
    df.to_csv("data/tesi_US_preprocessed.csv", index=None)
else:
    print("loading preprocessed data")
    df = pd.read_csv("data/tesi_US_preprocessed.csv")

ids = list(df['id'])
print("using model: lda_{}_{}".format(n_topics, n_features))
lda = joblib.load("models/lda_{}_{}.pkl".format(n_topics, n_features))
probs = lda.transform(cv.transform(df['preprocessed']))

print("saving topics...")
with open("out/probs_{}_{}.csv".format(n_topics, n_features), "w") as outfile:
    outfile.write("id,topic,prob\n")
    for id, t_probs in zip(ids, probs):
        highest_probs = heapq.nlargest(n_top_topics, t_probs)
        for prob in sorted(highest_probs, reverse=True):
            ind = list(t_probs).index(prob)
            outfile.write(str(id)+","+str(ind)+","+str(t_probs[ind])[0:6]+"\n")
