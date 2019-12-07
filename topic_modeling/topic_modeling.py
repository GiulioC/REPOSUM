from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from treetagger import TreeTagger
import pyLDAvis.sklearn
import joblib, sys, re
import pandas as pd
import utils

################################################################################

file_path = "../data/tesi_US/US_PhD_dissertations.xlsx"
n_features = 20000
min_df_cv = 0.0
max_df_cv = 1.0
use_preprocessed_data = False
n_topics = 20
n_top_words = 30

################################################################################

if len(sys.argv) == 2:
    n_topics = int(sys.argv[1])
elif len(sys.argv) > 2:
    n_topics = [int(sys.argv[i]) for i in range(1,len(sys.argv))]

cv = CountVectorizer(
    stop_words=set(list(stopwords)+utils.manual_stopwords),
    analyzer="word",
    min_df=min_df_cv,
    max_df=max_df_cv,
    max_features=n_features
)
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

print("training vectorizer")
TDmat = cv.fit_transform(df['preprocessed'])
joblib.dump(cv, "models/cv_{}.pkl".format(n_features))

if isinstance(n_topics, list):
    topic_numbers = n_topics
else:
    topic_numbers = [n_topics]

for num in topic_numbers:
    lda = LatentDirichletAllocation(
        n_components=num,
        max_iter=12,
        learning_method='online',
        learning_offset=30.,
        random_state=0,
        n_jobs=6
    )
    print("training lda with {} topics".format(num))
    lda.fit(cv.transform(df['preprocessed']))
    utils.print_top_words(lda, cv.get_feature_names(), n_top_words)

    joblib.dump(lda, "models/lda_{}_{}.pkl".format(num, n_features))
    utils.visualize_lda(lda, TDmat, cv, True, "html/lda_{}_{}.html".format(num, n_features))
