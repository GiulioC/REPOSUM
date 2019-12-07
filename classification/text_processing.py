from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import pandas as pd
import nltk, re

def build_vectorizer(type, min_df=0.0, max_df=1.0, n_features=20000):
    if type == "cv":
        return CountVectorizer(stop_words=list(nltk.corpus.stopwords.words('english')),
                                analyzer="word",
                                min_df=min_df,
                                max_df=max_df,
                                max_features=n_features)
    elif type == "tfidf":
        return TfidfVectorizer(stop_words=list(nltk.corpus.stopwords.words('english')),
                                analyzer="word")

def lemmatize_data(data_series, output=True):
    wnl = WordNetLemmatizer()
    res = []

    for num, title in enumerate(data_series):
        if output:
            print(num, end="\r")
        words = [wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in pos_tag(word_tokenize(str(title)))]
        res.append(' '.join(words))
    if output:
        print("")
    return pd.Series(res)

def preprocess_text_data(text_data, processer, check_length=True):
    text_data = text_data.apply(lambda x: " ".join(processer(x)))
    text_data = text_data.apply(lambda x: re.sub(r'[0-9]+',"",x))
    if check_length:
        text_data = text_data.apply(lambda x: " ".join(s for s in x.split() if len(s) > 3))
    return text_data
