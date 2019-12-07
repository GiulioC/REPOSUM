import pandas as pd
import pyLDAvis.sklearn

manual_stopwords = [
    'theory', 'work', 'chapter', 'majority', 'study', 'dissertation', 'claim',
    'philosophy', 'argue', 'argument', 'approach', 'research', 'student',
    'participant', 'data', 'read', 'literature', 'write', 'text', 'literary',
    'object', 'change', 'project', 'view', 'problem', 'concept', 'discourse',
    'test', 'report', 'result', 'hypothesis'
]

def preprocess_data(data_df, analyzer, tt):
    data_df = data_df.applymap(lambda x: str(x).strip())
    data_df['abstract'] = data_df['abstract'].apply(lambda x: x.lower())
    data_df = data_df[data_df['abstract'] != 'abstract not available.']
    data_df = data_df[data_df['abstract'] != 'nessun elemento disponibile.']
    data_df['abstract'] = data_df['abstract'].apply(lambda x: " ".join(word for word in analyzer(x)))
    data_df['abstract'] = data_df['abstract'].apply(lambda x: " ".join(word for word in x.split() if len(word) > 2))
    data_df = data_df[data_df['abstract'] != '']
    data_df.loc[:,'preprocessed'] = lemmatize_data(data_df['abstract'], tt)
    data_df = data_df.drop(['abstract'], axis=1)
    return data_df

def lemmatize_data(data, treetagger):
    lemm_data = []
    for count, data_abstract in enumerate(data):
        print(count, end="\r")
        abstract = ''
        if len(data_abstract) != 0:
            for word, _, lemma in treetagger.tag(data_abstract):
                if lemma == '<unknown>':
                    abstract += word + ' '
                elif "|" in lemma:
                    parts = lemma.split("|")
                    abstract += min(parts, key=len) + ' '
                else:
                    abstract += lemma + ' '
        lemm_data.append(abstract)
    return lemm_data

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print()
        message = "Topic #%d: " % int(topic_idx+1)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print("\n\n")

def visualize_lda(lda_model, TDmatrix, vectorizer, sort_t, path):
    panel = pyLDAvis.sklearn.prepare(lda_model, TDmatrix, vectorizer, mds='tsne', sort_topics=sort_t)
    pyLDAvis.save_html(panel, path)
