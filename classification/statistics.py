from sklearn.feature_extraction.text import CountVectorizer
from heapq import nsmallest, nlargest
import matplotlib.pyplot as plt
import dataset_utils as dsu
import pandas as pd
import numpy as np
import re

# select the stat(s) you want to compute
stat_year = False
stat_corpus = False
stat_train = False
stat_test = False
stat_feature_score = False

vectorizer = CountVectorizer(stop_words="english", analyzer="word")
analyzer = vectorizer.build_analyzer()

if stat_year:
    #anno pubblicazione
    title = "Years Distribution"
    xx = "number of documents"
    yy = "year"
    plt.clf()

    file_abs = dsu.read_dataset_UK_ethos(True)
    file_nabs = dsu.read_dataset_UK_ethos(False)

    for file, color in zip([file_abs, file_nabs], ['r','b']):

        years = set()
        plot_data = {i:0 for i in range(1850,2030)}
        for index,row in file.iterrows():
            year = row['anno']
            if year == ' dcterms_issued:: @@MISSING-DATA' or year == 'dcterms_issued:: @@MISSING-DATA':
                continue
            years.add(int(year))
            try:
                plot_data[int(year)] = plot_data[int(year)] + 1
            except KeyError:
                continue

        print(nsmallest(10, list(years)))
        print(nlargest(10, list(years)))

        x = [k for k,v in plot_data.items()]
        y = [v for k,v in plot_data.items()]
        plt.plot(x,y, color)

    plt.ylabel(xx)
    plt.xlabel(yy)
    plt.title(title)
    plt.legend(('Abtracts', 'No abstracts'), loc=2)
    plt.show()

if stat_corpus:
    #words statistics
    tot_words = 0
    tot_words_prep = 0
    unique_words = set()
    unique_words_prep = set()

    tot_words_abs = 0
    tot_words_prep_abs = 0
    unique_words_abs = set()
    unique_words_prep_abs = set()

    tot_words_nabs = 0
    tot_words_prep_nabs = 0
    unique_words_nabs = set()
    unique_words_prep_nabs = set()

    file_abs = dsu.read_dataset_UK_ethos(True)
    file_nabs = dsu.read_dataset_UK_ethos(False)

    for file, process_abstract in zip([file_abs, file_nabs], [True,False]):
        for index,row in file.iterrows():
            print(index, end="\r")
            title = row['titolo']
            subject = row['argomento']
            if process_abstract:
                abstract = row['abs']

            if re.search(r'MISSING-DATA',title) is None:
                ws = title.split()
                tot_words += len(ws)
                if process_abstract:
                    tot_words_abs += len(ws)
                else:
                    tot_words_nabs += len(ws)
                for w in ws:
                    unique_words.add(w)
                    if process_abstract:
                        unique_words_abs.add(w)
                    else:
                        unique_words_nabs.add(w)
                tokens = analyzer(title)
                tot_words_prep += len(tokens)
                if process_abstract:
                    tot_words_prep_abs += len(tokens)
                else:
                    tot_words_prep_nabs += len(tokens)
                for w in tokens:
                    unique_words_prep.add(w)
                    if process_abstract:
                        unique_words_prep_abs.add(w)
                    else:
                        unique_words_prep_nabs.add(w)
            if re.search(r'MISSING-DATA',subject) is None:
                ws = subject.split()
                tot_words += len(ws)
                if process_abstract:
                    tot_words_abs += len(ws)
                else:
                    tot_words_nabs += len(ws)
                for w in ws:
                    unique_words.add(w)
                    if process_abstract:
                        unique_words_abs.add(w)
                    else:
                        unique_words_nabs.add(w)
                tokens = analyzer(subject)
                tot_words_prep += len(tokens)
                if process_abstract:
                    tot_words_prep_abs += len(tokens)
                else:
                    tot_words_prep_nabs += len(tokens)
                for w in tokens:
                    unique_words_prep.add(w)
                    if process_abstract:
                        unique_words_prep_abs.add(w)
                    else:
                        unique_words_prep_nabs.add(w)
            if process_abstract:
                if re.search(r'MISSING-DATA',abstract) is None:
                    ws = abstract.split()
                    tot_words += len(ws)
                    tot_words_abs += len(ws)
                    for w in ws:
                        unique_words.add(w)
                        unique_words_abs.add(w)
                    tokens = analyzer(abstract)
                    tot_words_prep += len(tokens)
                    tot_words_prep_abs += len(tokens)
                    for w in tokens:
                        unique_words_prep.add(w)
                        unique_words_prep_abs.add(w)
        print("\n")

    print("Total words:",tot_words)
    print("Total words prep:",tot_words_prep)
    print("Total unique words:",len(unique_words))
    print("Total unique words prep:",len(unique_words_prep))
    print("Avg word per doc:",tot_words/(file_abs.count()[0]+file_nabs.count()[0]))
    print("Avg word per doc prep:",tot_words_prep/(file_abs.count()[0]+file_nabs.count()[0]))
    print("")
    print("Total words abs:",tot_words_abs)
    print("Total words prep abs:",tot_words_prep_abs)
    print("Total unique words abs:",len(unique_words_abs))
    print("Total unique words prep abs:",len(unique_words_prep_abs))
    print("Avg word per doc abs:",tot_words_abs/(file_abs.count()[0]))
    print("Avg word per doc prep abs:",tot_words_prep_abs/(file_abs.count()[0]))
    print("")
    print("Total words nabs:",tot_words_nabs)
    print("Total words prep nabs:",tot_words_prep_nabs)
    print("Total unique words nabs:",len(unique_words_nabs))
    print("Total unique words prep nabs:",len(unique_words_prep_nabs))
    print("Avg word per doc nabs:",tot_words_nabs/(file_nabs.count()[0]))
    print("Avg word per doc prep nabs:",tot_words_prep_nabs/(file_nabs.count()[0]))

if stat_train:
    #training set words statistics
    tot_words = 0
    tot_words_prep = 0
    unique_words = set()
    unique_words_prep = set()

    positive_samples_train, negative_samples_train = dsu.read_train_samples()

    for index,row in positive_samples_train.iterrows():
        print(index)
        title = row['title']
        subject = row['subject']

        if re.search(r'MISSING-DATA',title) is None:
            ws = title.split()
            tot_words += len(ws)
            for w in ws:
                unique_words.add(w)
            tokens = analyzer(title)
            tot_words_prep += len(tokens)
            for w in tokens:
                unique_words_prep.add(w)
        if re.search(r'MISSING-DATA',subject) is None:
            ws = subject.split()
            tot_words += len(ws)
            for w in ws:
                unique_words.add(w)
            tokens = analyzer(subject)
            tot_words_prep += len(tokens)
            for w in tokens:
                unique_words_prep.add(w)

    for index,row in negative_samples_train.iterrows():
        print(index)
        title = row['title']
        subject = row['subject']

        if re.search(r'MISSING-DATA',title) is None:
            ws = title.split()
            tot_words += len(ws)
            for w in ws:
                unique_words.add(w)
            tokens = analyzer(title)
            tot_words_prep += len(tokens)
            for w in tokens:
                unique_words_prep.add(w)
        if re.search(r'MISSING-DATA',subject) is None:
            ws = subject.split()
            tot_words += len(ws)
            for w in ws:
                unique_words.add(w)
            tokens = analyzer(subject)
            tot_words_prep += len(tokens)
            for w in tokens:
                unique_words_prep.add(w)

    print("Total words:",tot_words)
    print("Total words prep:",tot_words_prep)
    print("Total unique words:",len(unique_words))
    print("Total unique words prep:",len(unique_words_prep))
    print("Avg word per doc:",tot_words/(positive_samples_train.count()[0]+negative_samples_train.count()[0]))
    print("Avg word per doc prep:",tot_words_prep/(positive_samples_train.count()[0]+negative_samples_train.count()[0]))

if stat_test:
    #test set words statistics
    tot_words = 0
    tot_words_prep = 0
    unique_words = set()
    unique_words_prep = set()

    test_df = dsu.read_test_samples()

    for index,row in test_df.iterrows():
        print(index)
        title = row['title']
        subject = row['subject']

        if re.search(r'MISSING-DATA',title) is None:
            ws = title.split()
            tot_words += len(ws)
            for w in ws:
                unique_words.add(w)
            tokens = analyzer(title)
            tot_words_prep += len(tokens)
            for w in tokens:
                unique_words_prep.add(w)
        if re.search(r'MISSING-DATA',subject) is None:
            ws = subject.split()
            tot_words += len(ws)
            for w in ws:
                unique_words.add(w)
            tokens = analyzer(subject)
            tot_words_prep += len(tokens)
            for w in tokens:
                unique_words_prep.add(w)

    print("Total words:",tot_words)
    print("Total words prep:",tot_words_prep)
    print("Total unique words:",len(unique_words))
    print("Total unique words prep:",len(unique_words_prep))
    print("Avg word per doc:",tot_words/test_df.count()[0])
    print("Avg word per doc prep:",tot_words_prep/test_df.count()[0])

if stat_feature_score:
    #plot top N feature scores
    num_top_features = 30
    forest = joblib.load("models/classifier.pkl")
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)[::-1][0:num_top_features]
    X = train_data
    feature_names = vectorizer.get_feature_names()

    plt.figure()
    plt.title("Features score")
    plt.bar(range(num_top_features), importances[indices],color="r", yerr=std[indices], align="center")
    plt.xticks(range(num_top_features), [feature_names[ind] for ind in indices])
    plt.xticks(rotation=90)
    plt.xlim([-1, num_top_features])
    plt.show()















#
