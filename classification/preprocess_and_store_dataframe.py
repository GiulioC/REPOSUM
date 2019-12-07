from sklearn.externals import joblib
import text_processing as tp
import dataset_utils as dsu
import pandas as pd
import os

out_dir = "preprocessed_data"

vectorizer = joblib.load("models/vectorizer.pkl")
analyzer = vectorizer.build_analyzer()

data_zip = [
    dsu.read_dataset_UK_ethos(True),
    dsu.read_dataset_UK_ethos(False)
]
out_files_zip = [
    "ethos_abs_preprocessed.csv",
    "ethos_no_abs_preprocessed.csv"
]

for data, out_file in zip(data_zip,out_files_zip):
    data = dsu.read_dataset_UK_ethos(True)
    print("data with abs size:",data.count())
    data_text = tp.lemmatize_data(data['titolo'])
    data_text = tp.preprocess_text_data(data_text, analyzer)
    print("preprocessed data size: {}".format(len(data_text)))
    data.loc[:,"preprocessed_data"] = data_text

    data.to_csv(
        os.path.join(out_dir, out_file),
        index=None,
        columns=["id","titolo","autore","univ","publisher","anno","abs","tipo","argomento","preprocessed_data"]
    )
