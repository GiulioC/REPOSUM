import dataset_utils as dsu
import pandas as pd
import re

phils = {
    "title":[],
    "subject":[],
    "id":[]
}
no_phils = {
    "title":[],
    "subject":[],
    "id":[]
}

def scan_philosophy(data):
    for index,row in data.iterrows():
        print(index, end="\r")

        if re.search(r'[P|p]hilosop',str(row['argomento'])) is not None:
            append_data(phils, row)
        else:
            append_data(no_phils, row)

def append_data(dictionary, row):
    dictionary['title'].append(row['titolo'].strip())
    dictionary['subject'].append(row['argomento'].strip())
    dictionary['id'].append(row['id'])

print("Scanning file with abstracts")
scan_philosophy(dsu.read_dataset_UK_ethos(True))
print("Scanning file without abstracts")
scan_philosophy(dsu.read_dataset_UK_ethos(False))

pd.DataFrame(phils).to_csv("data/philosophy.csv", index=None)
pd.DataFrame(no_phils).to_csv("data/no_philosophy.csv", index=None)
