import pandas as pd

# column 2: titolo
# column 5: lingua
# column 9: autore
# column 11: CIA
# column 13: ID documento
# column 14: ISBN
# column 17: fonte
# column 20: keywords
# column 21: pagine
# column 22: dipartimento
# column 23: data
# column 24: abstract
# column 25: url
# column 26: #dissertazione/tesi
# column 27: luogo
# column 28: istituzione
# column 29: num_pagine
# column 30: subject
# column 31: tipo di fonte
# column 32: relatore
tesi_US = "../../data/tesi_US/US_PhD_dissertations.xlsx"

# uketdterms:ethosid
# dc:title
# dc:creator
# uketdterms:institution
# dc:publisher
# dcterms:issued
# dcterms:abstract
# dc:type
# uketdterms:qualificationname
# uketdterms:qualificationlevel
# dc:identifier
# dc:source
# dc:subjectxsi
# dc:subject
tesi_UK_abs_ethos = "../../data/tesi_UK/tab_separated_value/Synapta_EThOS.tsv"
tesi_UK_no_abs_ethos = "../../data/tesi_UK/tab_separated_value/Synapta_EThOS_No_Abstract.tsv"

phil_train_file = "data/philosophy_train.csv"
no_phil_train_file = "data/nophilosophy_train.csv"
test_samples_file = "data/test_set_1000.tsv"

def read_dataset_US(**kwargs):
    return pd.read_excel(
        tesi_US,
        usecols=[2,5,6,9,11,13,14,17,20,21,22,23,24,25,26,27,28,29,30,31,32],
        names=['titolo','lingua','anno','autore','CIA','id','ISBN','fonte','keywords','pagine','dipartimento','data','abstract','url','tipo_doc','luogo','istituzione','num_pagine','subject','tipo_fonte','relatore'],
        **kwargs
    )

def read_dataset_UK_ethos(abs=True):
    if abs:
        return pd.read_csv(tesi_UK_abs_ethos, delimiter="\t", skiprows=1, names=["id", "titolo","autore","univ","publisher","anno","abs","tipo","qname", "qlevel", "identifier", "source", "sxsi", "argomento", "unnamed"])
    else:
        return pd.read_csv(tesi_UK_no_abs_ethos, delimiter="\t", skiprows=1, names=["id","titolo","autore","univ","publisher","anno","tipo","qname", "qlevel", "identifier", "source", "sxsi", "argomento", "unnamed"])

def read_train_samples():
    phils = pd.read_csv(phil_train_file)
    nphils = pd.read_csv(no_phil_train_file)
    return phils, nphils

def read_test_samples():
    return pd.read_csv(test_samples_file, delimiter="\t", names=['title','creator','university','publisher', 'year','abstract','type','subject','id','philosophy'])
