import dataset_utils as dsu
import pandas as pd
import numpy as np

def shuffle_df(df):
    s = np.arange(df.count()[0])
    np.random.shuffle(s)
    return df.iloc[s]

phils_df = pd.read_csv("data/philosophy.csv")
no_phils_df = pd.read_csv("data/no_philosophy.csv")

phils_df = shuffle_df(phils_df)
no_phils_df = shuffle_df(no_phils_df)

phils_df_test = philsDF.iloc[:500]
phils_df_train = philsDF.iloc[500:]
no_phils_df_test = nophilsDF.iloc[:500]
no_phils_df_train = nophilsDF.iloc[500:phils_df_train.count()[0]*10+500]

phils_df_train.to_csv("data/philosophy_train.csv", index=None)
phils_df_test.to_csv("data/philosophy_test.csv", index=None)
no_phils_df_train.to_csv("data/nophilosophy_train.csv", index=None)
no_phils_df_test.to_csv("data/nophilosophy_test.csv", index=None)
