import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy as cp
import csv


df = pd.read_csv("data_sample.csv", header=None, error_bad_lines=False, sep='\t',
                   names=["test id", "test date", "birth date", "test result", "patient id"])

print(df.describe())
counts = df["test id"].value_counts()
print(counts)



# X = data[data.price > 0]
# X = X[X.sqft_lot15 > 0]
# X = X.dropna(axis=0, how='any')  # drops rows in which there's a 'nan' value
# X.drop_duplicates(inplace=True)
# X.drop('id', axis=1, inplace=True)