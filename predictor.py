import numpy as np
import pred_util as pu
import matplotlib.pyplot as plt
import pandas as pd

RARENESS = 1000


data = pu.cleaned_data('data_sample.csv')

# we shall now identify a relatively rare feature - which occurs around 1,000 times (1,
# 000 patients had taken it), and then construct a feature vector for every patient that has done
# it, considering the rare test as the label of this features vector, and all the tests that has
# been taken more times than the chosen one - as the features. if a patient hasn't done a
# certain test, it's value in the feature vector would be 0.

times_taken = data["test id"].value_counts()
least_common = times_taken[times_taken >= RARENESS]
rare_test = least_common[-1:-2:-1]
feature_tests = least_common[:-1:]
tests_dic = feature_tests.to_dict()
print(tests_dic, rare_test)
# least_common_df = df.loc[df["test id"] == least_common]

# print(np.shape(counts))
# print(df['test result'])
# print(df.describe(include='all'))

# now that we have cleaned and organized our data - we would like to create a feature vector for
# each patient. the features are of course the different tests. in order to do so - we would
# establish for each patient a dictionary of the tests he has done in which their value would be
# the test result, and a dictionary of all tests, in which their value would be their mean value
# over all patients. this way, when establishing the feature vectors - we would substitute the
# mean value of a test result if the patient hasn't done it.