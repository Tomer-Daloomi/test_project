import numpy as np
import pred_util as pu
import matplotlib.pyplot as plt
import pandas as pd

RARENESS = 1000
TR = 0.7
C = 1e5
ALPHA = 0.1

data = pu.cleaned_data('data_sample.csv')

# we shall now identify a relatively rare feature - which occurs around 1,000 times (1,
# 000 patients had taken it), and then construct a feature vector for every patient that has done
# it, considering the rare test as the label of this features vector, and all the tests that has
# been taken more times than the chosen one - as the features. if a patient hasn't done a
# certain test, it's value in the feature vector would be 0.

times_taken = data["test id"].value_counts()  # ordered from top to bottom
more_common = times_taken[times_taken >= RARENESS]

rare_test = np.array(more_common[-1:-2:-1].index)  # the id str of the chosen rare test
tests_array = np.array(more_common[:-1:].index)  # all the id's of the tests that were taken
# more often than the 'rare test' as keys

X, y = pu.features_and_labels(data, rare_test, tests_array)

# It is now time for training and testing! let us begin with putting 30% of the data aside for
# testing, and keeping 70% for training

n = np.shape(X)[0]

X_tr = X[0:int(TR * n)]
X_tst = X[int(TR * n):]

y_tr = y[0:int(TR * n)]
y_tst = y[int(TR * n):]

# we would like to check, whether the y values could be any number within some limits, or if they
# could only get the value of a discrete group of test results (i.e. a rational number between
# [1,100] or an int in the group {0, 1, 2, 3, 4}).

# in order to determine - we should use the threshold value 0.5 for repetitiveness of labels,
# which effectively means:
num_of_unique_labels = len(np.unique(y))
num_of_labels = len(y)
discrete = True if (num_of_unique_labels / num_of_labels) < 0.5 else False


H = []
# if the labels could only get few discrete values, we should obviously treat our problem as a
# classification problem.
if discrete:
    H.append(pu.logistic_regression(C, X_tr, y_tr))
# otherwise - it should be treated as a regression problem
else:
    H.append(pu.lasso_regression(ALPHA, X_tr, y_tr))
    H.append(pu.linear_regression(X_tr, y_tr))


# in order to check how well does the new hypothesis work, we shall calculate the error:
for h in H:
    tr_error = pu.loss_0_1(h, X_tr, y_tr)
    tst_error = pu.loss_0_1(h, X_tst, y_tst)
    print('training error: ', tr_error, ' test error: ', tst_error)


