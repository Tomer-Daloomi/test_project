import pandas as pd
import numpy as np
from sklearn import linear_model


def cleaned_data(address):
    """
    gets the address of the medical tests csv file, parses it, organizing and cleaning it -
    returning the data for farther use

    :param address: the file's address
    :return: a DataFrame object with all the relevant data
    """

    df = pd.read_csv(address, header=None, error_bad_lines=False, sep='\t',
                     names=["test id", "test date", "birth date", "test result", "patient id"])

    # first - we would like to remove the test dates and birth dates - and turn both into a new
    # column of "Age of taking the test". the age would be written in days, for comfort reasons.
    df["test date"] = pd.to_datetime(df["test date"], errors="coerce")
    df["birth date"] = pd.to_datetime(df["birth date"], errors="coerce")
    # the error="coerce" means that if an error occurs, turn the problematic value into "NaN"

    df["age"] = df["test date"].sub(df["birth date"], axis=0)

    # and convert the 'Timedelta' object into a float of the total number of years
    df["age"] = df["age"].dt.total_seconds() / (24 * 60 * 60 * 365)

    # By using df.describe() we can see that some ages are negative, and others return "NaN".
    # we shall eliminate these lines. (we could as well fill them with the mean value of their
    # category, but we have a lot of data, so the effect would be minor).

    df = df[df['age'] > 0]
    df = df.dropna(axis=0, how='any')

    # now we no longer have use for these two columns
    df.drop('test date', axis=1, inplace=True)
    df.drop('birth date', axis=1, inplace=True)

    # let us also get rid of all the test results that aren't plain digits, but rather has order
    # relation (i.e. '3 >' and such).
    df = df[df["test result"].str.isdigit()]
    df["test result"] = df["test result"].astype('float')
    df = df[df["test result"] != 0]

    return df


def features_and_labels(data, rare_test, tests_array):
    """
    the function creates a bundle of n feature vectors, a vector for every patient that took the
    "rare test", each vector is of length d.

    :param data:
    :param rare_test:
    :param tests_array:
    :return: n X d matrix of features, and n X 1 vector of labels
    """

    # first we create a list of the patients that have taken the rare test
    # and a labels vector with the rare test results

    rare_patients = []
    y = []

    for index, row in data.iterrows():
        if row['test id'] == rare_test:
            rare_patients.append(row['patient id'])
            y.append(row['test result'])

    # then, we create a feature vector for each of these patients, containing their age,
    # and all their test result, out of the test that are enlisted in tests_dic

    features_matrix = []

    for patient in rare_patients:

        feature_vector = []
        patient_df = data[data['patient id'] == patient]

        # first we add the patient's age to the feature vector
        feature_vector.append(patient_df['age'][patient_df['age'].first_valid_index()])

        # then we add the relevant tests results to the vector
        patient_tests_dic = pd.Series(patient_df['test result'].values, index=patient_df[
            'test id']).to_dict()

        for test in tests_array:
            if test in patient_tests_dic:
                feature_vector.append(patient_tests_dic[test])
            else:
                feature_vector.append(0)

        features_matrix.append(feature_vector)

    X = np.matrix(features_matrix)
    y = np.array(y).T

    return X, y


def logistic_regression(C, X, y):

    log_reg = linear_model.LogisticRegression(C, multi_class='ovr')
    # we shall create a Nearest Neighbors classifier
    log_reg.fit(X, y)

    return log_reg


def lasso_regression(alpha, X_tr, y_tr):

    lasso = linear_model.Lasso(alpha=0.1)
    lasso.fit(X_tr, y_tr)

    return lasso


def linear_regression(X_tr, y_tr):

    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_tr, y_tr)

    return lin_reg


def loss_0_1(h, X, y):
    """
    computes the test error using the 0-1 loss function

    :param h: d X 1 matrix, where d- is the dimension of the hypothesis vector
    :param X: n X d matrix with the samples
    :param y: d X 1 matrix with the labels of the samples
    :return:  0-1 loss for the hypothesis. (sign of inner product as label)
    """
    n = X.shape[0]

    pred = np.inner(X, h)
    diff = sum([1 if y[j] != pred[j] else 0 for j in range(n)])
    error = diff / n

    return error
