import pandas as pd


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
