import pandas as pd


def make_df(cols, ind):
    """ Make a DataFrame """
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)
