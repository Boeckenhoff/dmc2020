import pandas as pd
import numpy as np

def one_hot_encoding(df, column):

    return pd.get_dummies(df, columns=column)
