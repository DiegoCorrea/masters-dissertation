from sklearn.metrics import mean_squared_error

from math import sqrt

from src.config.variables import value_label, original_value_label


def rmse(df):
    return sqrt(mean_squared_error(df[value_label].tolist(), df[original_value_label].tolist()))
