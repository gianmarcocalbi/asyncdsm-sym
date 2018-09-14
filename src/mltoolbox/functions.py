# import the necessary packages
import csv
import math

import numpy as np

from src import utils


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))



def estimate_linear_regression_beta(_X, _y):
    # return np.linalg.inv(_X.T.dot(_X)).dot(_X.T).dot(_y)
    return np.linalg.lstsq(_X, _y, rcond=None)[0]


def estimate_unbiased_beta(_X, _y):
    _X = np.delete(_X, 0, axis=1)
    _y = _y - 1
    return np.concatenate(([1], estimate_biased_beta(_X, _y)))


def estimate_biased_beta(_X, _y):
    # return np.linalg.inv(_X.T.dot(_X)).dot(_X.T).dot(_y)
    return np.linalg.lstsq(_X, _y, rcond=None)[0]

"""
def compute_mse(w, X, y, activation_func, y_hat_func):
    N = X.shape[0]
    predictions = activation_func(y_hat_func(X, w))
    linear_error = np.absolute(y - predictions)
    return np.sum(np.power(linear_error, 2)) / N


def compute_mae(w, X, y, activation_func, y_hat_func):
    N = X.shape[0]
    predictions = activation_func(y_hat_func(X, w))
    linear_error = np.absolute(y - predictions)
    return np.sum(linear_error) / N


def compute_score(w, X, y, activation_func, y_hat_func):
    return accuracy_score(y, activation_func(y_hat_func(X, w)), normalize=True)
"""
