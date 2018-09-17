import csv
import math

import numpy as np

from src import utils

def unisvm_dual_averaging_dataset(n, label_flip_prob=0.05):
    X = np.random.uniform(-1, 1, n)
    y = -np.sign(X)
    for i in range(len(y)):
        if np.random.uniform(0, 1) < label_flip_prob:
            y[i] *= -1
    w = np.zeros(1)
    return X.reshape(-1, 1), y, w


def eigvecsvm_dataset_from_adjacency_matrix(adj_mat, c=0.1):
    M = utils.uniform_weighted_Pn_from_adjacency_matrix(adj_mat)
    eigvals, eigvecs = np.linalg.eig(M)
    lambda2nd_index = np.argsort(np.abs(eigvals))[-2]

    u = eigvecs[:, lambda2nd_index].real
    u_abs_max_index = np.argmax(np.abs(u))
    u /= -u[u_abs_max_index]

    X = np.abs(u + c)
    y = -np.sign(u + c)
    w = np.zeros(1)
    return X.reshape(-1, 1), y, w


def unireg_dataset(n_samples):
    X = np.ones((n_samples, 1))
    if n_samples % 2 == 0:
        half = np.arange(1, 1 + n_samples / 2)
        y = np.concatenate([half, -half])
        y.sort()
    else:
        y = (np.arange(n_samples) / n_samples * 100) - 50
    w = np.zeros(1)
    return X, y, w


def svm_dual_averaging_dataset(n_samples, n_features, label_flip_prob=0.05):
    X = []
    y = np.zeros(n_samples)
    # w = np.random.normal(0, 1, n_features)
    w = np.ones(n_features)
    """
    w_norm_2 = math.sqrt(np.inner(w, w))
    if w_norm_2 > 5:
        w = w / w_norm_2 * 5"""
    # e = np.random.normal(0, 1, n_features)

    for i in range(n_samples):
        x = np.random.normal(0, 1, n_features)
        x /= math.sqrt(np.inner(x, x))
        X.append(x)
        flip = 1
        if np.random.uniform(0, 1) < label_flip_prob:
            flip = -1
        y[i] = flip * np.sign(x.T.dot(w))

        # x = np.sign(x.T.dot(e)) * x  # + np.random.normal(error_mean, error_std_dev)

    return np.array(X), y, w


def load_susy_svm_dataset(n_samples):
    X = []
    y = []
    with open('./dataset/SUSY.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if 0 <= i < n_samples:
                y.append(row[-1])
                X.append(row[1:])
            elif i >= n_samples:
                break
            i += 1
    X = np.c_[np.ones(n_samples), np.array(X, dtype=float)]
    y = np.array(y, dtype=float)
    w = np.zeros(X.shape[1])
    return X, y, w


def load_slice_localization_reg_dataset(n_samples):
    X = []
    y = []
    with open('./dataset/slice_localization_dataset.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        i = -1
        for row in reader:
            if 0 <= i < n_samples:
                y.append(row[-1])
                X.append(row[0:-1])
            elif i >= n_samples:
                break
            i += 1
    X = np.c_[np.ones(n_samples), np.array(X, dtype=float)]
    y = np.array(y, dtype=float)
    w = np.zeros(X.shape[1])
    return X, y, w


def load_appliances_energy_reg_dataset(n_samples):
    X = []
    y = []
    with open('./dataset/appliances_energy_prediction_training.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        i = -1
        for row in reader:
            if 0 <= i < n_samples:
                y.append(row[1])
                X.append(row[2:-3])
            elif i >= n_samples:
                break
            i += 1
    X = np.c_[np.ones(n_samples), np.array(X, dtype=float)]
    y = np.array(y, dtype=float)
    w = np.zeros(X.shape[1])
    return X, y, w


def reg_dataset(n_samples, n_features, error_mean=0, error_std_dev=1):
    # w = np.random.uniform(-50, +50, n_features + 1)
    w = np.ones(n_features + 1)
    X = np.c_[np.ones(n_samples), np.random.uniform(0, 1, (n_samples, n_features))]
    y = X.dot(w) + np.random.normal(error_mean, error_std_dev, n_samples)
    return X, y, w


def reg_dataset_from_function(n_samples, n_features, func, domain_radius=0.5, domain_center=0.5,
        error_mean=0, error_std_dev=1):
    """
    Parameters
    ----------
    n_samples : int
        Amount of samples to generate in the training set.

    n_features : int
        Amount of feature each sample will have.

    func : callable
        Generator function: takes x and return y. Then the sample will be (x,y).

    domain_radius : float
        Threshold for samples' domains. If a value of

    domain_center : float
    error_mean : float
    error_std_dev : float
    error_coeff : float


    Returns
    -------
    X : numpy.ndarray
        Matrix of samples
    y : numpy.array
        Matrix of function values for such samples.
    """

    X = []
    y = np.zeros(n_samples)
    w = np.ones(n_features + 1)
    K = np.random.uniform(domain_center - domain_radius, domain_center + domain_radius, n_features)

    for i in range(n_samples):
        x = np.ones(n_features + 1)
        for j in range(n_features):
            x[j + 1] = np.random.uniform(-K[j] + domain_radius, K[j] + domain_radius)
        X.append(x)
        y[i] = func(x, w) + np.random.normal(error_mean, error_std_dev)

    return [np.array(X), y, w]


def sample_from_function_old(n_samples, n_features, func, domain_radius=1, domain_center=0,
        subdomains_radius=1, error_mean=0, error_std_dev=1):
    """
    Parameters
    ----------
    n_samples : int
        Amount of samples to generate in the training set.
    n_features : int
        Amount of feature each sample will have.
    func : callable
        Generator function: takes x and return y. Then the sample will be (x,y).
    domain_radius : float
        Threshold for samples' domains. If a value of
    domain_center : float
    subdomains_radius : float
    error_mean : float
    error_std_dev : float
    error_coeff : float
    Returns
    -------
    X : numpy.ndarray
        Matrix of samples
    y : numpy.array
        Matrix of function values for such samples.
    """

    X = []
    y = np.zeros(n_samples)
    w = np.ones(n_features)
    features_domain = []

    for j in range(n_features):
        feature_j_domain_center = np.random.uniform(
            domain_center - domain_radius + subdomains_radius,
            domain_center + domain_radius - subdomains_radius
        )
        features_domain.append(
            (feature_j_domain_center - subdomains_radius, feature_j_domain_center + subdomains_radius)
        )

    for i in range(n_samples):
        x = np.zeros(n_features)
        for j in range(n_features):
            x[j] = np.random.uniform(features_domain[j][0], features_domain[j][1])
        X.append(x)
        y[i] = func(x, w) + np.random.normal(error_mean, error_std_dev)

    return np.array(X), y
