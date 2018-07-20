# import the necessary packages
import numpy as np
import abc, types, warnings, math
from sklearn.metrics import accuracy_score
from src import utils


class LossFunctionAbstract:
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def compute_value(y, y_hat_f):
        raise NotImplementedError('f method not implemented in LossFunctionAbstract --> child class')

    @staticmethod
    @abc.abstractmethod
    def compute_gradient(y, y_hat_f, y_hat_f_gradient):
        raise NotImplementedError('f_gradient method not implemented in LossFunctionAbstract child class')


class HingeLossFunction(LossFunctionAbstract):
    @staticmethod
    def compute_value(y, y_hat_f):
        return (1 - y * y_hat_f).clip(min=0)

    @staticmethod
    def compute_gradient(y, y_hat_f, y_hat_f_gradient):
        h = HingeLossFunction.compute_value(y, y_hat_f)
        return np.sum(((-y * y_hat_f_gradient.T) * np.sign(h)).T, axis=0)

    @staticmethod
    def compute_gradient2(y, y_hat_f, y_hat_f_gradient):
        N = len(y)
        P = y_hat_f_gradient.shape[1]
        h = HingeLossFunction.compute_value(y, y_hat_f)
        G = np.zeros((N, P))
        for i in range(N):
            if h[i] > 0:
                G[i] = - y[i] * y_hat_f_gradient[i]
            elif h[i] == 0:
                for j in range(len(y_hat_f_gradient[i])):
                    inf = min(0, y_hat_f_gradient[i][j])
                    sup = max(0, y_hat_f_gradient[i][j])
                    G[i][j] = np.random.uniform(inf, sup)
                G[i] *= -y[i]
        G = np.sum(G, axis=0)

        return G


class EdgyHingeLossFunction(LossFunctionAbstract):
    @staticmethod
    def compute_value(y, y_hat_f):
        return (1 - y * np.sign(y_hat_f)).clip(min=0)

    @staticmethod
    def compute_gradient(y, y_hat_f, y_hat_f_gradient):
        h = HingeLossFunction.compute_value(y, y_hat_f)
        return (-y * y_hat_f_gradient.T).T * np.sign(h)

    @staticmethod
    def compute_gradient2(y, y_hat_f, y_hat_f_gradient):
        N = len(y)
        P = y_hat_f_gradient.shape[1]
        h = HingeLossFunction.compute_value(y, y_hat_f)
        G = np.zeros((N, P))
        for i in range(N):
            if h[i] > 0:
                G[i] = - y[i] * y_hat_f_gradient[i]
            elif h[i] == 0:
                for j in range(len(y_hat_f_gradient[i])):
                    inf = min(0, y_hat_f_gradient[i][j])
                    sup = max(0, y_hat_f_gradient[i][j])
                    G[i][j] = np.random.uniform(inf, sup)
                G[i] *= -y[i]
        G = np.sum(G, axis=0)

        return G


class SquaredLossFunction(LossFunctionAbstract):
    @staticmethod
    def compute_value(y, y_hat_f):
        return np.power(y - y_hat_f, 2)

    @staticmethod
    def compute_gradient(y, y_hat_f, y_hat_f_gradient):
        # the minus sign from the derivative of "- y_hat_f" is represented as follows:
        #   - (y - y_hat_f) = y_hat_f - y
        return y_hat_f_gradient.T.dot(y_hat_f - y)


class YHatFunctionAbstract:
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def compute_value(X, W):
        raise NotImplementedError('f method not implemented in YHatFunctionAbstract child class')

    @staticmethod
    @abc.abstractmethod
    def compute_gradient(X, W):
        raise NotImplementedError('f_gradient method not implemented in YHatFunctionAbstract child class')


class LinearYHatFunction(YHatFunctionAbstract):
    @staticmethod
    def compute_value(X, W):
        return X.dot(W)

    @staticmethod
    def compute_gradient(X, W):
        return X


class ParaboloidYHatFunction(YHatFunctionAbstract):
    @staticmethod
    def compute_value(X, W):
        return np.power(X, 2).dot(W)

    @staticmethod
    def compute_gradient(X, W):
        return np.power(X, 2)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class SampleGenerator:
    def __init__(self):
        # a lot of useful parameters
        pass


def generate_unidimensional_svm_training_set_from_expander_adj_mat(adj_mat, c=0.1):
    Pn = utils.Pn_from_adjacency_matrix(adj_mat)
    eigvals, eigvecs = np.linalg.eig(Pn)
    lambda2nd_index = np.argsort(np.abs(eigvals))[-2]

    u = eigvecs[:, lambda2nd_index].real
    u_abs_max_index = np.argmax(np.abs(u))
    u /= -u[u_abs_max_index]

    X = np.abs(u + c)
    y = -np.sign(u + c)
    w = np.zeros(1)
    return X.reshape(-1, 1), y, w


def generate_unidimensional_regression_training_set(n_samples):
    X = np.ones((n_samples, 1))
    if n_samples % 2 == 0:
        half = np.arange(1, 1 + n_samples / 2)
        y = np.concatenate([half, -half])
        y.sort()
    else:
        y = (np.arange(n_samples) / n_samples * 100) - 50
    w = np.zeros(1)
    return X, y, w


def generate_svm_dual_averaging_training_set(n_samples, n_features, label_flip_prob=0.05):
    X = []
    y = np.zeros(n_samples)
    w = np.random.normal(0, 1, n_features)
    w_norm_2 = math.sqrt(np.inner(w, w))
    if w_norm_2 > 5:
        w = w / w_norm_2 * 5
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


def generate_regression_training_set(n_samples, n_features, error_mean=0, error_std_dev=1):
    w = np.random.uniform(-50, +50, n_features + 1)
    # w = 50 * np.ones(n_features + 1)
    # X = np.c_[np.ones(n_samples), np.random.normal(0,1,(n_samples, n_features))]
    X = np.c_[np.ones(n_samples), np.random.uniform(0, 1, (n_samples, n_features))]
    y = X.dot(w) + np.random.normal(error_mean, error_std_dev, n_samples)

    return X, y, w


def generate_regression_training_set_from_function(n_samples, n_features, func, domain_radius=0.5, domain_center=0.5,
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
            # todo: implement threshold
        X.append(x)
        y[i] = func(x, w) + np.random.normal(error_mean, error_std_dev)

    return np.array(X), y


def linear_function(_X, _w):
    return _X.dot(_w)


def sphere_function(_X, _w):
    return np.tanh(np.sum(np.power(_X, 2)))


def rosenbrock_function(_X, _w):
    n = len(_X)
    v = 0
    for i in range(0, n - 1):
        v += 100 * (_X[i + 1] - _X[i] ** 2) + (1 - _X[i]) ** 2
    return v


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
