# import the necessary packages
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import math

np.random.seed(2894)


class TrainingModel:
    def __init__(self, X, y, activation_function, learning_rate):
        self.X = X  # sample instances matrix
        self.y = y  # sample function's values array
        # todo: remove or exploit self.f
        self.activation_function = activation_function  # network function (usually just like <X,y>)
        self.learning_rate = learning_rate  # learning rate alpha

        # bias inserted as w0 = (1,...,1)
        self.X = np.c_[np.ones((X.shape[0])), X]

        # weight vector random sampled from uniform distribution
        self.W = np.random.uniform(size=(X.shape[1] + 1,))

        self.squared_loss_log = []
        self.score_log = []

    def score(self):
        if len(self.score_log) > 0:
            return self.score_log[-1]
        else:
            return 0

    def squared_loss(self):
        if len(self.squared_loss_log) > 0:
            return self.squared_loss_log[-1]
        else:
            return self._compute_squared_loss()

    def _compute_squared_loss(self):
        # get the prediction
        predictions = self.activation_function(self.X.dot(self.W))

        # compute the linear error as the difference between predicted y and real y values
        linear_error = predictions - self.y

        # compute the loss function loss(f(X), y)
        return np.sum(linear_error ** 2) / (2 * self.X.shape[0])

    def gradient_descent_step(self):
        """
        Gradient descent step function.
        :return: None
        """
        self._gradient_descent_weight_update(self.X, self.y)

    def stochastic_gradient_descent_step(self):
        """
        Stochastic gradient descent step function. Computes a step of the SGD.
        :return: None
        """
        # pick a random row index in [0, height of X)
        pick = np.random.randint(0, self.X.shape[0])

        # compute gradient weight update consider only pick-th example
        self._gradient_descent_weight_update(self.X[pick], self.y[pick])

    def batch_gradient_descent_step(self, batch_size):
        """
        Batch gradient descent step function.
        :param batch_size: size of the batch of X on which computing the gradient
        :return: None
        """
        # determine the mini batch upon which computing the SGD
        batch_indices = np.random.choice(self.X.shape[0], min(batch_size, self.X.shape[0]), replace=False)

        # extract subsamples
        sub_X = np.take(self.X, batch_indices, axis=0)
        sub_y = np.take(self.y, batch_indices, axis=0)

        # compute weight update on the subsamples
        self._gradient_descent_weight_update(sub_X, sub_y)

    def _gradient_descent_weight_update(self, X, y):
        """
        Gradient descent core function.
        Updates the weight following the direction of the gradient.
        :param X: examples matrix
        :param y: examples values for the target function (oracle-provided)
        :return: None
        """
        N = self.X.shape[0]  # amount of samples in original X matrix
        M = X.shape[0]  # amount of samples in current batch

        ## BEGIN: only for statistical purpose

        # get the prediction
        predictions = self.activation_function(np.power(self.X,2).dot(self.W))

        # compute the linear error as the difference between predicted y and real y values
        linear_error = predictions - self.y

        # compute the loss function loss(f(X), y)
        mean_square_error = np.sum(np.apply_along_axis(lambda x : x*x, 0, linear_error)) / (2 * N)
        self.squared_loss_log.append(mean_square_error)
        self.score_log.append(1 - sum(np.apply_along_axis(abs, 0, linear_error)) / N)

        ## END: only for statistical purpose

        # compute the gradient
        batch_linear_error = self.activation_function(np.power(X,2).dot(self.W)) - y
        # gradient = X.T.dot(batch_linear_error) / M

        gradient = np.power(X, 2).T.dot(batch_linear_error) / M

        # update W following the steepest gradient descent
        self.W -= self.learning_rate * gradient

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

def perceptron_loss_function(_X, _y, _w):
    pass

def hinge_loss_function(_X, _y, _W):
    pass


class SampleGenerator:
    def __init__(self):
        # a lot of useful parameters
        pass

    @staticmethod
    def sample_from_function(n_samples, n_features, func, domain, biased=False):
        X = []
        y = []
        w = []

        for _ in range(n_features):
            w.append(np.random.uniform(0, 1))

        for i in range(n_samples):
            x = np.random.uniform(-domain, domain, n_features)
            X.append(x)
            y.append(func(x, w) +  np.random.choice([-1,1]) * np.random.rand() * int(biased))
        return np.array(X), np.array(y)


def linear_function(_X, _w):
    return _X.dot(_w)

def sphere_function(_X, _w):
    return np.tanh(np.sum(np.power(_X,2)))

def rosenbrock_function(_X, _w):
    n = len(_X)
    v = 0
    for i in range(0, n-1):
        v += 100 * (_X[i+1] - _X[i] ** 2) + (1 - _X[i]) ** 2
    return v
