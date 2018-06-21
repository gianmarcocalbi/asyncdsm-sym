import numpy as np
import abc
from src import mltoolbox


class ObjectiveFunction:
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def compute(loss, y, y_hat):
        return np.sum(loss.f(y, y_hat)) / len(y)

    @staticmethod
    def compute_gradient(loss, y, y_hat):
        return np.sum(loss.f_gradient(y, y_hat)) / len(y)
        loss_f_gradient = self.loss.f_gradient(self.y, y_hat_f, y_hat_f_gradient)


class Metrics:
    __metaclass__ = abc.ABCMeta


class MeanSquaredError(Metrics):
    fullname = "Mean Squared Error"
    shortname = "MSE"
    id = 'mse'
    loss = mltoolbox.SquaredLossFunction

    def __init__(self):
        pass

    @staticmethod
    def compute(w, X, y, activation_func, y_hat_func):
        N = X.shape[0]
        predictions = activation_func(y_hat_func(X, w))
        linear_error = np.absolute(y - predictions)
        return np.sum(np.power(linear_error, 2)) / N


METRICS = {
    MeanSquaredError.id : MeanSquaredError
}

OBJ_FUNCTIONS = {
    MeanSquaredError.id : MeanSquaredError
}