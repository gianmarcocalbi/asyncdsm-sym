import numpy as np
import abc
from src.mltoolbox import functions


class ObjectiveFunction:
    __metaclass__ = abc.ABCMeta
    loss_func = None

    def __init__(self, y_hat_func=functions.LinearYHatFunction):
        self.y_hat_func = y_hat_func

    def compute_value(self, X, y, w):
        return np.sum(self.loss_func.f(y, self.y_hat_func(X, w))) / len(y)

    @staticmethod
    def compute_gradient(loss, y, y_hat):
        return np.sum(loss.f_gradient(y, y_hat), axis=1) / len(y)


class Metrics:
    __metaclass__ = abc.ABCMeta


class MeanSquaredError(ObjectiveFunction):
    fullname = "Mean Squared Error"
    shortname = "MSE"
    id = 'mse'
    loss_func = functions.HingeLossFunction



METRICS = {
    MeanSquaredError.id : MeanSquaredError
}

OBJ_FUNCTIONS = {
    MeanSquaredError.id : MeanSquaredError
}