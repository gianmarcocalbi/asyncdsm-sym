import numpy as np
import abc
from src.mltoolbox import functions


class Metrics:
    __metaclass__ = abc.ABCMeta
    loss_func = None
    y_hat_func = functions.LinearYHatFunction

    def compute_value(self, X, y, w):
        return np.sum(self.loss_func.compute_value(y, self.y_hat_func.compute_value(X, w))) / len(y)

    def compute_gradient(self, X, y, w):
        return np.sum(self.loss_func.compute_gradient(
            y,
            self.y_hat_func.compute_value(X, w),
            self.y_hat_func.compute_gradient(X, w)
        ), axis=1) / len(y)


class MeanSquaredError(Metrics):
    fullname = "Mean Squared Error"
    shortname = "MSE"
    id = 'mse'
    loss_func = functions.SquaredLossFunction


class HingeLoss(Metrics):
    fullname = "Hinge Loss"
    shortname = "HL"
    id = 'hinge_loss'
    loss_func = functions.HingeLossFunction


class EdgyHingeLoss(Metrics):
    fullname = "Edgy Hinge Loss"
    shortname = "eHL"
    id = 'edgy_hinge_loss'
    loss_func = functions.EdgyHingeLossFunction


class Score(Metrics):
    fullname = "Score"
    shortname = "score"
    id = 'score'
    loss_func = functions.EdgyHingeLossFunction

    def compute_value(self, X, y, w):
        return 1 - (super().compute_value(X, y, w) / 2)
        # return 1 - (np.sum(self.loss_func.compute_value(y, np.sign(self.y_hat_func.compute_value(X, w)))) / len(y) / 2)


METRICS = {
    MeanSquaredError.id: MeanSquaredError(),
    HingeLoss.id: HingeLoss(),
    EdgyHingeLoss.id: EdgyHingeLoss(),
    Score.id: Score()
}
