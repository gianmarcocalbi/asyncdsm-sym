import numpy as np
import abc
from sklearn.metrics import accuracy_score


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


class ContinuousHingeLossFunction(LossFunctionAbstract):
    @staticmethod
    def compute_value(y, y_hat_f):
        return 1 - y * y_hat_f

    @staticmethod
    def compute_gradient(y, y_hat_f, y_hat_f_gradient):
        return np.sum(-y * y_hat_f_gradient, axis=0)


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
    def compute_gradient_iteratively(y, y_hat_f, y_hat_f_gradient):
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
        return y_hat_f_gradient.T.dot(y_hat_f - y) / 2


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


class Metrics:
    __metaclass__ = abc.ABCMeta
    loss_func = None
    y_hat_func = LinearYHatFunction

    def compute_value(self, X, y, w):
        return np.sum(self.loss_func.compute_value(y, self.y_hat_func.compute_value(X, w))) / len(y)

    def compute_gradient(self, X, y, w):
        return self.loss_func.compute_gradient(
            y,
            self.y_hat_func.compute_value(X, w),
            self.y_hat_func.compute_gradient(X, w)
        ) / len(y)


class MeanSquaredError(Metrics):
    fullname = "Mean Squared Error"
    shortname = "MSE"
    id = 'mse'
    loss_func = SquaredLossFunction


class ContinuousHingeLoss(Metrics):
    fullname = "Continuous Hinge Loss"
    shortname = "cHL"
    id = "cont_hinge_loss"
    loss_func = ContinuousHingeLossFunction


class HingeLoss(Metrics):
    fullname = "Hinge Loss"
    shortname = "HL"
    id = 'hinge_loss'
    loss_func = HingeLossFunction


class EdgyHingeLoss(Metrics):
    fullname = "Edgy Hinge Loss"
    shortname = "eHL"
    id = 'edgy_hinge_loss'
    loss_func = EdgyHingeLossFunction


class Score(Metrics):
    fullname = "Score"
    shortname = "score"
    id = 'score'
    loss_func = EdgyHingeLossFunction

    def compute_value(self, X, y, w):
        # return accuracy_score(y, np.sign(self.y_hat_func.compute_value(X, w)))
        return 1 - (super().compute_value(X, y, w) / 2)
        # return 1 - (np.sum(self.loss_func.compute_value(y, np.sign(self.y_hat_func.compute_value(X, w)))) / len(y) / 2)


METRICS = {
    MeanSquaredError.id: MeanSquaredError(),
    HingeLoss.id: HingeLoss(),
    EdgyHingeLoss.id: EdgyHingeLoss(),
    Score.id: Score(),
    ContinuousHingeLoss.id : ContinuousHingeLoss()
}
