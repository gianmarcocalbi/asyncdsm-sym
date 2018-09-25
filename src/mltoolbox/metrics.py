import numpy as np
import abc


class LossFunctionAbstract:
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def compute_value(X, y, w):
        raise NotImplementedError('f method not implemented in LossFunctionAbstract --> child class')

    @staticmethod
    @abc.abstractmethod
    def compute_gradient(X, y, w):
        raise NotImplementedError('f_gradient method not implemented in LossFunctionAbstract child class')


class SquaredLossFunction(LossFunctionAbstract):
    @staticmethod
    def compute_value(X, y, w):
        return np.power(y - X.dot(w), 2)

    @staticmethod
    def compute_gradient(X, y, w):
        # the minus sign from the derivative of "- X.dot(w)" is represented as follows:
        #   - (y - X.dot(w)) = X.dot(w) - y
        return (X.T * (X.dot(w) - y) / 2).T


class ContinuousHingeLossFunction(LossFunctionAbstract):
    @staticmethod
    def compute_value(X, y, w):
        return 1 - y * X.dot(w)

    @staticmethod
    def compute_gradient(X, y, w):
        return (-y * X.T).T


class HingeLossFunction(LossFunctionAbstract):
    @staticmethod
    def compute_value(X, y, w):
        return (1 - y * X.dot(w)).clip(min=0)

    @staticmethod
    def compute_gradient(X, y, w):
        h = HingeLossFunction.compute_value(X, y, w)
        return ((-y * X.T) * np.sign(h)).T

    @staticmethod
    def compute_gradient2(X, y, w):
        N = len(y)
        P = X.shape[1]
        h = HingeLossFunction.compute_value(y, X.dot(w))
        G = np.zeros((N, P))
        for i in range(N):
            if h[i] > 0:
                G[i] = - y[i] * X[i]
            elif h[i] == 0:
                for j in range(len(X[i])):
                    inf = min(0, X[i][j])
                    sup = max(0, X[i][j])
                    G[i][j] = np.random.uniform(inf, sup)
                G[i] *= -y[i]
        G = np.sum(G, axis=0)

        return G


class EdgyHingeLossFunction(LossFunctionAbstract):
    @staticmethod
    def compute_value(X, y, w):
        return (1 - y * np.sign(X.dot(w))).clip(min=0)

    @staticmethod
    def compute_gradient(X, y, w):
        h = HingeLossFunction.compute_value(X, y, w)
        return (-y * X.T).T * np.sign(h)

    @staticmethod
    def compute_gradient_iteratively(X, y, w):
        N = len(y)
        P = X.shape[1]
        h = HingeLossFunction.compute_value(y, X.dot(w))
        G = np.zeros((N, P))
        for i in range(N):
            if h[i] > 0:
                G[i] = - y[i] * X[i]
            elif h[i] == 0:
                for j in range(len(X[i])):
                    inf = min(0, X[i][j])
                    sup = max(0, X[i][j])
                    G[i][j] = np.random.uniform(inf, sup)
                G[i] *= -y[i]
        G = np.sum(G, axis=0)

        return G


# no more used
class LinearYHatFunction:
    @staticmethod
    def compute_value(X, w):
        return X.dot(w)

    @staticmethod
    def compute_gradient(X, w):
        return X


#############
## METRICS ##
#############
class Metrics:
    __metaclass__ = abc.ABCMeta
    loss_func = None

    def compute_value(self, X, y, w):
        return np.sum(self.loss_func.compute_value(X, y, w) / len(y))

    def compute_gradient(self, X, y, w):
        return np.sum(self.loss_func.compute_gradient(X, y, w) / len(y), axis=0)


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
    ContinuousHingeLoss.id: ContinuousHingeLoss()
}
