# import the necessary packages
import abc
import math
import warnings
import numpy as np
from src.mltoolbox import functions


class Task:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def step(self, *args):
        raise NotImplementedError('step method not implemented in Task child class')


class Trainer(Task):
    __metaclass__ = abc.ABCMeta

    def __init__(self, X, y, real_w, obj_function, starting_weights_domain, verbose):
        self.verbose = verbose
        self.obj_function = obj_function
        self.X = X
        self.y = y
        self.real_w = real_w
        self.real_y = self.obj_function.y_hat_func.compute_value(self.X, self.real_w)
        self.N = self.X.shape[0]
        self.iteration = 0

        if not (isinstance(starting_weights_domain, tuple) or isinstance(starting_weights_domain, list)):
            starting_weights_domain = [starting_weights_domain, starting_weights_domain]

        self.w = [
            np.random.uniform(
                starting_weights_domain[0],
                starting_weights_domain[1],
                size=(self.X.shape[1],)
            )
        ]
        # self.W = np.zeros(X.shape[1])

    @abc.abstractmethod
    def step(self, *args):
        raise NotImplementedError('step method not implemented in Trainer child class')

    def get_w(self):
        return np.copy(self.w[-1])

    def get_w_at_iteration(self, iteration):
        return np.copy(self.w[iteration])

    def set_w(self, new_w):
        self.w[-1] = new_w

    def set_w_at_iteration(self, new_w, iteration):
        self.w[iteration] = new_w


class GradientDescentTrainerAbstract(Trainer):
    __metaclass__ = abc.ABCMeta

    def __init__(self, X, y, real_w, obj_function, starting_weights_domain, alpha, learning_rate, metrics, real_metrics,
                 real_metrics_toggle, shuffle, verbose):
        super().__init__(X, y, real_w, obj_function, starting_weights_domain, verbose)

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        self.metrics = metrics
        self.real_metrics = real_metrics
        self.real_metrics_toggle = real_metrics_toggle

        self.logs = {
            "obj_function": [],
            "real_obj_function": [],
            "metrics": {},
            "real_metrics": {}
        }

        # instantiate logs' lists for both metrics and real_metrics
        for mk in self.metrics:
            self.logs["metrics"][mk] = []

        if self.real_metrics_toggle:
            for rmk in self.real_metrics:
                self.logs["real_metrics"][rmk] = []

        self.logs["obj_function"] = self.logs["metrics"][self.obj_function.id]

        if self.real_metrics_toggle:
            self.logs["real_obj_function"] = self.logs["real_metrics"][self.obj_function.id]

        self._compute_all_metrics()

    def _compute_all_metrics(self):
        for m in self.metrics:
            self._compute_metrics(m, real=False)
        for rm in self.real_metrics:
            self._compute_metrics(rm, real=True)

    def get_alpha(self):
        alpha = self.alpha
        if self.learning_rate == "root_decreasing" and self.iteration > 1:
            alpha /= math.sqrt(self.iteration)
        return alpha

    def get_metrics_value(self, m, real=False):
        if real:
            return self.logs["real_metrics"][m][-1]
        return self.logs["metrics"][m][-1]

    def get_metrics_value_at_iteration(self, m, it, real=False):
        if real:
            return self.logs["real_metrics"][m][it]
        return self.logs["metrics"][m][it]

    def _compute_metrics(self, m, real=False):
        if not real:
            y = self.y
            metrics = self.metrics
            metrics_log = self.logs["metrics"]
        else:
            y = self.real_y
            metrics = self.real_metrics
            metrics_log = self.logs["real_metrics"]

        if len(metrics_log[m]) == self.iteration:
            try:
                val = metrics[m].compute_value(self.X, y, self.get_w())
            except:
                raise
            metrics_log[m].append(val)
        elif len(metrics_log[m]) == self.iteration + 1:
            val = metrics_log[m][self.iteration]
            warnings.warn("Unexpected behaviour: metrics {} already computed in node".format(m))
        else:
            raise Exception("Unexpected {} metrics log size in node".format(m))
        return val

    @abc.abstractmethod
    def step(self, *args):
        raise NotImplementedError('step method not implemented in GradientDescentTrainerAbstract child class')


class LinearRegressionGradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, *args):
        super().__init__(*args)
        self.beta = None

    def step(self):
        if self.beta is None:
            self.beta = functions.estimate_linear_regression_beta(self.X, self.y)
        self.w.append(self.beta)
        self.iteration += 1
        self._compute_all_metrics()


class GradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, *args):
        super().__init__(*args)

    def step(self, avg_w):
        # update W following the steepest gradient descent
        gradient = self.obj_function.compute_gradient(self.X, self.y, self.get_w())
        self.w.append(avg_w - self.get_alpha() * gradient)
        self.iteration += 1
        self._compute_all_metrics()


class StochasticGradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, *args):
        super().__init__(*args)

    def step(self, avg_w):
        pick = np.random.randint(0, self.X.shape[0])
        X_pick = self.X[pick]
        y_pick = self.y[pick]
        gradient = self.obj_function.compute_gradient(X_pick, y_pick, self.get_w())
        self.w.append(avg_w - self.get_alpha() * gradient)
        self.iteration += 1
        self._compute_all_metrics()


class BatchGradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, batch_size, *args):
        super().__init__(*args)
        self.batch_size = batch_size
        if batch_size == 1:
            warnings.warn(
                "BatchGradientDescentTrainer started with batch_size = 1, it is preferable to use "
                "StochasticGradientDescentTrainer instead")

    def step(self, avg_w):
        # determine the mini batch upon which computing the SGD
        batch_indices = np.random.choice(self.X.shape[0], min(self.batch_size, self.X.shape[0]), replace=False)

        # extract subsamples
        X_batch = np.take(self.X, batch_indices, axis=0)
        y_batch = np.take(self.y, batch_indices, axis=0)
        gradient = self.obj_function.compute_gradient(X_batch, y_batch, self.get_w())
        self.w.append(avg_w - self.get_alpha() * gradient)
        self.iteration += 1
        self._compute_all_metrics()


class DualAveragingGradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, r, X, y, real_w, obj_function, starting_weights_domain, alpha, learning_rate,
                 real_metrics, metrics, shuffle, verbose):
        if learning_rate != "root_decreasing":
            warnings.warn(
                "Dual averaging method is running with wrong alpha (={}), it should bel root_decreasing".format(
                    learning_rate))
        super().__init__(X, y, real_w, obj_function, starting_weights_domain, alpha, learning_rate, metrics,
                         real_metrics, shuffle, verbose)
        self.z = [np.zeros(len(self.get_w()))]
        self.r = r

    def step(self, avg_z):
        gradient = self.obj_function.compute_gradient(self.X, self.y, self.get_w())
        z = avg_z + gradient
        self.z.append(z)
        phi = -self.get_alpha() * z
        phi_norm_2 = math.sqrt(np.inner(phi, phi))
        if phi_norm_2 > self.r:
            phi = (phi / phi_norm_2) * self.r
        self.w.append(phi)
        self.iteration += 1
        self._compute_all_metrics()

    def get_z(self):
        return np.copy(self.z[-1])

    def get_z_at_iteration(self, iteration):
        return np.copy(self.z[iteration])

    def set_z(self, new_z):
        self.z[-1] = new_z

    def set_z_at_iteration(self, new_z, iteration):
        self.z[iteration] = new_z
