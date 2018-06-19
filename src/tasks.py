# import the necessary packages
import numpy as np
import abc, types, warnings, math
from src import mltoolbox


class Task:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def step(self, *args):
        raise NotImplementedError('step method not implemented in Task child class')


class Trainer(Task):
    __metaclass__ = abc.ABCMeta

    def __init__(self, X, y, real_w, y_hat, starting_weights_domain, activation_func=None):

        self.X = X
        self.y = y
        self.real_w = real_w
        self.y_hat = y_hat
        self.N = self.X.shape[0]
        self.iteration = 0

        # self.W = np.zeros(X.shape[1])
        self.w = [
            np.random.uniform(
                starting_weights_domain[0],
                starting_weights_domain[1],
                size=(self.X.shape[1],)
            )
        ]

        if not activation_func is types.FunctionType:
            if activation_func == "sigmoid":
                activation_func = mltoolbox.sigmoid
            elif activation_func == "sign":
                activation_func = np.sign
            elif activation_func == "tanh":
                activation_func = np.tanh
            else:
                activation_func = lambda x: x

        self.activation_func = activation_func

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

    def __init__(self, X, y, real_w, y_hat, starting_weights_domain, activation_func, loss, penalty,
                 alpha, learning_rate, metrics, shuffle, verbose):
        super().__init__(X, y, real_w, y_hat, starting_weights_domain, activation_func)
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        self.verbose = verbose
        self.metrics = metrics
        self.y_hat = y_hat
        self.score_log = []
        self.mean_absolute_error_log = []
        self.mean_squared_error_log = []
        self.real_mean_squared_error_log = []
        self.available_metrics = [
            'score',
            'mean_absolute_error',
            'mean_squared_error',
            "real_mean_squared_error"
        ]
        self._compute_metrics()

    def _compute_metrics(self):
        if self.metrics == "all":
            self.metrics = self.available_metrics
        elif isinstance(self.metrics, str):
            self.metrics = [self.metrics]

        for metric in self.metrics:
            # todo: remove eval!!!
            eval("self.compute_" + metric + "()")

    def get_alpha(self):
        alpha = self.alpha
        if self.learning_rate == "root_decreasing" and self.iteration > 1:
            alpha /= math.sqrt(self.iteration)
        return alpha

    def get_score(self):
        return self.score_log[-1]

    def get_mean_absolute_error(self):
        return self.mean_absolute_error_log[-1]

    def get_mean_squared_error(self):
        return self.mean_squared_error_log[-1]

    def get_real_mean_squared_error(self):
        return self.real_mean_squared_error_log[-1]

    def compute_score(self):
        # todo: compute score
        if len(self.score_log) == self.iteration:
            self.score_log.append(0)
        elif len(self.score_log) == self.iteration + 1:
            self.score_log[self.iteration] = 0
        else:
            raise Exception('Unexpected mean_absolute_error_log size')
        return 0

    def compute_mean_absolute_error(self):
        mae = mltoolbox.compute_mae(self.get_w(), self.X, self.y, self.activation_func, self.y_hat.f)
        if len(self.mean_absolute_error_log) == self.iteration:
            self.mean_absolute_error_log.append(mae)
        elif len(self.mean_absolute_error_log) == self.iteration + 1:
            self.mean_absolute_error_log[self.iteration] = mae
        else:
            raise Exception('Unexpected mean_absolute_error_log size')

        return mae

    def compute_mean_squared_error(self):
        mse = mltoolbox.compute_mse(
            self.get_w(),
            self.X,
            self.y,
            self.activation_func,
            self.y_hat.f
        )
        if len(self.mean_squared_error_log) == self.iteration:
            self.mean_squared_error_log.append(mse)
        elif len(self.mean_squared_error_log) == self.iteration + 1:
            self.mean_squared_error_log[self.iteration] = mse
        else:
            raise Exception('Unexpected mean_squared_error_log size')

        return mse

    def compute_real_mean_squared_error(self):
        real_values = self.activation_func(self.y_hat.f(self.X, self.real_w))
        rmse = mltoolbox.compute_mse(
            self.get_w(),
            self.X,
            real_values,
            self.activation_func,
            self.y_hat.f
        )
        # real_mean_squared_error -= variance

        if len(self.real_mean_squared_error_log) == self.iteration:
            self.real_mean_squared_error_log.append(rmse)
        elif len(self.real_mean_squared_error_log) == self.iteration + 1:
            self.real_mean_squared_error_log[self.iteration] = rmse
        else:
            raise Exception('Unexpected real_mean_squared_error_log size')

        return rmse

    @abc.abstractmethod
    def step(self, *args):
        raise NotImplementedError('step method not implemented in GradientDescentTrainerAbstract child class')


class LinearRegressionGradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, *args):
        super().__init__(*args)
        self.beta = None

    def step(self):
        if self.beta is None:
            self.beta = mltoolbox.estimate_linear_regression_beta(self.X, self.y)

        self.w.append(self.beta)

        self.iteration += 1
        self._compute_metrics()


class GradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, *args):
        super().__init__(*args)

    def step(self):
        # update W following the steepest gradient descent
        y_hat_f = self.y_hat.f(self.X, self.get_w())
        y_hat_f_gradient = self.y_hat.f_gradient(self.X, self.y)
        loss_f_gradient = self.loss.f_gradient(self.y, y_hat_f, y_hat_f_gradient)
        gradient = loss_f_gradient / self.N

        self.w.append(self.get_w() - self.get_alpha() * gradient)
        # self.W = estimate_beta(self.X, self.y)

        self.iteration += 1
        self._compute_metrics()


class StochasticGradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, *args):
        super().__init__(*args)

    def step(self):
        pick = np.random.randint(0, self.X.shape[0])
        X_pick = self.X[pick]
        y_pick = self.y[pick]
        y_hat_f = self.y_hat.f(X_pick, self.get_w())
        y_hat_f_gradient = self.y_hat.f_gradient(X_pick, y_pick)
        loss_f_gradient = self.loss.f_gradient(y_pick, y_hat_f, y_hat_f_gradient)
        gradient = loss_f_gradient

        self.w.append(self.get_w() - self.get_alpha() * gradient)
        self.iteration += 1
        self._compute_metrics()


class BatchGradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, batch_size, *args):
        super().__init__(*args)
        self.batch_size = batch_size
        if batch_size == 1:
            warnings.warn(
                "BatchGradientDescentTrainer started with batch_size = 1, it is preferable to use StochasticGradientDescentTrainer instead")

    def step(self):
        # determine the mini batch upon which computing the SGD
        batch_indices = np.random.choice(self.X.shape[0], min(self.batch_size, self.X.shape[0]), replace=False)

        # extract subsamples
        X_batch = np.take(self.X, batch_indices, axis=0)
        y_batch = np.take(self.y, batch_indices, axis=0)

        y_hat_f = self.y_hat.f(X_batch, self.W)
        y_hat_f_gradient = self.y_hat.f_gradient(X_batch, y_batch)
        loss_f_gradient = self.loss.f_gradient(y_batch, y_hat_f, y_hat_f_gradient)
        gradient = loss_f_gradient / self.batch_size
        self.w.append(self.get_w() - self.get_alpha() * gradient)
        self.iteration += 1
        self._compute_metrics()


class DualAveragingGradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, r, X, y, real_w, y_hat, starting_weights_domain, activation_func, loss, penalty,
                 alpha, learning_rate, metrics, shuffle, verbose):
        if learning_rate != "root_decreasing":
            warnings.warn(
                "Dual averaging method is running with wrong alpha (={}), it should bel root_decreasing".format(
                    learning_rate))
        super().__init__(X, y, real_w, y_hat, starting_weights_domain, activation_func, loss, penalty,
                         alpha, learning_rate, metrics, shuffle, verbose)
        self.z = [np.zeros(len(self.get_w()))]
        self.r = r

    def step(self, avg_z):
        y_hat_f = self.y_hat.f(self.X, self.get_w())
        y_hat_f_gradient = self.y_hat.f_gradient(self.X, self.y)
        loss_f_gradient = self.loss.f_gradient(self.y, y_hat_f, y_hat_f_gradient)
        gradient = loss_f_gradient / self.N
        z = avg_z + gradient
        self.z.append(z)
        phi = -self.get_alpha() * z
        phi_norm_2 = math.sqrt(np.inner(phi, phi))
        if phi_norm_2 > self.r:
            input("X")
            phi = (phi / phi_norm_2) * self.r

        self.w.append(phi)

        self.iteration += 1
        self._compute_metrics()

    def get_z(self):
        return np.copy(self.z[-1])

    def get_z_at_iteration(self, iteration):
        return np.copy(self.z[iteration])

    def set_z(self, new_z):
        self.z[-1] = new_z

    def set_z_at_iteration(self, new_z, iteration):
        self.z[iteration] = new_z

