# import the necessary packages
import numpy as np
import abc, types, warnings


class Trainer:
    def __init__(self, X, y, y_hat, activation_func=None):
        self.X = X
        self.y = y
        self.y_hat = y_hat
        self.N = self.X.shape[0]
        self.iteration = 0

        # bias inserted as w0 = (1,...,1)
        self.X = np.c_[np.ones((X.shape[0])), X]

        self.W = np.random.uniform(size=(X.shape[1] + 1,))  # todo: remove "+ 1"?

        if not activation_func is types.FunctionType:
            if activation_func == "sigmoid":
                activation_func = sigmoid
            elif activation_func == "sign":
                activation_func = np.sign
            elif activation_func == "tanh":
                activation_func = np.tanh
            else:
                activation_func = lambda x: x

        self.activation_func = activation_func


class GradientDescentTrainerAbstract(Trainer):
    __metaclass__ = abc.ABCMeta

    def __init__(self, X, y, y_hat, activation_func=None, loss="hinge", penalty='l2',
                 alpha=0.0001, learning_rate="constant", metrics="all", shuffle=False, verbose=False):
        """GD Trainer Interface

        Parameters
        ----------

        C : float
            Maximum step size (regularization). Defaults to 1.0.

        fit_intercept : bool
            Whether the intercept should be estimated or not. If False, the
            data is assumed to be already centered. Defaults to True.

        max_iter : int, optional
            The maximum number of passes over the training data (aka epochs).
            It only impacts the behavior in the ``fit`` method, and not the
            `partial_fit`.
            Defaults to 5. Defaults to 1000 from 0.21, or if tol is not None.

            .. versionadded:: 0.19

        tol : float or None, optional
            The stopping criterion. If it is not None, the iterations will stop
            when (loss > previous_loss - tol). Defaults to None.
            Defaults to 1e-3 from 0.21.

            .. versionadded:: 0.19

        shuffle : bool, default=True
            Whether or not the training data should be shuffled after each epoch.

        verbose : integer, optional
            The verbosity level

        loss : string, optional
            The loss function to be used:
            The possible options are 'hinge', 'log', 'modified_huber',
            'squared_hinge', 'perceptron', or a regression loss: 'squared_loss',
            'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.

        epsilon : float
            If the difference between the current prediction and the correct label
            is below this threshold, the model is not updated.

        random_state : int, RandomState instance or None, optional, default=None
            The seed of the pseudo random number generator to use when shuffling
            the data.  If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random number
            generator; If None, the random number generator is the RandomState
            instance used by `np.random`.

        warm_start : bool, optional
            When set to True, reuse the solution of the previous call to fit as
            initialization, otherwise, just erase the previous solution.

        average : bool or int, optional
            When set to True, computes the averaged SGD weights and stores the
            result in the ``coef_`` attribute. If set to an int greater than 1,
            averaging will begin once the total number of samples seen reaches
            average. So average=10 will begin averaging after seeing 10 samples.

            .. versionadded:: 0.19
               parameter *average* to use weights averaging in SGD

        n_iter : int, optional
            The number of passes over the training data (aka epochs).
            Defaults to None. Deprecated, will be removed in 0.21.

            .. versionchanged:: 0.19
                Deprecated

        Attributes
        ----------
        coef_ : array, shape = [1, n_features] if n_classes == 2 else [n_classes,\
                n_features]
            Weights assigned to the features.

        intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
            Constants in decision function.

        n_iter_ : int
            The actual number of iterations to reach the stopping criterion.
        """
        super().__init__(X, y, y_hat, activation_func=activation_func)
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        self.verbose = verbose
        self.matrics = metrics
        self.y_hat = y_hat
        self.score_log = []
        self.mean_linear_error_log = []

        self._compute_metrics()

    def _compute_metrics(self):
        if self.metrics == "all":
            self.metrics = ['score', 'mean_linear_error']
        for metric in self.metrics:
            # todo: remove eval!!!
            eval("self.compute_" + metric + "()")

    def get_score(self):
        return self.score_log[-1]

    def get_mean_linear_error(self):
        return self.mean_linear_error_log[-1]

    def compute_score(self):
        if len(self.score_log) == self.iteration:
            self.score_log.append(0)
        elif len(self.score_log) == self.iteration + 1:
            self.score_log[self.iteration] = 0
        else:
            raise Exception('Unexpected mean_linear_error_log size')
        return 0

    def compute_mean_linear_error(self):
        predictions = self.activation_func(self.y_hat.f(self.X, self.W))
        linear_error = self.y - predictions
        mean_linear_error = linear_error / self.N
        if len(self.mean_linear_error_log) == self.iteration:
            self.mean_linear_error_log.append(mean_linear_error)
        elif len(self.mean_linear_error_log) == self.iteration + 1:
            self.mean_linear_error_log[self.iteration] = mean_linear_error
        else:
            raise Exception('Unexpected mean_linear_error_log size')

        return mean_linear_error

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError('step method not implemented in GradientDescentTrainerAbstract child class')


class GradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        print(self.compute_mean_linear_error())

        # update W following the steepest gradient descent
        self.W += self.alpha * np.sum(self.loss.f_gradient(self.y, self.y_hat.f(self.X, self.W),
                                                           self.y_hat.f_gradient(self.X, self.y))) / self.N


class StochasticGradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        pass


class BatchGradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, *args, batch_size=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        if batch_size == 1:
            warnings.warn(
                "BatchGradientDescentTrainer started with batch_size = 1, it is preferable to use StochasticGradientDescentTrainer instead")

    def step(self):
        pass


class LossFunctionAbstract:
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def f(y, y_hat_f):
        raise NotImplementedError('f method not implemented in LossFunctionAbstract --> child class')

    @staticmethod
    @abc.abstractmethod
    def f_gradient(y, y_hat_f, y_hat_f_gradient):
        raise NotImplementedError('f_gradient method not implemented in LossFunctionAbstract child class')


class SquaredLossFunction(LossFunctionAbstract):
    @staticmethod
    def f(y, y_hat_f):
        return pow(2, y - y_hat_f) / 2

    @staticmethod
    def f_gradient(y, y_hat_f, y_hat_f_gradient):
        return (y - y_hat_f).T.dot(y_hat_f_gradient)


class YHatFunctionAbstract:
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def f(X, W):
        raise NotImplementedError('f method not implemented in YHatFunctionAbstract child class')

    @staticmethod
    @abc.abstractmethod
    def f_gradient(X, W):
        raise NotImplementedError('f_gradient method not implemented in YHatFunctionAbstract child class')


class LinearYHatFunction(YHatFunctionAbstract):
    @staticmethod
    def f(X, W):
        return X.T.dot(W)

    @staticmethod
    def f_gradient(X, W):
        return X


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
            y.append(func(x, w) + np.random.choice([-1, 1]) * np.random.rand() * int(biased))
        return np.array(X), np.array(y)


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
