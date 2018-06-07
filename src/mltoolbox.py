# import the necessary packages
import numpy as np
import abc, types, warnings


class Task:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError('step method not implemented in Task child class')


class Trainer(Task):
    __metaclass__ = abc.ABCMeta

    def __init__(self, X, y, y_hat, activation_func):

        # bias inserted as w0 = (1,...,1)
        self.X = np.c_[np.ones((X.shape[0])), X]
        self.y = y
        self.y_hat = y_hat
        self.N = self.X.shape[0]
        self.iteration = 0

        #self.W = np.zeros(X.shape[1] + 1)
        self.W = np.random.uniform(-2, 8, size=(X.shape[1] + 1,))  # todo: remove "+ 1"?
        self.W_log = [np.copy(self.W)]

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

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError('step method not implemented in Trainer child class')


class GradientDescentTrainerAbstract(Trainer):
    __metaclass__ = abc.ABCMeta

    def __init__(self, X, y, y_hat, activation_func, loss, penalty,
                 alpha, learning_rate, metrics, shuffle, verbose):
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
        super().__init__(X, y, y_hat, activation_func)
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
        predictions = self.activation_func(self.y_hat.f(self.X, self.W))
        linear_error = np.absolute(self.y - predictions)
        mean_absolute_error = np.sum(linear_error) / self.N
        if len(self.mean_absolute_error_log) == self.iteration:
            self.mean_absolute_error_log.append(mean_absolute_error)
        elif len(self.mean_absolute_error_log) == self.iteration + 1:
            self.mean_absolute_error_log[self.iteration] = mean_absolute_error
        else:
            raise Exception('Unexpected mean_absolute_error_log size')

        return mean_absolute_error

    def compute_mean_squared_error(self):
        predictions = self.activation_func(self.y_hat.f(self.X, self.W))
        linear_error = np.absolute(self.y - predictions)
        mean_squared_error = np.sum(np.power(linear_error, 2)) / self.N
        if len(self.mean_squared_error_log) == self.iteration:
            self.mean_squared_error_log.append(mean_squared_error)
        elif len(self.mean_squared_error_log) == self.iteration + 1:
            self.mean_squared_error_log[self.iteration] = mean_squared_error
        else:
            raise Exception('Unexpected mean_squared_error_log size')

        return mean_squared_error

    def compute_real_mean_squared_error(self):
        W = np.ones(len(self.W))
        real_values = self.activation_func(self.y_hat.f(self.X, W))
        predictions = self.activation_func(self.y_hat.f(self.X, self.W))
        linear_error = np.absolute(real_values - predictions)
        real_mean_squared_error = np.sum(np.power(linear_error, 2)) / self.N

        # real_mean_squared_error -= variance

        if len(self.real_mean_squared_error_log) == self.iteration:
            self.real_mean_squared_error_log.append(real_mean_squared_error)
        elif len(self.real_mean_squared_error_log) == self.iteration + 1:
            self.real_mean_squared_error_log[self.iteration] = real_mean_squared_error
        else:
            raise Exception('Unexpected real_mean_squared_error_log size')

        return real_mean_squared_error

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError('step method not implemented in GradientDescentTrainerAbstract child class')


class GradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, *args):
        super().__init__(*args)

    def step(self):
        # update W following the steepest gradient descent
        y_hat_f = self.y_hat.f(self.X, self.W)
        y_hat_f_gradient = self.y_hat.f_gradient(self.X, self.y)
        loss_f_gradient = self.loss.f_gradient(self.y, y_hat_f, y_hat_f_gradient)
        gradient = loss_f_gradient / self.N

        self.W -= self.alpha * gradient
        # self.W = estimate_beta(self.X, self.y)

        self.W_log.append(np.copy(self.W))
        self.iteration += 1
        self._compute_metrics()


class StochasticGradientDescentTrainer(GradientDescentTrainerAbstract):
    def __init__(self, *args):
        super().__init__(*args)

    def step(self):
        pick = np.random.randint(0, self.X.shape[0])
        X_pick = self.X[pick]
        y_pick = self.y[pick]
        y_hat_f = self.y_hat.f(X_pick, self.W)
        y_hat_f_gradient = self.y_hat.f_gradient(X_pick, y_pick)
        loss_f_gradient = self.loss.f_gradient(y_pick, y_hat_f, y_hat_f_gradient)
        gradient = loss_f_gradient
        self.W -= self.alpha * gradient
        self.W_log.append(np.copy(self.W))
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
        self.W -= self.alpha * gradient
        self.W_log.append(np.copy(self.W))
        self.iteration += 1
        self._compute_metrics()


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
        # the minus sign from the derivative of "- y_hat_f" is represented as follows:
        #   - (y - y_hat_f) = y_hat_f - y
        return y_hat_f_gradient.T.dot(y_hat_f - y)


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
        return X.dot(W)

    @staticmethod
    def f_gradient(X, W):
        return X


class ParaboloidYHatFunction(YHatFunctionAbstract):
    @staticmethod
    def f(X, W):
        return np.power(X, 2).dot(W)

    @staticmethod
    def f_gradient(X, W):
        return np.power(X, 2)


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


def sample_from_function(n_samples, n_features, func, domain_radius=0.5, domain_center=0.5,
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
    w = np.ones(n_features)
    K = np.random.uniform(domain_center - domain_radius, domain_center + domain_radius, n_features)

    for i in range(n_samples):
        x = np.zeros(n_features)
        for j in range(n_features):
            x[j] = np.random.uniform(-K[j] + domain_radius, K[j] + domain_radius)
        X.append(x)
        y[i] = func(x, w) + np.random.normal(error_mean, error_std_dev)

    return np.array(X), y


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


def estimate_beta(_X, _y):
    # _X = np.delete(_X, 1, axis=1)
    # return np.concatenate(([1], np.linalg.inv(_X.T.dot(_X)).dot(_X.T).dot(_y)))
    return np.linalg.inv(_X.T.dot(_X)).dot(_X.T).dot(_y)


def compute_mse(W, X, y, activation_func, y_hat_func):
    N = X.shape[0]
    predictions = activation_func(y_hat_func(X, W))
    linear_error = np.absolute(y - predictions)
    return np.sum(np.power(linear_error, 2)) / N


def compute_mae(W, X, y, activation_func, y_hat_func):
    N = X.shape[0]
    predictions = activation_func(y_hat_func(X, W))
    linear_error = np.absolute(y - predictions)
    return np.sum(linear_error) / N
