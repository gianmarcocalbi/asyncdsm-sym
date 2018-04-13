# import the necessary packages
import numpy as np
import abc, types


class TrainerlInterface:
    def __init__(self, X, y, activation_func=None):
        self.X = X
        self.y = y
        self.N = self.X.shape[0]

        # bias inserted as w0 = (1,...,1)
        self.X = np.c_[np.ones((X.shape[0])), X]

        self.W = np.random.uniform(size=(X.shape[1] + 1,)) #todo: remove "+ 1"?

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


class GradientDescentTrainerInterface(TrainerlInterface):
    def __init__(self, X, y, activation_func=None, loss="hinge", penalty='l2',
                 alpha=0.0001, shuffle=False, verbose=0, learning_rate="optimal"):
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
        super().__init__(X, y, activation_func=activation_func)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.loss = loss
        self.penalty = penalty
        self.verbose = verbose
        self.shuffle = shuffle


class GradientDescentTrainer(GradientDescentTrainerInterface):
    def __init__(self):
        super().__init__()


class StochasticGradientDescentTrainer(GradientDescentTrainerInterface):
    def __init__(self):
        super().__init__()


class BatchGradientDescentTrainer(GradientDescentTrainerInterface):
    def __init__(self):
        super().__init__()


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
        predictions = self.activation_function(np.power(self.X, 2).dot(self.W))

        # compute the linear error as the difference between predicted y and real y values
        linear_error = predictions - self.y

        # compute the loss function loss(f(X), y)
        mean_square_error = np.sum(np.apply_along_axis(lambda x: x * x, 0, linear_error)) / (2 * N)
        self.squared_loss_log.append(mean_square_error)
        self.score_log.append(1 - sum(np.apply_along_axis(abs, 0, linear_error)) / N)

        ## END: only for statistical purpose

        # compute the gradient
        batch_linear_error = self.activation_function(np.power(X, 2).dot(self.W)) - y
        # gradient = X.T.dot(batch_linear_error) / M

        gradient = np.power(X, 2).T.dot(batch_linear_error) / M

        # update W following the steepest gradient descent
        self.W -= self.learning_rate * gradient


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
