# import the necessary packages
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import math

np.random.seed(2894)


class TrainingModel:
    def __init__(self, X, y, f, learning_rate, batch_size=1):
        self.X = X  # sample instances matrix
        self.y = y  # sample function's values array
        self.f = f  # network function (usually just like <X,y>)
        self.learning_rate = learning_rate  # learning rate alpha
        self.batch_size = batch_size

        # bias inserted as w0
        self.X = np.c_[np.ones((X.shape[0])), X]
        self.W = np.random.uniform(size=(X.shape[1] + 1,))
        self.loss_log = []

    def gradient_descent_step(self):
        """
        Gradient descent step function.
        :return: None
        """
        N = self.X.shape[0]

        # get the prediction values by exploiting the sigmoid function
        # predictions = self.sigmoid(self.X.dot(self.W))
        predictions = self.X.dot(self.W)

        # compute the linear error as the difference between predicted y and y
        linear_error = predictions - self.y

        # compute the error function (mean square error)
        mean_square_error = np.sum(linear_error ** 2) / (2 * N)
        self.loss_log.append(mean_square_error)

        # compute the gradient
        gradient = self.X.T.dot(linear_error) / N

        # update W following the steepest gradient descent
        self.W -= self.learning_rate * gradient

    def stochastic_gradient_descent_step(self):
        # determine the mini batch upon which compute the SGD
        batch_indices = np.random.choice(self.X.shape[0], min(self.batch_size, self.X.shape[0]), replace=False)
        M = batch_indices.shape[0]

        sub_X = np.take(self.X, batch_indices, axis=0)
        sub_y = np.take(self.y, batch_indices, axis=0)

        # get the prediction values by exploiting the sigmoid function
        predictions = sub_X.dot(self.W)

        # compute the linear error as the difference between predicted y and y
        linear_error = predictions - sub_y

        # compute the loss function loss(f(X), y)
        mean_square_error = np.sum(linear_error ** 2) / (2 * M)
        self.loss_log.append(mean_square_error)

        # compute the gradient
        gradient = sub_X.T.dot(linear_error) / M

        # update W following the steepest gradient descent
        self.W -= self.learning_rate * gradient

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))


class SampleGenerator:
    def __init__(self):
        # a lot of useful parameters
        pass

    @staticmethod
    def generate_linear_function_sample(n_samples, n_features, domain):
        X = []
        y = []
        w = []
        f = lambda _x, _w: _x.dot(_w)  # + np.random.rand()/domain

        for _ in range(n_features):
            w.append(np.random.uniform(-10, 10))

        for i in range(n_samples):
            x = np.random.uniform(-domain, domain, n_features)
            X.append(x)
            y.append(f(x, w))
        return np.array(X), np.array(y)


if __name__ == "__main__":
    """
    (__X, __y) = make_blobs(n_samples=10, n_features=10, centers=2, cluster_std=2, random_state=20)
    lm = TrainingModel(__X, __y, lambda x: 2 * x, 0.005)
    while True:
        lm.stochastic_gradient_descent_step()
        print(lm.loss_log[-1])
    """
    print(SampleGenerator.generate_linear_function_sample(10, 1, 10))
