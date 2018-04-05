# import the necessary packages
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_regression, make_blobs
import numpy as np
import random
import time

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
        # get the prediction values by exploiting the sigmoid function
        predictions = self.sigmoid(self.X.dot(self.W))

        # compute the linear error as the difference between predicted y and y
        linear_error = predictions - self.y

        # compute the loss function loss(f(X), y)
        loss = np.sum(linear_error ** 2)
        self.loss_log.append(loss)

        # compute the gradient
        gradient = self.X.T.dot(linear_error) / self.X.shape[0]

        # update W following the steepest gradient descent
        self.W -= self.learning_rate * gradient

    def stochastic_gradient_descent_step(self):
        # determine the mini batch upon which compute the SGD
        M = np.random.choice(self.X.shape[0], self.batch_size)

        sub_X = np.take(self.X, M, axis=0)
        sub_y = np.take(self.y, M, axis=0)

        # get the prediction values by exploiting the sigmoid function
        predictions = self.sigmoid(sub_X.dot(self.W))

        # compute the linear error as the difference between predicted y and y
        linear_error = predictions - sub_y

        # compute the loss function loss(f(X), y)
        loss = np.sum(linear_error ** 2)
        self.loss_log.append(loss)

        # compute the gradient
        gradient = sub_X.T.dot(linear_error) / sub_X.shape[0]

        # update W following the steepest gradient descent
        self.W -= self.learning_rate * gradient

        pass

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))


class SampleGenerator:
    def __init__(self):
        # a lot of useful parameters
        pass

    def generate_sample(self):
        return make_regression()


if __name__ == "__main__":
    (__X, __y) = make_blobs(n_samples=10, n_features=10, centers=2, cluster_std=2, random_state=20)
    lm = TrainingModel(__X, __y, lambda x: 2 * x, 0.005)
    while True:
        lm.stochastic_gradient_descent_step()
        print(lm.loss_log[-1])