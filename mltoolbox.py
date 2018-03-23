# import the necessary packages
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_regression, make_blobs
import numpy as np
import random
import time

np.random.seed(2894)


class TrainingModel:
    def __init__(self, X, y, f, learning_rate):
        self.X = X
        self.y = y
        self.f = f
        self.learning_rate = learning_rate
        # bias inserted as w0
        self.X = np.c_[np.ones((X.shape[0])), X]
        self.W = np.random.uniform(size=(X.shape[1] + 1,))
        self.loss_log = []

    def gradient_descent_step(self):
        predictions = self.sigmoid(self.X.dot(self.W))
        linear_error = predictions - self.y
        loss = np.sum(linear_error ** 2)
        self.loss_log.append(loss)
        gradient = self.X.T.dot(linear_error) / self.X.shape[0]
        self.W -= self.learning_rate * gradient

    def stochastic_gradient_descent_step(self):
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
    (X, y) = make_blobs(n_samples=10, n_features=10, centers=2, cluster_std=2, random_state=20)
    lm = TrainingModel(X, y, lambda x: 2 * x, 0.005)
    for i in range(10):
        lm.gradient_descent_step()
        print(lm.loss_log[-1])