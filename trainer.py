# import the necessary packages
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np


class TrainingModel:

    def __init__(self, X, y, f, learning_rate):
        self.X = X
        self.y = y
        self.f = f
        self.learning_rate = learning_rate
        # insert bias as w0
        X = np.c_[np.ones((X.shape[0])), X]
        self.W = np.random.uniform(size=(X.shape[1],))
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
