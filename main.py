import random, math, time
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize
from src.graph_generator import GraphGenerator
from src import model, mltoolbox, console
from curses import wrapper
import matplotlib.pyplot as plt

seed = 28041994
np.random.seed(seed)
random.seed(seed)


def main(stdscr):
    console.stdout.screen = stdscr
    console.stdout.open()

    # __adjacency_matrix = GraphGenerator.generate_d_regular_graph_by_edges(5, ["i->i+1"])
    __adjacency_matrix = GraphGenerator.generate_complete_graph(1)
    __markov_matrix = normalize(__adjacency_matrix, axis=1, norm='l1')
    # __X, __y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)
    __X, __y = mltoolbox.SampleGenerator.sample_from_function(1000, 50, mltoolbox.linear_function, 10, biased=True)
    # __X, __y = np.loadtxt("./dataset/largescale_challenge/alpha/alpha_train.dat"), np.loadtxt("./dataset/largescale_challenge/alpha/alpha_train.lab")

    __setup = {
        "iteration_amount": 1000
    }

    __training_setup = {
        "X": __X,
        "y": __y,
        "learning_rate": 0.01,
        "activation_function": "identity",  # sigmoid, sign, tanh, identity, whatever other name will lead to identity
        "method": "stochastic",  # classic, stochastic, batch
        "batch_size": 10  # matters only for batch method
    }

    __cluster = model.Cluster(__adjacency_matrix, __training_setup, __setup)
    __cluster.run()

    console.stdout.close()

    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    # plt.yscale('log')
    # plt.axis(ymax=0.50)
    # plt.annotate('Error {}'.format(__cluster.nodes[0].training_model.loss_log[-1]),
    #   xy=(len(__cluster.nodes[0].training_model.loss_log)/2, 5))
    plt.plot(list(range(0, len(__cluster.nodes[0].training_model.loss_log))),
             __cluster.nodes[0].training_model.loss_log)
    plt.show()


if __name__ == "__main__":
    wrapper(main)
