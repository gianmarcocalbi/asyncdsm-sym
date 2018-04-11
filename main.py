import random, math, time
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize
from src.graph_generator import GraphGenerator
from src import model, mltoolbox, console
from curses import wrapper

seed = 28041993
np.random.seed(seed)
random.seed(seed)

def main(stdscr):
    console.stdout.screen = stdscr
    console.stdout.open()

    # __adjacency_matrix = GraphGenerator.generate_d_regular_graph_by_edges(5, ["i->i+1"])
    __adjacency_matrix = GraphGenerator.generate_complete_graph(4)
    __markov_matrix = normalize(__adjacency_matrix, axis=1, norm='l1')
    # __X, __y = make_blobs(n_samples=10000, n_features=100, centers=2, cluster_std=2, random_state=20)
    __X, __y = mltoolbox.SampleGenerator.generate_linear_function_sample(10000, 100, 1)
    # __X, __y = np.loadtxt("./dataset/largescale_challenge/alpha/alpha_train.dat"), np.loadtxt("./dataset/largescale_challenge/alpha/alpha_train.lab")

    __setup = {
        "iteration_amount": math.inf
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

if __name__ == "__main__":
    wrapper(main)