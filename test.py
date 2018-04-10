import random, math
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize

from src.graph_generator import GraphGenerator
from src import model, mltoolbox

np.random.seed(2894)
random.seed(2894)

if __name__ == "__main__":
    # __adjacency_matrix = GraphGenerator.generate_d_regular_graph_by_edges(5, ["i->i+1"])
    __adjacency_matrix = GraphGenerator.generate_complete_graph(100)
    __markov_matrix = normalize(__adjacency_matrix, axis=1, norm='l1')
    # (__X, __y) = make_blobs(n_samples=10000, n_features=10, centers=2, cluster_std=2, random_state=20)
    (__X, __y) = mltoolbox.SampleGenerator.generate_linear_function_sample(10, 10, 9)

    np.savetxt("./out/X", __X)
    # np.savetxt("./out/y", __y)
