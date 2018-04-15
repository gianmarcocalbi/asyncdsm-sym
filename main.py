import random, math, time
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize
from sklearn import linear_model
from src.model import Cluster
from src.graph_generator import GraphGenerator
from src import mltoolbox
import matplotlib.pyplot as plt

seed = 28041994
np.random.seed(seed)
random.seed(seed)


def main():
    # console.stdout.screen = stdscr
    # console.stdout.open()

    # __adjacency_matrix = GraphGenerator.generate_d_regular_graph_by_edges(5, ["i->i+1"])
    adjacency_matrix = GraphGenerator.generate_complete_graph(1)
    # __markov_matrix = normalize(__adjacency_matrix, axis=1, norm='l1')
    # X, y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)
    X, y = mltoolbox.SampleGenerator.sample_from_function(1000, 10, mltoolbox.linear_function, 1, error_std_dev=1,
                                                          error_coeff=0)
    # __X, __y = np.loadtxt("./dataset/largescale_challenge/alpha/alpha_train.dat"), np.loadtxt("./dataset/largescale_challenge/alpha/alpha_train.lab")
    X, y = np.array([np.arange(5)]).T, np.arange(5) * 2

    cluster = Cluster(adjacency_matrix)

    cluster.setup(
        X, y, mltoolbox.LinearYHatFunction,
        max_iter=math.inf,
        method="classic",
        batch_size=1,
        activation_func=None,
        loss=mltoolbox.SquaredLossFunction,
        penalty='l2',
        alpha=0.01,
        learning_rate="constant",
        metrics="all",
        shuffle=False,
        verbose=False
    )

    cluster.run()

    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    # plt.yscale('log')
    # plt.axis(ymax=0.50)
    # plt.annotate('Error {}'.format(__cluster.nodes[0].training_model.squared_loss_log[-1]),
    #   xy=(len(__cluster.nodes[0].training_model.squared_loss_log)/2, 5))
    plt.plot(list(range(0, len(cluster.nodes[0].training_model.squared_loss_log))),
             cluster.nodes[0].training_model.squared_loss_log)
    plt.show()

    # console.print("Score: {}".format(cluster.nodes[0].training_model.score()))

    # input("Press an key")

    # console.stdout.close()


def main1():
    # __X, __y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)
    __X, __y = mltoolbox.SampleGenerator.sample_from_function(1000, 100, mltoolbox.linear_function, 1,
                                                              sigma=False)
    cls = linear_model.SGDClassifier(loss="squared_loss", max_iter=100000)
    cls.fit(__X, __y)
    print(cls.score(__X, __y))


def main2():
    __X, __y = mltoolbox.SampleGenerator.sample_from_function(1000, 3, mltoolbox.linear_function, 10,
                                                              sigma=False)
    cls = linear_model.SGDRegressor(penalty='none', alpha=0.01, max_iter=1000, shuffle=False, learning_rate='constant')
    cls.fit(__X, __y)
    print(cls.score(__X, __y))
    print(cls.predict(np.array([2, 4, 8]).reshape(1, -1)))


switch = 0

if __name__ == "__main__":
    if switch == 0:
        main()
    elif switch == 1:
        main1()
    elif switch == 2:
        main2()
