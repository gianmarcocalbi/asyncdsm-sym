import random, math, time
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize
from sklearn import linear_model
from src.model import Cluster
from src.graph_generator import GraphGenerator
from src import mltoolbox
import matplotlib.pyplot as plt

seed = 2
np.random.seed(seed)
random.seed(seed)


def main():
    # console.stdout.screen = stdscr
    # console.stdout.open()

    # adjacency_matrix = GraphGenerator.generate_complete_graph(20)
    # adjacency_matrix = GraphGenerator.generate_d_regular_graph_by_edges(20, ["i->i+1"])
    adjacency_matrix = GraphGenerator.generate_d_regular_graph_by_edges(20, ["i->i+1", "i->i-1", "i->i+10"])
    # adjacency_matrix = np.diag(np.ones(20))

    # markov_matrix = normalize(__adjacency_matrix, axis=1, norm='l1')

    # X, y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)

    # """
    X, y = mltoolbox.sample_from_function(
        10000, 100, mltoolbox.LinearYHatFunction.f,
        domain_radius=10,
        domain_center=0,
        subdomains_radius=2,
        error_mean=0,
        error_std_dev=1,
        error_coeff=1
    )
    # """

    """
    X = np.loadtxt("./dataset/largescale_challenge/alpha/alpha_train.dat")
    y = np.loadtxt("./dataset/largescale_challenge/alpha/alpha_train.lab")
    """

    # X, y = np.array([np.arange(5)]).T, np.arange(5) * 2

    """
    X = np.array([[1, 12, 14, -2, -25, -27, -9, -10, 24, -17]]).T
    y = np.array([2, 24, 28, -4, -50, -54, -18, -20, 48, -34])
    """

    cluster = Cluster(adjacency_matrix)

    cluster.setup(
        X, y, mltoolbox.LinearYHatFunction,
        max_iter=4000,
        method="stochastic",
        batch_size=20,
        activation_func=None,
        loss=mltoolbox.SquaredLossFunction,
        penalty='l2',
        epsilon = 0.01,
        alpha=0.0005,
        learning_rate="constant",
        metrics="all",
        shuffle=True,
        verbose=False
    )

    cluster.run()

    #np.savetxt("out/clique_global_mean_squared_error_log", cluster.global_mean_squared_error_log, delimiter=',')
    #np.savetxt("out/clique_iterations_time_log", cluster.iterations_time_log, delimiter=',')

    #np.savetxt("out/cycle_global_mean_squared_error_log", cluster.global_mean_squared_error_log, delimiter=',')
    #np.savetxt("out/cycle_iterations_time_log", cluster.iterations_time_log, delimiter=',')

    np.savetxt("out/expander_global_mean_squared_error_log", cluster.global_mean_squared_error_log, delimiter=',')
    np.savetxt("out/expander_iterations_time_log", cluster.iterations_time_log, delimiter=',')

    alpha = cluster.nodes[0].training_task.alpha

    """
    n_iter = len(cluster.global_mean_squared_error_log)
    plt.title("MSE over global iterations (α={})".format(alpha))
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.ylim(ymax=50)
    plt.annotate('MSE = {}'.format(cluster.get_global_mean_squared_error()),
                 xy=(n_iter / 2, 5))
    plt.plot(
        list(range(0, n_iter)),
        cluster.global_mean_squared_error_log
    )
    plt.show()
    """

    """
    plt.title("Global iterations over cluster clock (α={})".format(alpha))
    plt.xlabel("Time (s)")
    plt.ylabel("Iteration")
    plt.ylim(ymax=50)
    plt.plot(
        list(range(0, len(cluster.iterations_time_log))),
        cluster.iterations_time_log
    )
    plt.show()
    """

    """
    plt.title("Nodes iterations over clock (α={})".format(alpha))
    plt.xlabel("Time (s)")
    plt.ylabel("Iteration")
    for node in cluster.nodes:
        plt.plot(
            list(range(0, len(node.log))),
            node.log
        )
    plt.show()
    """

    """
    plt.title("MSE over time (α={})".format(alpha))
    plt.xlabel("Time (s)")
    plt.ylabel("MSE")
    plt.ylim(ymax=50)
    plt.plot(
        cluster.iterations_time_log,
        cluster.global_mean_squared_error_log
    )
    plt.show()
    """

    # console.print("Score: {}".format(cluster.nodes[0].training_model.score()))

    # input("Press an key")

    # console.stdout.close()


def main1():
    # __X, __y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)
    X, y = mltoolbox.sample_from_function(100000, 10, mltoolbox.linear_function, 1, error_std_dev=1,
                                          error_coeff=0)
    cls = linear_model.SGDClassifier(loss="squared_loss", max_iter=100000)
    cls.fit(X, y)
    print(cls.score(X, y))


def main2():
    X, y = mltoolbox.sample_from_function(1000, 10, mltoolbox.linear_function, 1, error_std_dev=1,
                                          error_coeff=0)
    cls = linear_model.SGDRegressor(penalty='none', alpha=0.01, max_iter=1000, shuffle=False, learning_rate='constant')
    cls.fit(X, y)
    print(cls.score(X, y))
    print(cls.predict(np.array([2, 4, 8]).reshape(1, -1)))


switch = 0

if __name__ == "__main__":
    if switch == 0:
        main()
    elif switch == 1:
        main1()
    elif switch == 2:
        main2()
