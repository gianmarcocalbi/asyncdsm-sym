import random, math, time, os, plotter
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize
from sklearn import linear_model
from src.model import Cluster
from src.graph_generator import GraphGenerator
from src import mltoolbox
import matplotlib.pyplot as plt


def main0():
    # console.stdout.screen = stdscr
    # console.stdout.open()

    ### BEGIN SETUP ###
    n = 20
    seed = 2
    graphs = (
        "clique",
        "cycle",
        "expand",
        "diag",
    )
    write_to_file = True
    plot_after_write_to_file = True
    sub_folder = "test_B2/"
    enable_insta_plot = False
    plots = (
        "iter_time",
        "mse_iter",
        "real-mse_iter",
        "mse_time",
        "real-mse_time",
    )
    ### END SETUP ###

    np.random.seed(seed)
    random.seed(seed)

    ### BEGIN ADJACENCY MATRIX GEN ###
    adjmats = []

    if "clique" in graphs:
        adjmats.append(GraphGenerator.generate_complete_graph(n))
    if "cycle" in graphs:
        adjmats.append(GraphGenerator.generate_d_regular_graph_by_edges(n, ["i->i+1"]))
    if "expand" in graphs:
        adjmats.append(
            GraphGenerator.generate_d_regular_graph_by_edges(n, ["i->i+1", "i->i-1", "i->i+{}".format(int(n / 2))]))
    if "diag" in graphs:
        adjmats.append(np.diag(np.ones(n)))
    ### BEGIN ADJACENCY MATRIX GEN ###

    # markov_matrix = normalize(__adjacency_matrix, axis=1, norm='l1')

    ### BEGIN TRAINING SET GEN ###
    # X, y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)

    # """
    X, y = mltoolbox.sample_from_function(
        1000, 100, mltoolbox.LinearYHatFunction.f,
        domain_radius=10,
        domain_center=0,
        error_mean=0,
        error_std_dev=1,
        error_coeff=4
    )
    # """

    """
    X, y = mltoolbox.sample_from_function_old(
        10000, 100, mltoolbox.LinearYHatFunction.f,
        domain_radius=10,
        domain_center=0,
        subdomains_radius=2,
        error_mean=0,
        error_std_dev=1,
        error_coeff=1
    )
    """

    """
    X = np.loadtxt("./dataset/largescale_challenge/alpha/alpha_train.dat")
    y = np.loadtxt("./dataset/largescale_challenge/alpha/alpha_train.lab")
    """

    # X, y = np.array([np.arange(5)]).T, np.arange(5) * 2

    """
    X = np.array([[1, 12, 14, -2, -25, -27, -9, -10, 24, -17]]).T
    y = np.array([2, 24, 28, -4, -50, -54, -18, -20, 48, -34])
    """
    ### BEGIN TRAINING SET GEN ###

    ### BEGIN MAIN STUFFS ###
    for a in range(len(adjmats)):
        adjmat = adjmats[a]
        graph = graphs[a]

        np.random.seed(seed)
        random.seed(seed)

        cluster = Cluster(adjmat)

        cluster.setup(
            X, y, mltoolbox.LinearYHatFunction,
            max_iter=1000,
            method="stochastic",
            batch_size=20,
            activation_func=None,
            loss=mltoolbox.SquaredLossFunction,
            penalty='l2',
            epsilon=None,
            alpha=0.0001,
            learning_rate="constant",
            metrics="all",
            shuffle=True,
            verbose=False
        )

        cluster.run()

        if write_to_file:
            if not os.path.exists("test_log/{}".format(sub_folder)):
                os.makedirs("test_log/{}".format(sub_folder))
            np.savetxt(
                "test_log/{}{}_global_real_mean_squared_error_log".format(sub_folder, graph),
                cluster.global_real_mean_squared_error_log,
                delimiter=','
            )
            np.savetxt(
                "test_log/{}{}_global_mean_squared_error_log".format(sub_folder, graph),
                cluster.global_mean_squared_error_log,
                delimiter=','
            )
            np.savetxt(
                "test_log/{}{}_iterations_time_log".format(sub_folder, graph),
                cluster.iterations_time_log,
                delimiter=','
            )


        alpha = cluster.nodes[0].training_task.alpha
        n_iter = len(cluster.global_mean_squared_error_log)

        if "iter_time" in plots and enable_insta_plot:
            plt.title("Global iterations over cluster clock (α={})".format(alpha))
            plt.xlabel("Time (s)")
            plt.ylabel("Iteration")
            plt.plot(
                list(range(0, len(cluster.iterations_time_log))),
                cluster.iterations_time_log
            )
            plt.show()

        if "mse_iter" in plots and enable_insta_plot:
            plt.title("MSE over global iterations (α={})".format(alpha))
            plt.xlabel("Iteration")
            plt.ylabel("MSE")
            plt.ylim(ymax=50)
            plt.annotate('MSE = {}'.format(cluster.get_global_mean_squared_error()),
                         xy=(n_iter / 2, 20))
            plt.scatter(
                list(range(0, n_iter)),
                cluster.global_mean_squared_error_log,
                s=0.1
            )
            plt.show()

        if "real-mse_iter" in plots and enable_insta_plot:
            plt.title("Real MSE over global iterations (α={})".format(alpha))
            plt.xlabel("Iteration")
            plt.ylabel("Real MSE")
            plt.ylim(ymax=50)
            plt.annotate('MSE = {}'.format(cluster.get_global_real_mean_squared_error()),
                         xy=(n_iter / 2, 20))
            plt.scatter(
                list(range(0, n_iter)),
                cluster.global_real_mean_squared_error_log,
                s=0.1
            )
            plt.show()

        if "mse_time" in plots and enable_insta_plot:
            plt.title("MSE over time (α={})".format(alpha))
            plt.xlabel("Time (s)")
            plt.ylim(ymax=50)
            plt.ylabel("MSE")
            plt.scatter(
                cluster.iterations_time_log,
                cluster.global_mean_squared_error_log
            )
            plt.show()

        if "real-mse_time" in plots and enable_insta_plot:
            plt.title("Real MSE over time (α={})".format(alpha))
            plt.xlabel("Time (s)")
            plt.ylim(ymax=50)
            plt.ylabel("Real MSE")
            plt.plot(
                cluster.iterations_time_log,
                cluster.global_real_mean_squared_error_log
            )
            plt.show()

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

    if write_to_file and plot_after_write_to_file:
        plotter.main(sub_folder)

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
    eval("main{}()".format(switch))
