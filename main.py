import random, math, time, os, plotter, pickle
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

    descriptor = """>>> Test Descriptor File
Test name
=========
test name

Test Description
================
sample description

"""

    setup = dict()

    setup['seed'] = 2
    setup['n'] = 100
    setup['graphs'] = (
        "clique",
        "cycle",
        "expand",
        "diag",
    )

    # TRAINING SET SETUP
    setup['n_samples'] = 1000
    setup['n_features'] = 100
    setup['sample_function'] = mltoolbox.LinearYHatFunction.f
    setup['domain_radius'] = 10
    setup['domain_center'] = 0
    setup['error_mean'] = 0
    setup['error_std_dev'] = 1
    setup['error_coeff'] = 4

    # CLUSTER SETUP
    setup['yhat'] = mltoolbox.LinearYHatFunction
    setup['max_iter'] = 100
    setup['method'] = "stochastic"
    setup['batch_size'] = 20
    setup['activation_func'] = None
    setup['loss'] = mltoolbox.SquaredLossFunction
    setup['penalty'] = 'l2'
    setup['epsilon'] = None
    setup['alpha'] = alpha = 0.0001
    setup['learning_rate'] = "constant"
    setup['metrics'] = "all"
    setup['shuffle'] = True
    setup['verbose'] = False

    # OUTPUT SETUP
    test_log_sub_folder = "test_B3/"
    overwrite_if_already_exists = False
    write_to_file = True
    plot_from_file = True
    save_descriptor = True
    save_setup = True
    instant_plotting = False
    instant_plots = (
        "iter_time",
        "mse_iter",
        "real-mse_iter",
        "mse_time",
        "real-mse_time",
    )
    ### END SETUP ###

    np.random.seed(setup['seed'])
    random.seed(setup['seed'])

    if not write_to_file:
        test_log_sub_folder = "temp/{}/".format(int(time.time()))
    else:
        if test_log_sub_folder[-1] != "/":
            test_log_sub_folder += "/"

    if overwrite_if_already_exists:
        c = 0
        tmp_test_log_sub_folder = test_log_sub_folder
        while os.path.exists("test_log/{}".format(tmp_test_log_sub_folder)):
            tmp_test_log_sub_folder = test_log_sub_folder[0:-1] + str(c) + "/"
            c += 1
        test_log_sub_folder = tmp_test_log_sub_folder

    if not os.path.exists("test_log/{}".format(test_log_sub_folder)):
        os.makedirs("test_log/{}".format(test_log_sub_folder))

    ### BEGIN ADJACENCY MATRIX GEN ###
    adjmats = []

    if "clique" in setup['graphs']:
        adjmats.append(GraphGenerator.generate_complete_graph(setup['n']))
    if "cycle" in setup['graphs']:
        adjmats.append(GraphGenerator.generate_d_regular_graph_by_edges(setup['n'], ["i->i+1"]))
    if "expand" in setup['graphs']:
        adjmats.append(
            GraphGenerator.generate_d_regular_graph_by_edges(setup['n'], ["i->i+1", "i->i-1",
                                                                          "i->i+{}".format(int(setup['n'] / 2))]))
    if "diag" in setup['graphs']:
        adjmats.append(np.diag(np.ones(setup['n'])))
    ### BEGIN ADJACENCY MATRIX GEN ###

    # markov_matrix = normalize(__adjacency_matrix, axis=1, norm='l1')

    ### BEGIN TRAINING SET GEN ###
    # X, y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)

    # """
    X, y = mltoolbox.sample_from_function(
        setup['n_samples'], setup['n_features'], setup['sample_function'],
        domain_radius=setup['domain_radius'],
        domain_center=setup['domain_center'],
        error_mean=setup['error_mean'],
        error_std_dev=setup['error_std_dev'],
        error_coeff=setup['error_coeff']
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
    descriptor += """

### BEGIN SETUP ###
n = {n}
seed = {seed}
graphs = {graphs}

# TRAINING SET SETUP
n_samples = {n_samples}
n_features = {n_features}
yhat = {yhat}
domain_radius = {domain_radius}
domain_center = {domain_center}
error_mean = {error_mean}
error_std_dev = {error_std_dev}
error_coeff = {error_coeff}

# CLUSTER SETUP
sample_function = {sample_function}
max_iter = {max_iter}
method = {method}
batch_size = {batch_size}
activation_func = {activation_func}
loss = {loss}
penalty = {penalty}
epsilon = {epsilon}
alpha = {alpha}
learning_rate = {learning_rate}
metrics = {metrics}
shuffle = {shuffle}
verbose = {verbose}
""".format(**setup)

    if save_descriptor:
        with open("test_log/{}descriptor.txt".format(test_log_sub_folder), "w") as f:
            f.write(descriptor)

    if save_setup:
        with open("test_log/{}setup.pkl".format(test_log_sub_folder), "wb") as f:
            pickle.dump(setup, f, pickle.HIGHEST_PROTOCOL)

    for a in range(len(adjmats)):
        adjmat = adjmats[a]
        graph = setup['graphs'][a]

        np.random.seed(setup['seed'])
        random.seed(setup['seed'])

        cluster = Cluster(adjmat)

        cluster.setup(
            X, y, setup['yhat'],
            max_iter=setup['max_iter'],
            method=setup['method'],
            batch_size=setup['batch_size'],
            activation_func=setup['activation_func'],
            loss=setup['loss'],
            penalty=setup['penalty'],
            epsilon=setup['epsilon'],
            alpha=setup['alpha'],
            learning_rate=setup['learning_rate'],
            metrics=setup['metrics'],
            shuffle=setup['shuffle'],
            verbose=setup['verbose']
        )

        cluster.run()

        np.savetxt(
            "test_log/{}{}_global_real_mean_squared_error_log".format(test_log_sub_folder, graph),
            cluster.global_real_mean_squared_error_log,
            delimiter=','
        )
        np.savetxt(
            "test_log/{}{}_global_mean_squared_error_log".format(test_log_sub_folder, graph),
            cluster.global_mean_squared_error_log,
            delimiter=','
        )
        np.savetxt(
            "test_log/{}{}_iterations_time_log".format(test_log_sub_folder, graph),
            cluster.iterations_time_log,
            delimiter=','
        )

        n_iter = len(cluster.global_mean_squared_error_log)

        if "iter_time" in instant_plots and instant_plotting:
            plt.title("Global iterations over cluster clock (α={})".format(alpha))
            plt.xlabel("Time (s)")
            plt.ylabel("Iteration")
            plt.plot(
                list(range(0, len(cluster.iterations_time_log))),
                cluster.iterations_time_log
            )
            plt.show()

        if "mse_iter" in instant_plots and instant_plotting:
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

        if "real-mse_iter" in instant_plots and instant_plotting:
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

        if "mse_time" in instant_plots and instant_plotting:
            plt.title("MSE over time (α={})".format(alpha))
            plt.xlabel("Time (s)")
            plt.ylim(ymax=50)
            plt.ylabel("MSE")
            plt.scatter(
                cluster.iterations_time_log,
                cluster.global_mean_squared_error_log
            )
            plt.show()

        if "real-mse_time" in instant_plots and instant_plotting:
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

    if plot_from_file:
        plotter.plot_from_files(test_log_sub_folder)

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
