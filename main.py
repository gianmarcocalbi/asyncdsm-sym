import random, math, time, os, plotter, pickle, shutil, datetime, pprint
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize
from sklearn import linear_model
from src.model import Cluster
from src import mltoolbox, graph_generator
import matplotlib.pyplot as plt

# degree = 0
DIAGONAL = lambda n: np.diag(np.ones(n))

# degree = 1
CYCLE = lambda n: graph_generator.generate_d_regular_graph_by_edges(n, ["i->i+1"])

# degree = 2
CYCLE_B = lambda n: graph_generator.generate_d_regular_graph_by_edges(n, ["i->i+1", "i->i-1"])
DIAM_EXP = lambda n: graph_generator.generate_d_regular_graph_by_edges(
    n,
    ["i->i+1", "i->i+{}".format(int(n / 2))])
ROOT_EXP = lambda n: graph_generator.generate_d_regular_graph_by_edges(
    n,
    ["i->i+1", "i->i+{}".format(int(math.sqrt(n)))])

# degree = 3
DIAM_EXP_B = lambda n: graph_generator.generate_d_regular_graph_by_edges(
    n,
    ["i->i+1", "i->i-1", "i->i+{}".format(int(n / 2))])

# degree = n
STAR = lambda n: graph_generator.generate_graph_by_edges(n, ["i->0", "0->i"])
CLIQUE = lambda n: graph_generator.generate_complete_graph(n)

# regular
REGULAR = lambda n, k: graph_generator.generate_d_regular_graph_by_degree(n, k)


def main0():
    # console.stdout.screen = stdscr
    # console.stdout.open()

    ### BEGIN SETUP ###

    begin_time = time.time()
    # descriptor text placed at the beginning of _descriptor.txt file within the test folder
    descriptor = """>>> Test Descriptor File
Title: test
Date: {}
Summary:

""".format(str(datetime.datetime.now()))

    setup_from_file = False
    setup_folder_path = plotter.get_last_temp_test_path()
    setup_file_path = os.path.join(setup_folder_path, ".setup.pkl")

    setup = dict()

    setup['seed'] = int(time.time())
    setup['n'] = 100

    setup['graphs'] = {
        "0_diagonal": DIAGONAL(setup['n']),
        #"1_cycle": CYCLE(setup['n']),  # degree = 1
        #"2_diam-expander": DIAM_EXP(setup['n']),  # degree = 2
        # "2_root-expander": ROOT_EXP(setup['n']),  # degree = 2
        # "3_regular": REGULAR(setup['n'], 3),  # degree = 3
        # "4_regular": REGULAR(setup['n'], 4),  # degree = 4
        # "8_regular": REGULAR(setup['n'], 8),  # degree = 8
        # "20_regular": REGULAR(setup['n'], 20),  # degree = 20
        # "50_regular": REGULAR(setup['n'], 50),  # degree = 50
        #"n-1_clique": CLIQUE(setup['n']),  # degree = n
        # "n-1_star": STAR(setup['n']),
    }

    # TRAINING SET SETUP
    setup['n_samples'] = 10000
    setup['n_features'] = 100
    setup['domain_radius'] = 5
    setup['domain_center'] = 0
    setup['error_mean'] = 0
    setup['error_std_dev'] = 2
    setup['sample_function'] = mltoolbox.LinearYHatFunction.f

    # CLUSTER SETUP
    setup['max_iter'] = None
    setup['max_time'] = 10000  # seconds
    setup['yhat'] = mltoolbox.LinearYHatFunction
    setup['method'] = "classic"
    setup['batch_size'] = 20
    setup['activation_func'] = None
    setup['loss'] = mltoolbox.SquaredLossFunction
    setup['penalty'] = 'l2'
    setup['epsilon'] = None
    setup['alpha'] = alpha = 1e-04
    setup['learning_rate'] = "constant"
    setup['metrics'] = "all"
    setup['metrics_type'] = 0
    setup['shuffle'] = True
    setup['verbose'] = False
    setup['time_distr_func'] = random.expovariate
    setup['time_distr_rate'] = 0

    if setup_from_file:
        with open(setup_file_path, 'rb') as setup_file:
            setup = pickle.load(setup_file)

    # OUTPUT SETUP
    save_test_to_file = False  # write output files to "test_log/{test_log_sub_folder}/" folder
    test_root = "test_log"  # don't touch this
    test_subfolder = "test_004_1e-4alphaALTmetrics10ksamples2mtimeDiagClique_classic"  # test folder inside test_log/
    temp_test_subfolder = datetime.datetime.now().strftime('%y-%m-%d_%H:%M:%S.%f')
    overwrite_if_already_exists = False  # overwrite the folder if it already exists or create a different one otherwise
    delete_folder_on_errors = True
    plot_from_file = True  # run plotter.py upon finishing
    save_plot_to_file = False
    save_descriptor = True  # create _descriptor.txt file
    save_setup = True  # save setup object dump in order to restore it for run the same simulation
    instant_plotting = False  # instantly plot single simulations results
    instant_plots = (
        "iter_time",
        "mse_iter",
        "real-mse_iter",
        "mse_time",
        "real-mse_ time",
    )
    ### END SETUP ###

    np.random.seed(setup['seed'])
    random.seed(setup['seed'])

    if not save_test_to_file:
        # if you don't want to store the file permanently they are however placed inside temp folder
        # in order to use them for a short and limited period of time (temp folder may be deleted manually)
        test_subfolder = os.path.join("temp", temp_test_subfolder)
        overwrite_if_already_exists = False

    test_path = os.path.normpath(os.path.join(test_root, test_subfolder))

    if not overwrite_if_already_exists:
        # determine a name for the new folder such that it doesn't coincide with any other folder
        c = 0
        tmp_test_path = test_path
        while os.path.exists(tmp_test_path):
            tmp_test_path = test_path + ".conflict." + str(c)
            c += 1
        test_path = tmp_test_path

    test_path = os.path.normpath(test_path)

    # create dir
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # define function to delete test folder (in case of errors)
    def delete_test_dir():
        if delete_folder_on_errors:
            shutil.rmtree(test_path)

    ### BEGIN ADJACENCY MATRIX GEN ###
    """
    adjmats = {}
    for g in setup['graphs']:
        if not isinstance(g, str):
            if isinstance(g, np.ndarray):
                pass
            else:
                delete_test_dir()
                raise Exception("Unknown dependency graph {} of type {}".format(str(g), str(type(g))))
        else:
            if g == "clique":
                adjmats[g] = GraphGenerator.generate_complete_graph(setup['n'])
            if g == "cycle":
                adjmats[g] = GraphGenerator.generate_d_regular_graph_by_edges(setup['n'], ["i->i+1"])
            if g == "expand":
                adjmats[g] = GraphGenerator.generate_d_regular_graph_by_edges(
                    setup['n'],
                    ["i->i+1", "i->i-1", "i->i+{}".format(int(setup['n'] / 2))]
                )
            if g == "diag":
                adjmats[g] = np.diag(np.ones(setup['n']))
    """

    ### END ADJACENCY MATRIX GEN ###

    # markov_matrix = normalize(__adjacency_matrix, axis=1, norm='l1')

    ### BEGIN TRAINING SET GEN ###
    # X, y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)

    # """
    X, y = mltoolbox.sample_from_function(
        setup['n_samples'], setup['n_features'], setup['sample_function'],
        domain_radius=setup['domain_radius'],
        domain_center=setup['domain_center'],
        error_mean=setup['error_mean'],
        error_std_dev=setup['error_std_dev']
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

    # save setup object dump
    if save_setup:
        with open(os.path.join(test_path, '.setup.pkl'), "wb") as f:
            pickle.dump(setup, f, pickle.HIGHEST_PROTOCOL)

    setup['string_graphs'] = pprint.PrettyPrinter(indent=4).pformat(setup['graphs']).replace('array([', 'np.array([')

    # Fill descriptor with setup dictionary
    descriptor += """
### BEGIN SETUP ###
n = {n}
seed = {seed}
graphs = {string_graphs}

# TRAINING SET SETUP
n_samples = {n_samples}
n_features = {n_features}
sample_function = {sample_function}
domain_radius = {domain_radius}
domain_center = {domain_center}
error_mean = {error_mean}
error_std_dev = {error_std_dev}

# CLUSTER SETUP
max_iter = {max_iter}
max_time = {max_time}
yhat = {yhat}
method = {method}
batch_size = {batch_size}
activation_func = {activation_func}
loss = {loss}
penalty = {penalty}
epsilon = {epsilon}
alpha = {alpha}
learning_rate = {learning_rate}
metrics = {metrics}
metrics_type = {metrics_type}
shuffle = {shuffle}
verbose = {verbose}
time_distr_func = {time_distr_func}
time_distr_rate = {time_distr_rate}
""".format(**setup)

    # save descriptor file
    if save_descriptor:
        with open(os.path.join(test_path, '.descriptor.txt'), "w") as f:
            f.write(descriptor)

    # simulation for each adjacency matrix in setup['graphs'] dict
    for graph, adjmat in setup['graphs'].items():
        # set the seed again (each simulation must perform on the same cluster setup)
        np.random.seed(setup['seed'])
        random.seed(setup['seed'])

        cluster = Cluster(adjmat, graph_name=graph)

        cluster.setup(
            X, y, setup['yhat'],
            max_iter=setup['max_iter'],
            max_time=setup['max_time'],
            method=setup['method'],
            batch_size=setup['batch_size'],
            activation_func=setup['activation_func'],
            loss=setup['loss'],
            penalty=setup['penalty'],
            epsilon=setup['epsilon'],
            alpha=setup['alpha'],
            learning_rate=setup['learning_rate'],
            metrics=setup['metrics'],
            metrics_type=setup['metrics_type'],
            shuffle=setup['shuffle'],
            verbose=setup['verbose']
        )

        try:
            cluster.run()
        except:
            # if the cluster throws an exception then delete the folder created to host its output files
            # the most common exception in cluster.run() is thrown when the SGD computation diverges
            delete_test_dir()
            raise

        # create output log files
        np.savetxt(
            os.path.join(test_path, "{}_global_real_mean_squared_error_log".format(graph)),
            cluster.global_real_mean_squared_error_log,
            delimiter=','
        )
        np.savetxt(
            os.path.join(test_path, "{}_global_mean_squared_error_log".format(graph)),
            cluster.global_mean_squared_error_log,
            delimiter=','
        )
        np.savetxt(
            os.path.join(test_path, "{}_iterations_time_log".format(graph)),
            cluster.iterations_time_log,
            delimiter=','
        )

        print("Logs of {} simulation created at {}".format(graph, test_path))

        n_iter = len(cluster.global_mean_squared_error_log)

        ## BEGIN INSTANT PLOTS ##
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
        ## INSTANT PLOTS END ##

    if save_descriptor:
        with open(os.path.join(test_path, '.descriptor.txt'), 'a') as f:
            f.write('\n\n# duration (hh:mm:ss): ' + time.strftime('%H:%M:%S', time.gmtime(time.time() - begin_time)))

    if plot_from_file:
        plotter.plot_from_files(test_path, save_plot_to_file)

        # console.stdout.close()


def main1():
    # __X, __y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)
    X, y = mltoolbox.sample_from_function(100000, 10, mltoolbox.linear_function, 1, error_std_dev=1)
    cls = linear_model.SGDClassifier(loss="squared_loss", max_iter=100000)
    cls.fit(X, y)
    print(cls.score(X, y))


def main2():
    X, y = mltoolbox.sample_from_function(1000, 10, mltoolbox.linear_function, 1, error_std_dev=1, )
    cls = linear_model.SGDRegressor(penalty='none', alpha=0.01, max_iter=1000, shuffle=False,
                                    learning_rate='constant')
    cls.fit(X, y)
    print(cls.score(X, y))
    print(cls.predict(np.array([2, 4, 8]).reshape(1, -1)))


switch = 0

if __name__ == "__main__":
    eval("main{}()".format(switch))
