import random, math, time, os, pickle, shutil, datetime, pprint, warnings, argparse
import numpy as np
from sklearn.datasets.samples_generator import make_blobs, make_regression
from sklearn.preprocessing import normalize
from sklearn import linear_model, svm
from src.cluster import Cluster
from src import mltoolbox, graph_generator, statistics
from src.plotter import Plotter, plot_from_files
from src.mltoolbox import functions
from termcolor import colored as col

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


def main0(
        seed=None,
        n=None,
        n_samples=None,
        n_features=None,
        time_const_weight=None
):
    # console.stdout.screen = stdscr
    # console.stdout.open()

    ### BEGIN SETUP ###

    begin_time = time.time()
    # descriptor text placed at the beginning of _descriptor.txt file within the test folder

    setup_from_file = False
    setup_folder_path = Plotter.get_temp_test_folder_path_by_index()
    setup_file_path = os.path.join(setup_folder_path, ".setup.pkl")

    setup = dict()

    setup['seed'] = int(time.time()) if seed is None else seed
    setup['n'] = 100 if n is None else n

    setup['graphs'] = {
        # "0_diagonal": DIAGONAL(setup['n']),
        "1_cycle": CYCLE(setup['n']),  # degree = 1
        # "2_cycle-bi": CYCLE_B(setup['n']), # degree = 2
        "2_diam-expander": DIAM_EXP(setup['n']),  # degree = 2
        # "2_root-expander": ROOT_EXP(setup['n']),  # degree = 2
        # "3_regular": REGULAR(setup['n'], 3),  # degree = 3
        "4_regular": REGULAR(setup['n'], 4),  # degree = 4
        # "5_regular": REGULAR(setup['n'], 5),  # degree = 5
        # "6_regular": REGULAR(setup['n'], 6),  # degree = 6
        # "7_regular": REGULAR(setup['n'], 7),  # degree = 7
        "8_regular": REGULAR(setup['n'], 8),  # degree = 8
        # "9_regular": REGULAR(setup['n'], 9),  # degree = 9
        # "10_regular": REGULAR(setup['n'], 10),  # degree = 10
        "20_regular": REGULAR(setup['n'], 20),  # degree = 20
        "50_regular": REGULAR(setup['n'], 50),  # degree = 50
        "n-1_clique": CLIQUE(setup['n']),  # degree = n
        # "n-1_star": STAR(setup['n']),
    }

    # TRAINING SET SETUP

    setup['n_samples'] = 1000 if n_samples is None else n_samples
    setup['n_features'] = 100 if n_features is None else n_features

    setup['generator_function'] = 'reg2'  # svm, reg, reg2, skreg

    setup['smv_label_flip_prob'] = 0.10  # <-- ONLY FOR SVM

    setup['error_mean'] = 0.0
    setup['error_std_dev'] = 1.0  # <--

    setup['node_error_mean'] = 0.0
    setup['node_error_std_dev'] = 0.0  # <--

    r = np.random.uniform(4, 6)
    c = np.random.uniform(1, 3.8) * np.random.choice([-1, 1])
    setup['starting_weights_domain'] = [0, 0]  # [c - r, c + r]

    # TRAINING SET ALMOST FIXED SETUP
    # SETUP USED ONLY BY REGRESSION 'reg':
    setup['domain_radius'] = 6
    setup['domain_center'] = 0

    # CLUSTER SETUP 1
    setup['max_iter'] = None
    setup['max_time'] = 60  # seconds
    setup['method'] = "classic"
    setup['dual_averaging_radius'] = 10

    setup['alpha'] = 1e-2
    setup['learning_rate'] = "root_decreasing"  # constant, root_decreasing

    setup['time_distr_class'] = statistics.ExponentialDistribution
    setup['time_distr_param'] = [1]  # [rate] for exponential, [alpha,sigma] for pareto, [a,b] for uniform
    setup['time_const_weight'] = 0 if time_const_weight is None else time_const_weight

    setup['obj_function'] = 'mse'  # mse, hinge_loss, edgy_hinge_loss, score

    setup['metrics'] = []
    setup['real_metrics'] = []
    setup['metrics_type'] = 0  # 0: avg w on whole TS, 1: avg errors in nodes, 2: node's on whole TS
    setup['metrics_nodes'] = 'all'  # single node ID, list of IDs, otherwise all will be take into account in metrics

    # CLUSTER ALMOST FIXED SETUP
    setup['batch_size'] = 20
    setup['epsilon'] = None
    setup['shuffle'] = True

    # VERBOSE FLAGS
    # verbose <  0: no print at all except from errors
    # verbose == 0: default messages
    # verbose == 1: verbose + default messages
    # verbose == 2: verbose + default messages + input required to continue after each message
    verbose_main = verbose = 0
    verbose_cluster = 0
    verbose_node = 0
    verbose_task = 0
    verbose_plotter = 0

    if setup_from_file:
        with open(setup_file_path, 'rb') as setup_file:
            setup = pickle.load(setup_file)

    # OUTPUT SETUP
    save_test_to_file = True  # write output files to "test_log/{test_log_sub_folder}/" folder
    test_subfolder = "test_015_exp1lambda_reg_err1_Win0-0_1e-2alpha_{}nodes{}samp{}feat_{}c_classic".format(
        setup['n'], setup['n_samples'], setup['n_features'], setup['time_const_weight']
    )  # test folder inside test_log/
    test_title = test_subfolder

    # OUTPUT ALMOST FIXED SETUP
    test_root = "test_log"  # don't touch this
    temp_test_subfolder = datetime.datetime.now().strftime('%y-%m-%d_%H.%M.%S.%f')
    compress = True
    overwrite_if_already_exists = False  # overwrite the folder if it already exists or create a different one otherwise
    delete_folder_on_errors = True
    instant_plot = False  # instantly plot single simulations results
    plots = (
        "hinge_loss_iter",
        # "real_mse_iter",
    )
    save_plot_to_file = False
    save_descriptor = True  # create _descriptor.txt file
    ### END SETUP ###

    np.random.seed(setup['seed'])
    random.seed(setup['seed'])

    if setup['n'] % 2 != 0 and setup['n'] > 1:
        warnings.warn("Amount of nodes is odd (N={}), keep in mind graph generator "
                      "can misbehave in undirected graphs generation with odd nodes amount (it can "
                      "generate directed graphs instead)".format(setup['n']))

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

    # markov_matrix = normalize(__adjacency_matrix, axis=1, norm='l1')

    ### BEGIN TRAINING SET GEN ###
    # X, y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)

    if setup['generator_function'] == 'reg':
        [X, y, w] = functions.generate_regression_training_set_from_function(
            setup['n_samples'], setup['n_features'], functions.LinearYHatFunction.compute_value,
            domain_radius=setup['domain_radius'],
            domain_center=setup['domain_center'],
            error_mean=setup['error_mean'],
            error_std_dev=setup['error_std_dev']
        )
    elif setup['generator_function'] == 'reg2':
        X, y, w = functions.generate_regression_training_set(
            setup['n_samples'], setup['n_features'],
            error_mean=setup['error_mean'],
            error_std_dev=setup['error_std_dev']
        )
    elif setup['generator_function'] == 'svm':
        X, y, w = functions.svm_dual_averaging_training_set(
            setup['n_samples'], setup['n_features'],
            label_flip_prob=setup['smv_label_flip_prob']
        )
    elif setup['generator_function'] == 'skreg':
        X, y, w = make_regression(
            n_samples=setup['n_samples'],
            n_features=setup['n_features'],
            n_informative=setup['n_features'],
            n_targets=1,
            bias=1,
            effective_rank=None,
            tail_strength=1.0,
            noise=setup['error_std_dev'],
            shuffle=True,
            coef=True,
            random_state=None
        )
    else:
        delete_test_dir()
        raise Exception("{} is not a good training set generator function".format(setup['generator_function']))

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
    ### END TRAINING SET GEN ###

    ### BEGIN MAIN STUFFS ###

    # save setup object dump
    with open(os.path.join(test_path, '.setup.pkl'), "wb") as f:
        pickle.dump(setup, f, pickle.HIGHEST_PROTOCOL)

    setup['string_graphs'] = pprint.PrettyPrinter(indent=4).pformat(setup['graphs']).replace('array([', 'np.array([')

    # Fill descriptor with setup dictionary
    descriptor = """>>> Test Descriptor File
Title: {}
Date: {}
Summary: 

""".format(
        test_title if save_test_to_file else '',
        str(datetime.datetime.fromtimestamp(begin_time))
    )

    for k, v in setup.items():
        descriptor += "{} = {}\n".format(k, v)
    descriptor += "\n"

    # save descriptor file
    if save_descriptor:
        with open(os.path.join(test_path, '.descriptor.txt'), "w") as f:
            f.write(descriptor)

    # simulation for each adjacency matrix in setup['graphs'] dict
    for graph, adjmat in setup['graphs'].items():
        # set the seed again (each simulation must perform on the same cluster setup)
        np.random.seed(setup['seed'])
        random.seed(setup['seed'])

        cluster = None

        try:
            cluster = Cluster(adjmat, graph_name=graph, verbose=verbose_cluster)

            cluster.setup(
                X, y, w,
                obj_function=setup['obj_function'],
                method=setup['method'],
                max_iter=setup['max_iter'],
                max_time=setup['max_time'],
                batch_size=setup['batch_size'],
                dual_averaging_radius=setup['dual_averaging_radius'],
                epsilon=setup['epsilon'],
                alpha=setup['alpha'],
                learning_rate=setup['learning_rate'],
                metrics=setup['metrics'],
                real_metrics=setup["real_metrics"],
                metrics_type=setup['metrics_type'],
                metrics_nodes=setup['metrics_nodes'],
                shuffle=setup['shuffle'],
                time_distr_class=setup['time_distr_class'],
                time_distr_param=setup['time_distr_param'],
                time_const_weight=setup['time_const_weight'],
                node_error_mean=setup['node_error_mean'],
                node_error_std_dev=setup['node_error_std_dev'],
                starting_weights_domain=setup['starting_weights_domain'],
                verbose_node=verbose_node,
                verbose_task=verbose_task
            )

            cluster.run()
        except:
            # if the cluster throws an exception then delete the folder created to host its output files
            # the most common exception in cluster.run() is thrown when the SGD computation diverges
            delete_test_dir()
            print(
                "Exception in cluster object\n",
                "cluster.iteration=" + str(cluster.iteration)
            )
            raise

        extension = ''
        if compress:
            extension = '.gz'

        np.savetxt(
            os.path.join(test_path, "{}_iter_time_log{}".format(graph, extension)),
            cluster.logs["iter_time"],
            delimiter=','
        )

        np.savetxt(
            os.path.join(test_path, "{}_avg_iter_time_log{}".format(graph, extension)),
            cluster.logs["avg_iter_time"],
            delimiter=','
        )
        np.savetxt(
            os.path.join(test_path, "{}_max_iter_time_log{}".format(graph, extension)),
            cluster.logs["max_iter_time"],
            delimiter=','
        )

        # Save metrics logs
        for metrics_id, metrics_log in cluster.logs["metrics"].items():
            np.savetxt(
                os.path.join(test_path, "{}_{}_log{}".format(graph, metrics_id, extension)),
                metrics_log,
                delimiter=','
            )

        # Save real metrics logs
        for real_metrics_id, real_metrics_log in cluster.logs["real_metrics"].items():
            np.savetxt(
                os.path.join(test_path, "{}_real_{}_log{}".format(graph, real_metrics_id, extension)),
                real_metrics_log,
                delimiter=','
            )

        print("Logs of {} simulation created at {}".format(graph, test_path))

    if save_descriptor:
        with open(os.path.join(test_path, '.descriptor.txt'), 'a') as f:
            f.write('\n\n# duration (hh:mm:ss): ' + time.strftime('%H:%M:%S', time.gmtime(time.time() - begin_time)))

    if save_plot_to_file or instant_plot:
        plot_from_files(
            test_folder_path=test_path,
            save_plots_to_test_folder=save_plot_to_file,
            instant_plot=instant_plot,
            plots=plots,
            verbose=verbose_plotter
        )


def main1():
    # __X, __y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)
    X, y = functions.generate_regression_training_set_from_function(100000, 10, functions.linear_function, 1,
                                                                    error_std_dev=1)
    cls = linear_model.SGDClassifier(loss="squared_loss", max_iter=100000)
    cls.fit(X, y)
    print(cls.score(X, y))


def main2():
    X, y, w = functions.svm_dual_averaging_training_set(
        100000, 100
    )
    cls = svm.LinearSVC(max_iter=1000, verbose=True)
    cls.fit(X, y)
    print(cls.score(X, y))


switch = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plotter'
    )

    parser.add_argument(
        '-p',
        '--plots',
        nargs='+',
        help='List of plots to create',
        required=False,
        action='store',
        dest='plots',
        default=()
    )

    parser.add_argument(
        '-f', '--folder-path',
        action='store',
        default=None,
        required=False,
        help='Test folder from which load logs to',
        dest='folder_path'
    )

    parser.add_argument(
        '-t', '--temp-index',
        action='store',
        default=0,
        required=False,
        help='Test folder from which load logs to',
        dest='temp_index'
    )

    parser.add_argument(
        '-s', '--save',
        action='store_true',
        default=False,
        required=False,
        help='Specify whether save to file or not',
        dest='s_flag'
    )

    args = parser.parse_args()

    eval("main{}()".format(switch))
