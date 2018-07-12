import random, math, time, os, pickle, shutil, datetime, pprint, warnings, argparse
import numpy as np
from sklearn.datasets.samples_generator import make_blobs, make_regression
from sklearn.preprocessing import normalize
from sklearn import linear_model, svm
from src.cluster import Cluster
from src import mltoolbox, graphs, statistics
from src.plotter import Plotter, plot_from_files
from src.mltoolbox import functions
import matplotlib.pyplot as plt
from termcolor import colored as col


def generate_test_subfolder_name(setup, test_num, *argslist, parent_folder=""):
    def join_name_parts(*args):
        name = ""
        for a in args:
            name += str(a) + "_"
        return name[:-1]

    dataset = setup['dataset']
    distr = setup['time_distr_class'].shortname + '-'.join([str(e) for e in setup['time_distr_param'][0]])
    if setup['dataset'] == 'svm':
        error = str(setup['smv_label_flip_prob']) + 'flip'
        nodeserror = ''
    else:
        error = str(setup['error_std_dev']) + 'err'
        nodeserror = str(setup['node_error_std_dev']) + 'nodeErr'
    alpha = setup['learning_rate'][0] + str(setup['alpha']) + 'alpha'
    nodes = str(setup['n']) + 'n'
    samp = str(setup['n_samples']) + 'samp'
    feat = str(setup['n_features']) + 'feat'
    time = ('INF' if setup['max_time'] is None else str(setup['max_time'])) + 'time'
    iter = ('INF' if setup['max_iter'] is None else str(setup['max_iter'])) + 'iter'
    c = str(setup['time_const_weight']) + 'c'
    method = setup['method']
    shuffle = 'shuf' if setup['shuffle'] else '!shuf'
    w_domain = 'Win[{},{}]'.format(setup['starting_weights_domain'][0], setup['starting_weights_domain'][1])
    metrics = 'mtrT{}{}'.format(setup['metrics_type'], setup['metrics_nodes'])

    name = "test_" + str(test_num)

    for a in argslist:
        try:
            name = join_name_parts(name, eval(a))
        except NameError:
            pass

    return os.path.normpath(os.path.join(parent_folder, name))


def generate_time_distr_param_list(N, params):
    k = len(params)
    time_distr_param_list = [params[0] for _ in range(int(math.ceil(N / k)))]
    for i in range(1, len(params)):
        time_distr_param_list += [params[i] for _ in range(int(math.floor(N / k)))]
    return time_distr_param_list


def main0(
        seed=None,
        n=None,
        n_samples=None,
        n_features=None,
        time_const_weight=None,
        time_distr_class=None,
        time_distr_param=None
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

    setup['graphs'] = graphs.generate_n_nodes_graphs(setup['n'], [
        # "0-diagonal",
        "1-cycle",
        # "2-uniform_edges",
        "2-cycle",
        # "3-uniform_edges",
        #"3-cycle",
        # "4-uniform_edges",
        "4-cycle",
        # "5-uniform_edges",
        # "5-cycle",
        # "8-uniform_edges",
        "8-cycle",
        # "10-uniform_edges",
        # "10-cycle",
        # "20-uniform_edges",
        "20-cycle",
        # "50-uniform_edges",
        "50-cycle",
        # "80-uniform_edges",
        #"80-cycle",
        "99-clique",
    ])

    # TRAINING SET SETUP

    setup['n_samples'] = 100 if n_samples is None else n_samples
    setup['n_features'] = 100 if n_features is None else n_features

    setup['dataset'] = 'unireg'  # svm, unireg, reg, reg2, skreg

    setup['smv_label_flip_prob'] = 0.0  # <-- ONLY FOR SVM

    setup['error_mean'] = 0.0
    setup['error_std_dev'] = 0.0  # <--

    setup['node_error_mean'] = 0.0
    setup['node_error_std_dev'] = 0.0  # <--

    r = np.random.uniform(4, 10)
    c = np.random.uniform(1.1, 7.8) * np.random.choice([-1, 1, 1, 1])
    setup['starting_weights_domain'] = [-20, -20]  # [1, 2] #[c - r, c + r]

    # TRAINING SET ALMOST FIXED SETUP
    # SETUP USED ONLY BY REGRESSION 'reg':
    setup['domain_radius'] = 8
    setup['domain_center'] = 0

    # CLUSTER SETUP 1
    setup['max_iter'] = 100
    setup['max_time'] = None  # seconds
    setup['method'] = "classic"
    setup['dual_averaging_radius'] = 10

    setup['alpha'] = 1e-2
    setup['learning_rate'] = "constant"  # constant, root_decreasing

    setup['time_distr_class'] = statistics.ExponentialDistribution if time_distr_class is None else time_distr_class
    setup['time_distr_param'] = generate_time_distr_param_list(
        setup['n'],
        [[1]]
    ) if time_distr_param is None else time_distr_param  # exp[rate], par[a,s], U[a,b]
    setup['time_const_weight'] = 0 if time_const_weight is None else time_const_weight

    setup['real_y_activation_func'] = None
    setup['obj_function'] = 'mse'  # mse, hinge_loss, edgy_hinge_loss, score

    setup['metrics'] = []
    setup['real_metrics'] = []
    setup['real_metrics_toggle'] = False  # False to disable real_metrics computation (to speed up computation)
    setup['metrics_type'] = 0  # 0: avg w on whole TS, 1: avg errors in nodes, 2: node's on whole TS
    setup['metrics_nodes'] = 'all'  # single node ID, list of IDs, 'all', 'worst', 'best'
    setup['shuffle'] = False  # <--

    # CLUSTER ALMOST FIXED SETUP
    setup['batch_size'] = 20
    setup['epsilon'] = None

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

    test_subfolder = generate_test_subfolder_name(setup,
        'u040',
        'shuffle',
        'w_domain',
        'metrics',
        'dataset',
        'distr',
        'error',
        # 'nodeserror',
        'alpha',
        'nodes',
        #'samp',
        #'feat',
        'time',
        'iter',
        'c',
        #'method',
        parent_folder=""
    )

    test_title = test_subfolder

    # OUTPUT ALMOST FIXED SETUP
    test_root = "test_log"  # don't touch this
    temp_test_subfolder = datetime.datetime.now().strftime('%y-%m-%d_%H.%M.%S.%f')
    compress = True
    overwrite_if_already_exists = False  # overwrite the folder if it already exists or create a different one otherwise
    delete_folder_on_errors = True
    instant_plot = True  # instantly plot single simulations results
    plots = (
        "mse_iter",
        # "real_mse_iter",
    )
    save_plot_to_file = True
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

    if setup['dataset'] == 'reg':
        [X, y, w] = functions.generate_regression_training_set_from_function(
            setup['n_samples'], setup['n_features'], functions.LinearYHatFunction.compute_value,
            domain_radius=setup['domain_radius'],
            domain_center=setup['domain_center'],
            error_mean=setup['error_mean'],
            error_std_dev=setup['error_std_dev']
        )
    elif setup['dataset'] == 'reg2':
        X, y, w = functions.generate_regression_training_set(
            setup['n_samples'], setup['n_features'],
            error_mean=setup['error_mean'],
            error_std_dev=setup['error_std_dev']
        )
    elif setup['dataset'] == 'unireg':
        X, y, w = functions.generate_unidimensional_regression_training_set(setup['n_samples'])
    elif setup['dataset'] == 'svm':
        X, y, w = functions.generate_svm_dual_averaging_training_set(
            setup['n_samples'], setup['n_features'],
            label_flip_prob=setup['smv_label_flip_prob']
        )
    elif setup['dataset'] == 'skreg':
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
        raise Exception("{} is not a good training set generator function".format(setup['dataset']))

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

    # setup['string_graphs'] = pprint.PrettyPrinter(indent=4).pformat(setup['graphs']).replace('array([', 'np.array([')

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

    w_logs = {}
    node_w_logs = {}

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
                real_y_activation_function=setup['real_y_activation_func'],
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
                real_metrics_toggle=setup['real_metrics_toggle'],
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

        extension = '.txt'
        if compress:
            extension += '.gz'

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

        w_logs[graph] = cluster.w
        node_w_logs[graph] = cluster.nodes[0].training_task.w

        print("Logs of {} simulation created at {}".format(graph, test_path))

    if save_descriptor:
        with open(os.path.join(test_path, '.descriptor.txt'), 'a') as f:
            f.write('\n\n# duration (hh:mm:ss): ' + time.strftime('%H:%M:%S', time.gmtime(time.time() - begin_time)))

    #"""
    colors = Plotter.generate_color_dict_from_degrees(
        list(w_logs.keys()), setup['n']
    )

    plt.title("W(it)")
    plt.xlabel("iter")
    plt.ylabel("W(iter)")
    plt.yscale('linear')
    for graph in w_logs:
        plt.plot(
            list(range(len(w_logs[graph]))),
            w_logs[graph],
            label=graph,
            color=colors[graph],
            marker='o',
            markersize=2
            # **kwargs
        )
    plt.legend()
    plt.show()
    plt.close()

    plt.title("W_0(it) (W of Node 0)")
    plt.xlabel("iter")
    plt.ylabel("W_0(iter)")
    plt.yscale('linear')
    for graph in node_w_logs:
        plt.plot(
            list(range(len(node_w_logs[graph]))),
            node_w_logs[graph],
            label=graph,
            color=colors[graph],
            marker='o',
            markersize=2
            # **kwargs
        )
    plt.legend()
    plt.show()
    plt.close()
    #"""

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
    X, y, w = functions.generate_svm_dual_averaging_training_set(
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
