import random, math, time, os, pickle, shutil, datetime, pprint, warnings, argparse
import numpy as np
from sklearn.datasets.samples_generator import make_blobs, make_regression
from sklearn.preprocessing import normalize
from sklearn import linear_model, svm
from src.cluster import Cluster
from src import mltoolbox, statistics
from src.plotter import Plotter, plot_from_files
from src.mltoolbox import functions
from src.graphs import generate_n_nodes_graphs
import matplotlib.pyplot as plt
from src.utils import *
from termcolor import colored as col


def generate_test_subfolder_name(setup, test_num, *argslist, parent_folder=""):
    def join_name_parts(*args):
        name = ""
        for a in args:
            name += str(a) + "_"
        return name[:-1]

    dataset = setup['dataset']
    distr = setup['time_distr_class'].shortname + '[' + '-'.join([str(e) for e in setup['time_distr_param'][0]]) + ']'
    distr_rule = str(setup['time_distr_param_rule']) + 'Rule'
    if setup['dataset'] == 'svm':
        error = str(setup['smv_label_flip_prob']) + 'flip'
        nodeserror = ''
    else:
        error = str(setup['error_std_dev']) + 'err'
        nodeserror = str(setup['node_error_std_dev']) + 'nodeErr'

    alpha = str(setup['learning_rate'][0].upper()) + str(setup['alpha']) + 'alpha'
    if setup['spectrum_dependent_learning_rate']:
        alpha = "sg" + alpha
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


def generate_time_distr_param_list(N, params, rule):
    if not isinstance(params[0], list) and not isinstance(params[0], tuple):
        params = [params]

    k = len(params)
    time_distr_param_list = []

    if rule == 'split':
        time_distr_param_list = [params[0] for _ in range(int(math.ceil(N / k)))]
        for i in range(1, len(params)):
            time_distr_param_list += [params[i] for _ in range(int(math.floor(N / k)))]
    elif rule == 'random':
        for i in range(N):
            time_distr_param_list.append(np.random.choice(params))
    elif rule == 'alternate':
        for i in range(N):
            time_distr_param_list.append(params[i % k])
    else:
        for i in range(N):
            time_distr_param_list.append(params[0])

    return time_distr_param_list


def main(
        seed=None,
        n=100,
        graphs=[],
        n_samples=None,
        n_features=100,
        dataset=None,
        smv_label_flip_prob=0.0,
        error_mean=0.0,
        error_std_dev=0.0,
        node_error_mean=0.0,
        node_error_std_dev=0.0,
        starting_weights_domain=None,
        max_iter=None,
        max_time=None,
        method='classic',
        alpha=None,
        learning_rate='constant',
        spectrum_dependent_learning_rate=False,
        time_distr_class=statistics.ExponentialDistribution,
        time_distr_param=(1,),
        time_distr_param_rule=None,
        time_const_weight=0,
        real_y_activation_func=None,
        obj_function='mse',
        metrics=[],
        real_metrics=[],
        real_metrics_toggle=False,
        metrics_type=0,
        metrics_nodes='all',
        shuffle=True,
        batch_size=20,
        epsilon=None,
        save_test_to_file=False,
        test_folder_name_struct=(
                'u040',
                'shuffle',
                'w_domain',
                'metrics',
                'dataset',
                'distr',
                'error',
                'nodeserror',
                'alpha',
                'nodes',
                'samp',
                'feat',
                'time',
                'iter',
                'c',
                'method',
        ),
        test_parent_folder="",
        instant_plot=False,
        plots=('mse_iter',),
        save_plot_to_file=False,
        plot_global_w=False,
        plot_node_w=False,
        verbose_main=0,
        verbose_cluster=0,
        verbose_node=0,
        verbose_task=0,
        verbose_plotter=0
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
    setup['n'] = n

    setup['graphs'] = generate_n_nodes_graphs(setup['n'], graphs)

    # TRAINING SET SETUP

    setup['n_samples'] = n_samples
    setup['n_features'] = n_features
    setup['dataset'] = dataset  # svm, unireg, reg, reg2, skreg
    setup['smv_label_flip_prob'] = smv_label_flip_prob
    setup['error_mean'] = error_mean
    setup['error_std_dev'] = error_std_dev
    setup['node_error_mean'] = node_error_mean
    setup['node_error_std_dev'] = node_error_std_dev

    r = np.random.uniform(4, 10)
    c = np.random.uniform(1.1, 7.8) * np.random.choice([-1, 1, 1, 1])
    setup['starting_weights_domain'] = starting_weights_domain  # [c - r, c + r]

    # TRAINING SET ALMOST FIXED SETUP
    # SETUP USED ONLY BY REGRESSION 'reg':
    setup['domain_radius'] = 8
    setup['domain_center'] = 0

    # CLUSTER SETUP 1
    setup['max_iter'] = max_iter
    setup['max_time'] = max_time  # seconds
    setup['method'] = method
    setup['dual_averaging_radius'] = 10

    setup['alpha'] = alpha
    setup['learning_rate'] = learning_rate  # constant, root_decreasing
    setup['spectrum_dependent_learning_rate'] = spectrum_dependent_learning_rate

    setup['time_distr_class'] = time_distr_class
    setup['time_distr_param'] = generate_time_distr_param_list(
        setup['n'],
        time_distr_param,
        time_distr_param_rule

    )  # exp[rate], par[a,s], U[a,b]
    setup['time_distr_param_rule'] = time_distr_param_rule
    setup['time_const_weight'] = time_const_weight
    setup['real_y_activation_func'] = real_y_activation_func
    setup['obj_function'] = obj_function  # mse, hinge_loss, edgy_hinge_loss, score

    setup['metrics'] = metrics
    setup['real_metrics'] = real_metrics
    setup['real_metrics_toggle'] = real_metrics_toggle  # False to disable real_metrics computation (for better perf.)
    setup['metrics_type'] = metrics_type  # 0: avg w on whole TS, 1: avg errors in nodes, 2: node's on whole TS
    setup['metrics_nodes'] = metrics_nodes  # single node ID, list of IDs, 'all', 'worst', 'best'
    setup['shuffle'] = shuffle  # <--

    # CLUSTER ALMOST FIXED SETUP
    setup['batch_size'] = batch_size
    setup['epsilon'] = epsilon

    # VERBOSE FLAGS
    # verbose <  0: no print at all except from errors
    # verbose == 0: default messages
    # verbose == 1: verbose + default messages
    # verbose == 2: verbose + default messages + input required to continue after each message
    verbose = verbose_main

    if setup_from_file:
        with open(setup_file_path, 'rb') as setup_file:
            setup = pickle.load(setup_file)

    # OUTPUT SETUP
    test_subfolder = generate_test_subfolder_name(setup,
        *test_folder_name_struct,
        parent_folder=test_parent_folder
    )

    test_title = test_subfolder

    # OUTPUT ALMOST FIXED SETUP
    test_root = "test_log"  # don't touch this
    temp_test_subfolder = datetime.datetime.now().strftime('%y-%m-%d_%H.%M.%S.%f')
    compress = True
    overwrite_if_already_exists = False  # overwrite the folder if it already exists or create a different one otherwise
    delete_folder_on_errors = True
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
    elif setup['dataset'] == 'unisvm':
        pass
    elif setup['dataset'] == 'unireg':
        X, y, w = functions.generate_unidimensional_regression_training_set(setup['n'])
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
        sg = None
        try:
            cluster = Cluster(adjmat, graph_name=graph, verbose=verbose_cluster)

            if setup['dataset'] == 'unisvm':
                X, y, w = functions.generate_unidimensional_svm_training_set_from_expander_adj_mat(adjmat)

            alpha = setup['alpha']
            if spectrum_dependent_learning_rate:
                if 'expander' in graph:
                    sg = Pn_spectral_gap_from_adjacency_matrix(adjmat)
                elif 'cycle' in graph:
                    sg = n_cycle_spectral_gap_approx_from_adjacency_matrix(adjmat)
                else:
                    sg = mtm_spectral_gap_from_adjacency_matrix(adjmat)

                alpha *= math.sqrt(sg)

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
                alpha=alpha,
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

            if setup['method'] is None:
                cluster.run_void()
            else:
                cluster.run_void()

            """except ValueError:
            print('Graph {} spectral gap sqrt raised ValueError exception (sg = {})'.format(
                graph,
                sg
            ))
            continue"""

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
        if not setup['method'] is None:
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

        if plot_global_w:
            w_logs[graph] = cluster.w

        if not plot_node_w is False:
            try:
                node_w_logs[graph] = np.array(cluster.nodes[plot_node_w[0]].training_task.w)
                for i in range(1, len(plot_node_w)):
                    node_w_logs[graph] += np.array(cluster.nodes[plot_node_w[i]].training_task.w)
                node_w_logs[graph] /= len(plot_node_w)
            except:
                plot_node_w = False

        print("Logs of {} simulation created at {}".format(graph, test_path))

    if save_descriptor:
        with open(os.path.join(test_path, '.descriptor.txt'), 'a') as f:
            f.write('\n\n# duration (hh:mm:ss): ' + time.strftime('%H:%M:%S', time.gmtime(time.time() - begin_time)))

    colors = Plotter.generate_color_dict_from_graph_keys(
        list(w_logs.keys()), setup['n']
    )

    if plot_global_w:
        plt.suptitle(test_subfolder)
        plt.title("W(it)")
        plt.xlabel("iter")
        plt.ylabel("Global W at iteration")
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

    if not plot_node_w is False:
        plt.suptitle(test_subfolder)
        plt.title("W_{0}(it) (W of Node {0} at iteration)".format(plot_node_w))
        plt.xlabel("iter")
        plt.ylabel("W_{}(iter)".format(plot_node_w))
        plt.yscale('linear')
        for graph in node_w_logs:
            plt.plot(
                list(range(len(node_w_logs[graph]))),
                [p[0] for p in node_w_logs[graph]],
                label=graph,
                color=colors[graph],
                marker='o',
                markersize=2
            )
        plt.legend()
        plt.show()
        plt.close()

    if save_plot_to_file or instant_plot:
        plot_from_files(
            test_folder_path=test_path,
            save_plots_to_test_folder=save_plot_to_file,
            instant_plot=instant_plot,
            plots=plots,
            verbose=verbose_plotter,
            test_tag=test_subfolder
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


if __name__ == "__main__":
    raise Exception("Main shouldn't be called directly")
    main(
        seed=None,
        n=100,
        graphs=(
            # "0-diagonal",
            "1-cycle",
                # "2-uniform_edges",
            "2-cycle",
                # "3-uniform_edges",
                # "3-cycle",
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
                # "80-cycle",
            "99-clique",
        ),
        n_samples=None,
        n_features=100,
        dataset=None,
        smv_label_flip_prob=0.0,
        error_mean=0.0,
        error_std_dev=0.0,
        node_error_mean=0.0,
        node_error_std_dev=0.0,
        starting_weights_domain=None,
        max_iter=None,
        max_time=None,
        method='classic',
        alpha=None,
        learning_rate='constant',
        time_distr_class=statistics.ExponentialDistribution,
        time_distr_param=(1,),
        time_distr_param_list=None,
        time_const_weight=0,
        real_y_activation_func=None,
        obj_function='mse',
        metrics=(),
        real_metrics=(),
        real_metrics_toggle=False,
        metrics_type=0,
        metrics_nodes='all',
        shuffle=True,
        batch_size=20,
        epsilon=None,
        save_test_to_file=False,
        test_folder_name_struct=(
            'u040',
            'shuffle',
            'w_domain',
            'metrics',
            'dataset',
            'distr',
            'error',
            'nodeserror',
            'alpha',
            'nodes',
            'samp',
            'feat',
            'time',
            'iter',
            'c',
            'method',
        ),
        test_parent_folder="",
        instant_plot=False,
        plots=('mse_iter',),
        save_plot_to_file=False,
        plot_global_w=False,
        plot_node_w=False,
        verbose_main=0,
        verbose_cluster=0,
        verbose_node=0,
        verbose_task=0,
        verbose_plotter=0
    )
