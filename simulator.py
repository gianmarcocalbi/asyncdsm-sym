import datetime
import shutil
import time
import random
from typing import *

import matplotlib.pyplot as plt
# from sklearn import linear_model, svm
from sklearn.datasets.samples_generator import make_regression

from src import statistics
from src.cluster import Cluster
from src.graphs import generate_n_nodes_graphs_list
from src.mltoolbox import datasets
from src.plotter import Plotter, plot_from_files
from src.utils import *


def run(
        seed: int = None,
        n: int = 100,
        graphs: Iterable[str] = (),
        n_samples: int = None,
        n_features: int = 100,
        dataset: str = None,
        smv_label_flip_prob: float = 0.0,
        error_mean: float = 0.0,
        error_std_dev: float = 0.0,
        node_error_mean: float = 0.0,
        node_error_std_dev: float = 0.0,
        starting_weights_domain: Union[List[float], Tuple[float]] = None,
        max_iter: int = None,
        max_time: float = None,
        method: Union[str, None] = 'classic',
        alpha: float = None,
        learning_rate: str = 'constant',
        spectrum_dependent_learning_rate: bool = False,
        dual_averaging_radius=10,
        time_distr_class: object = statistics.ExponentialDistribution,
        time_distr_param: list = (1,),
        time_distr_param_rule: str = None,
        time_const_weight: float = 0,
        real_y_activation_func: callable = None,
        obj_function: str = 'mse',
        average_model_toggle: bool = False,
        metrics: list = (),
        real_metrics: list = (),
        real_metrics_toggle: bool = False,
        metrics_type: int = 0,
        metrics_nodes: str = 'all',
        shuffle: bool = True,
        batch_size: int = 20,
        epsilon: float = None,
        save_test_to_file: bool = False,
        test_folder_name_struct: list = (
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
                'method'
        ),
        test_parent_folder: str = "",
        instant_plot: bool = False,
        plots: list = ('mse_iter',),
        save_plot_to_file: bool = False,
        plot_global_w: bool = False,
        plot_node_w: Union[bool, int, List[int]] = False,
        verbose_main: int = 0,
        verbose_cluster: int = 0,
        verbose_node: int = 0,
        verbose_task: int = 0,
        verbose_plotter: int = 0
):
    """
    Main method.

    Parameters
    ----------
    seed : int or None:
        Random simulation seed. If None will be taken from current time.
    n : int
        Amount of nodes in the cluster.
    graphs: List[str]
        List of topologies to run the simulation with.
    n_samples : int
        Total number of samples in the generated dataset.
    n_features : int
        Number of feature each sample will have.
    dataset : str
        Dataset label:
        - "reg": general customizable linear regression dataset;
        - "unireg": unidimensional regression;
        - "svm": multidimensional classification problem;
        - "unisvm": unidimensional dataset that changes with topology spectral gap;
        - "skreg" : regression dataset from sklearn library,
        - "sloreg" and "susysvm" from UCI's repository.
    smv_label_flip_prob : float
        Probability that a label is flipped in svm dataset generation.
        Kind of noise added in the dataset.
    error_mean : float
        Mean of noise to introduce in regression datasets.
    error_std_dev : float
        Standard deviation of noise introduced in regression datasets.
    node_error_mean : float
        Mean of the per-node noise introduced in each node's sample.
        Be careful because if used with SVM this can change values of labels.
    node_error_std_dev : float
        Standard deviation of the per-node noise introduced in each node's sample.
        Be careful because if used with SVM this can change values of labels.
    starting_weights_domain : List[float]
        In the form of [a,b]. Domain of each node's w is uniformly randomly picked within a and b.
    max_iter : int
        Maximum iteration after which the simulation is stopped.
    max_time : float
        Maximum time value after which the simulation is stopped.
    epsilon : float
        Accuracy threshold for objective function below which the simulation is stopped.
    method : str
        - "classic" : classic gradient descent, batch is equal to the whole dataset;
        - "stochastic" : stochastic gradient descent;
        - "batch" : batch gradient descent;
        - "subgradient" : subgradient projected gradient descent;
        - "dual_averaging" : dual averaging method.
    alpha : float
        Learning rate constant coefficient.
    learning_rate : str
        - 'constant' : the learning rate never changes during the simulation (it is euqual to alpha);
        - 'root_decreasing' : learning rate is alpha * 1/math.sqrt(K) where K = #iter.
    spectrum_dependent_learning_rate : bool
        If True the learning rate is also multiplied by math.sqrt(spectral_gap), so it is different for each graph.
    dual_averaging_radius : int
        Radius of the projection on the feasible set.
    time_distr_class : object
        Class of the random time distribution.
    time_distr_param : list or list of list
        Parameters list.
        See Also generate_time_distr_param_list.
    time_distr_param_rule : str
        Parameters distribution rule.
        See Also generate_time_distr_param_list.
    time_const_weight : float
        Weight assigned to constant part of the computation time.
        It is calculated as T_u(t) = E[X_u] * c + (1-c) * X_u(t).
    real_y_activation_func : function
        Activation function applied on real_y calculation.
    obj_function : str
        Identifier of the objective function (one of those declared in metrics.py).
    average_model_toggle : bool
        If True then the average over time of parameter vector is used istead of just x(k).
    metrics : list of str
        List of additional metrics to compute (objective function is automatically added to this list).
    real_metrics : list of str
        List of real metrics to compute (with regards to the real noiseless model).
    real_metrics_toggle : bool
        If False real metrics are not computed (useful to speed up the computation).
    metrics_type : int
        - 0 : metrics are computed over the whole dataset using model W equal to the avg of nodes' locla models;
        - 1 : metrics are computed as AVG of local nodes' metrics;
        - 2 : metrics are computed over the whole dataset using the model only from metrics_nodes (see below).
    metrics_nodes : int or list of int
        If type is int then it will be put into a list and treated as [int].
        Depends on the value of metrics_type:
        - metrics_type == 0 : no effects;
        - metrics_type == 1 : metrics are computed as avg of local metrics of nodes inside metrics_nodes list;
        - metrics_type == 2 : metrics are computed over the whole dataset using the model obtained as mean of
            nodes inside metrics_nodes.
    shuffle : bool
        If True the dataset is shuffled before being split into nodes, otherwise the dataset is untouched.
    batch_size : int
        Useful only for batch gradient descent, is the size of the batch.
    save_test_to_file : bool
        If True the test is saved to specified folder, otherwise it is stored into tempo folder.
    test_folder_name_struct : list
        See generate_test_subfolder_name.
    test_parent_folder : str
        Parent test folder: the test will be located in ./test_log/{$PARENT_FOLDER}/{$TEST_NAME_FOLDER}.
        Can be more than one-folder-deep!
    instant_plot : bool
        If True plots will be prompted upon finishing simulation. Be careful since it will pause the thread!
    plots : list of str
        List of plots' names to create / prompt upon finishing simulation.
        See plotter.py.
    save_plot_to_file : bool
        If True plots will be saved into .../{$TEST_FOLDER_NAME}/plots/ folder.
    plot_global_w : bool
        If True global W will be prompted after finishing simulation.
        This plot is never automatically saved, save it by yourself if you need to keep it.
    plot_node_w : list or False
        List of nodes to plot w which. If False nothing will be prompted.
    verbose_main : int
        Verbose policy in simulator.py script.
        - <0 : no print at all except from errors (unsafe).
        -  0 : default messages;
        -  1 : verbose + default messages
        -  2 : verbose + default messages + input required to continue after each message (simulation will be paused
            after each message and will require to press ENTER to go on, useful for debugging).
    verbose_cluster : int
        Verbose policy in cluster.py script.
        See verbose_main.
    verbose_node : int
        Verbose policy in node.py script.
        See verbose_main.
    verbose_task : int
        Verbose policy in tasks.py script.
        See verbose_main.
    verbose_plotter : int
        Verbose policy in plotter.py script.
        See verbose_main.

    Returns
    -------
    None
    """

    ### BEGIN SETUP ###

    begin_time = time.time()
    # descriptor text placed at the beginning of _descriptor.txt file within the test folder

    setup_from_file = False
    setup_folder_path = Plotter.get_temp_test_folder_path_by_index()
    setup_file_path = os.path.join(setup_folder_path, ".setup.pkl")

    setup = dict()

    setup['seed'] = int(time.time()) if seed is None else seed
    setup['n'] = n

    setup['graphs'] = generate_n_nodes_graphs_list(setup['n'], graphs)

    # TRAINING SET SETUP

    setup['n_samples'] = n_samples
    setup['n_features'] = n_features
    setup['dataset'] = dataset  # svm, unireg, reg, reg2, skreg
    setup['smv_label_flip_prob'] = smv_label_flip_prob
    setup['error_mean'] = error_mean
    setup['error_std_dev'] = error_std_dev
    setup['node_error_mean'] = node_error_mean
    setup['node_error_std_dev'] = node_error_std_dev

    # r = np.random.uniform(4, 10)
    # c = np.random.uniform(1.1, 7.8) * np.random.choice([-1, 1, 1, 1])
    # starting_weights_domain = [c - r, c + r]
    setup['starting_weights_domain'] = starting_weights_domain

    # TRAINING SET ALMOST FIXED SETUP
    # SETUP USED ONLY BY REGRESSION 'reg':
    setup['domain_radius'] = 8
    setup['domain_center'] = 0

    # CLUSTER SETUP 1
    setup['max_iter'] = max_iter
    setup['max_time'] = max_time  # units of time
    setup['method'] = method
    setup['dual_averaging_radius'] = dual_averaging_radius

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
    setup['obj_function'] = obj_function  # mse, hinge_loss, edgy_hinge_loss, cont_hinge_loss, score
    setup['average_model_toggle'] = average_model_toggle

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
    X, y, w = None, None, None
    # X, y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)

    if setup['dataset'] == 'reg':
        X, y, w = datasets.reg_dataset(
            setup['n_samples'], setup['n_features'],
            error_mean=setup['error_mean'],
            error_std_dev=setup['error_std_dev']
        )
    elif setup['dataset'] == 'svm':
        X, y, w = datasets.svm_dual_averaging_dataset(
            setup['n_samples'], setup['n_features'],
            label_flip_prob=setup['smv_label_flip_prob']
        )
    elif setup['dataset'] == 'unireg':
        X, y, w = datasets.unireg_dataset(setup['n'])
    elif setup['dataset'] == 'unisvm':
        X, y, w = datasets.unisvm_dual_averaging_dataset(
            setup['n'],
            label_flip_prob=setup['smv_label_flip_prob']
        )
    elif setup['dataset'] == 'enereg':
        X, y, w = datasets.load_appliances_energy_reg_dataset(setup['n_samples'])
    elif setup['dataset'] == 'sloreg':
        X, y, w = datasets.load_slice_localization_reg_dataset(setup['n_samples'])
    elif setup['dataset'] == 'susysvm':
        X, y, w = datasets.load_susy_svm_dataset(setup['n_samples'])
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
    elif setup['dataset'] in ['eigvecsvm', 'alteigvecsvm', 'multieigvecsvm']:
        pass
    else:
        delete_test_dir()
        raise Exception("{} is not a good training set generator function".format(setup['dataset']))

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

    ## SIMULATIONS
    # simulation for each adjacency matrix in setup['graphs'] dict
    for graph, adjmat in setup['graphs'].items():
        # set the seed again (each simulation must perform on the same cluster setup)
        np.random.seed(setup['seed'])
        random.seed(setup['seed'])

        cluster = None
        try:
            cluster = Cluster(adjmat, graph_name=graph, verbose=verbose_cluster)

            if setup['dataset'] in ['eigvecsvm', 'alteigvecsvm', 'multieigvecsvm']:
                # if using the ones matrix with this dataset, something wrong happens
                # so we use the last adj_mat also for the clique
                if 'clique' in graph:
                    max_deg = 0
                    max_deg_adjmat = adjmat
                    for G, A in setup['graphs'].items():
                        if 'clique' in G:
                            continue
                        d = int(G.split('-')[0])
                        if d > max_deg:
                            max_deg_adjmat = A
                            max_deg = d
                    if setup['dataset'] == 'eigvecsvm':
                        X, y, w = datasets.eigvecsvm_dataset_from_adjacency_matrix(max_deg_adjmat)
                    elif setup['dataset'] == 'alteigvecsvm':
                        X, y, w = datasets.eigvecsvm_dataset_from_expander(
                            setup['n'],
                            max_deg,
                            matrix_type='uniform-weighted'
                        )
                    elif setup['dataset'] == 'multieigvecsvm':
                        X, y, w = datasets.multieigvecsvm_dataset_from_expander(
                            setup['n_samples'], setup['n'], max_deg)
                else:
                    if setup['dataset'] == 'eigvecsvm':
                        X, y, w = datasets.eigvecsvm_dataset_from_adjacency_matrix(adjmat)
                    elif setup['dataset'] == 'alteigvecsvm':
                        deg = int(graph.split('-')[0])
                        X, y, w = datasets.eigvecsvm_dataset_from_expander(
                            setup['n'],
                            deg,
                            matrix_type='uniform-weighted'
                        )
                    elif setup['dataset'] == 'multieigvecsvm':
                        deg = int(graph.split('-')[0])
                        X, y, w = datasets.multieigvecsvm_dataset_from_expander(
                            setup['n_samples'], setup['n'], deg)

            alpha = setup['alpha']
            if spectrum_dependent_learning_rate:
                alpha *= math.sqrt(uniform_weighted_Pn_spectral_gap_from_adjacency_matrix(adjmat))

            cluster.setup(
                X, y, w,
                real_y_activation_function=setup['real_y_activation_func'],
                obj_function=setup['obj_function'],
                average_model_toggle=average_model_toggle,
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
                cluster.run()
            else:
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

    colors = Plotter.generate_rainbow_color_dict_from_graph_keys(
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


if __name__ == "__main__":
    pass
    # this module's functions are called from inside main.py script
