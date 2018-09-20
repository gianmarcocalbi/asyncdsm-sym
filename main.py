import argparse

from sklearn.preprocessing import normalize

import simulator
from src import statistics, graphs
from src.utils import *


def get_graphs(graph_type, nodes):
    deg = {
        100: [
            2,
            3,
            4,
            8,
            16,
            32,
            64,
            99
        ],
        400: [2, 3, 4, 6, 20, 50, 100, 200, 300, 399],
        1000: [2, 3, 4, 8, 16, 20, 30, 40, 50, 100, 200, 500, 999]
    }[nodes]

    g = []

    for d in deg:
        if d == nodes - 1:
            g.append(str(d) + '-clique')
        else:
            g.append(str(d) + '-' + graph_type)

    return g


def run(core=-1):
    """
    Parameters
    ----------
    core: int
        Integer from argparse module to be used as switcher.

    Returns
    -------
    None
    """

    """for g_name, A in graphs.generate_n_nodes_graphs(100, get_graphs('cycle', 100)).items():
        M = A / sum(A[0])
        eigenvals = np.linalg.eigvals(M)
        np.sort(eigenvals)
        print(g_name + ' ' + str(abs(eigenvals[1])))"""

    test_on_eigvecsvm_dataset(seed=22052010, graph_type='expander', n=100, distr='par', metrics_nodes='worst',
        alt_exp=True, alert=False)

    if core == 0:
        test_on_eigvecsvm_dataset(seed=22052010, graph_type='expander', n=100, distr='par', metrics_nodes='worst',
            alert=False)
    elif core == 1:
        test_on_eigvecsvm_dataset(seed=22052010, graph_type='expander', n=100, distr='unif', metrics_nodes='worst',
            alert=False)
    elif core == 2:
        test_on_eigvecsvm_dataset(seed=22052010, graph_type='expander', n=100, distr='exp', metrics_nodes='worst',
            alert=False)
    elif core == 3:
        test_on_eigvecsvm_dataset(seed=22052010, graph_type='undir_cycle', n=100, distr='par', metrics_nodes='worst',
            alert=False)
    elif core == 4:
        test_on_eigvecsvm_dataset(seed=22052010, graph_type='undir_cycle', n=100, distr='unif', metrics_nodes='worst',
            alert=False)
    elif core == 5:
        test_on_eigvecsvm_dataset(seed=22052010, graph_type='undir_cycle', n=100, distr='exp', metrics_nodes='worst',
            alert=False)
    elif core == 6:
        test_on_eigvecsvm_dataset(seed=22052010, graph_type='expander', n=100, distr='par', metrics_nodes='worst',
            alt_exp=True, alert=False)
    elif core == 7:
        test_on_eigvecsvm_dataset(seed=22052010, graph_type='expander', n=100, distr='unif', metrics_nodes='worst',
            alt_exp=True, alert=False)
    elif core == 8:
        test_on_eigvecsvm_dataset(seed=22052010, graph_type='expander', n=100, distr='exp', metrics_nodes='worst',
            alt_exp=True, alert=False)


def test_on_eigvecsvm_dataset(
        seed=None,
        graph_type='expander',
        graphs_list=None,
        n=100, distr='par',
        metrics_nodes='all',
        alt_exp=False,
        alert=True
):
    if alert:
        print('test_exp_on_unisvm_dataset()')
        print('n={}, distr={}, metrics_nodes={}'.format(n, distr, metrics_nodes))
        input("click [ENTER] to continue or [CTRL]+[C] to abort")

    time_distr_class = {
        'exp': statistics.ExponentialDistribution, 'unif': statistics.UniformDistribution,
        'par': statistics.Type2ParetoDistribution
    }[distr]
    time_distr_param = {'exp': [[1]], 'unif': [[0, 2]], 'par': [[3, 2]], }[distr]
    if metrics_nodes in ['worst', 'all']:
        metrics_type = {'worst': 2, 'all': 0, 'best': 2}[metrics_nodes]
    else:
        metrics_type = 2

    if graphs_list is None or len(graphs_list) == 0:
        graphs_list = get_graphs(graph_type, n)

    if alt_exp:
        dataset = 'alteigvecsvm'
    else:
        dataset = 'eigvecsvm'

    simulator.run(
        seed=seed,
        n=n,
        graphs=graphs_list,
        dataset=dataset,
        starting_weights_domain=[1, 1],
        smv_label_flip_prob=0.00,
        max_iter=5000,
        max_time=None,
        alpha=1e-1,
        learning_rate='constant',
        spectrum_dependent_learning_rate=False,
        time_distr_class=time_distr_class,
        time_distr_param=time_distr_param,
        obj_function='cont_hinge_loss',
        epsilon=-math.inf,
        average_model_toggle=True,
        metrics=[],
        metrics_type=metrics_type,
        metrics_nodes=metrics_nodes,
        shuffle=False,
        save_test_to_file=True,
        test_folder_name_struct=[
            '3'+graph_type,
            'dataset',
            'alpha',
            'nodes',
            'shuffle',
            'w_domain',
            'distr',
            'time',
            'iter',
            'metrics'
        ],
        test_parent_folder="",
        instant_plot=False,
        plots=['cont_hinge_loss_iter', 'cont_hinge_loss_time'],
        save_plot_to_file=False,
        plot_global_w=False,
        plot_node_w=False
    )


def test_exp_on_newreg_dataset(seed=None, n=100, distr='par', metrics_nodes='all', alert=True):
    if alert:
        print('test_exp_on_reg_dataset()')
        print('n={}, distr={}, metrics_nodes={}'.format(n, distr, metrics_nodes))
        input("click [ENTER] to continue or [CTRL]+[C] to abort")

    time_distr_class = {
        'exp': statistics.ExponentialDistribution, 'unif': statistics.UniformDistribution,
        'par': statistics.Type2ParetoDistribution, 'real': statistics.SparkRealTimings
    }[distr]
    time_distr_param = {'exp': [[1]], 'unif': [[0, 2]], 'par': [[3, 2]], 'real': []}[distr]
    graphs = {
        100: ['2-expander', '3-expander', '4-expander', '8-expander', '10-expander', '20-expander', '50-expander',
            '80-expander', '99-clique', ],
        400: ['2-expander', '3-expander', '4-expander', '8-expander', '20-expander', '50-expander', '100-expander',
            '200-expander', '300-expander', '399-clique', ],
        1000: ['2-expander', '3-expander', '4-expander', '8-expander', '16-expander', '20-expander', '30-expander',
            '40-expander', '50-expander', '100-expander', '200-expander', '500-expander', '999-clique', ]
    }[n]
    metrics_type = {'worst': 2, 'all': 0}[metrics_nodes]

    simulator.run(
        seed=seed,
        n=n,
        graphs=graphs,
        n_samples=1000,
        n_features=100,
        dataset='newreg',
        method='subgradient',
        dual_averaging_radius=5,
        error_std_dev=1,
        starting_weights_domain=[20, 30],
        max_iter=200,
        max_time=None,
        alpha=1,
        learning_rate='constant',
        spectrum_dependent_learning_rate=False,
        time_distr_class=time_distr_class,
        time_distr_param=time_distr_param,
        obj_function='mse',
        metrics=[],
        metrics_type=metrics_type,
        metrics_nodes=metrics_nodes,
        shuffle=True,
        save_test_to_file=True,
        test_folder_name_struct=[
            '',
            'dataset',
            'w_domain',
            'nodes',
            'distr',
            'metrics',
            'alpha',
            'samp',
            'feat',
            'time',
            'iter'
        ],
        test_parent_folder="",
        instant_plot=False,
        plots=['mse_iter', 'mse_time'],
        save_plot_to_file=False,
        plot_global_w=False,
        plot_node_w=False
    )


def test_exp_on_reg2_dataset(seed=None, n=100, distr='par', metrics_nodes='all', alert=True):
    if alert:
        print('test_exp_on_reg_dataset()')
        print('n={}, distr={}, metrics_nodes={}'.format(n, distr, metrics_nodes))
        input("click [ENTER] to continue or [CTRL]+[C] to abort")

    time_distr_class = {
        'exp': statistics.ExponentialDistribution, 'unif': statistics.UniformDistribution,
        'par': statistics.Type2ParetoDistribution, 'real': statistics.SparkRealTimings
    }[distr]
    time_distr_param = {'exp': [[1]], 'unif': [[0, 2]], 'par': [[3, 2]], 'real': []}[distr]
    graphs = {
        100: ['2-expander', '3-expander', '4-expander', '8-expander', '10-expander', '20-expander', '50-expander',
            '80-expander', '99-clique', ],
        400: ['2-expander', '3-expander', '4-expander', '8-expander', '20-expander', '50-expander', '100-expander',
            '200-expander', '300-expander', '399-clique', ],
        1000: ['2-expander', '3-expander', '4-expander', '8-expander', '16-expander', '20-expander', '30-expander',
            '40-expander', '50-expander', '100-expander', '200-expander', '500-expander', '999-clique', ]
    }[n]
    metrics_type = {'worst': 2, 'all': 0}[metrics_nodes]

    simulator.run(
        seed=seed,
        n=n,
        graphs=graphs,
        n_samples=1000,
        n_features=100,
        dataset='reg2',
        method='subgradient',
        dual_averaging_radius=300000,
        error_std_dev=1,
        starting_weights_domain=[2, 3],
        max_iter=200,
        max_time=None,
        alpha=1e-1,
        learning_rate='constant',
        spectrum_dependent_learning_rate=False,
        time_distr_class=time_distr_class,
        time_distr_param=time_distr_param,
        obj_function='mse',
        metrics=[],
        metrics_type=metrics_type,
        metrics_nodes=metrics_nodes,
        shuffle=True,
        save_test_to_file=False,
        test_folder_name_struct=[
            'conv_synt_reg',
            # 'dataset',
            # 'w_domain',
            'nodes',
            'distr',
            'metrics',
            'alpha',
            'samp',
            'feat',
            'time',
            'iter'
        ],
        test_parent_folder="",
        instant_plot=True,
        plots=['mse_iter', 'mse_time'],
        save_plot_to_file=False,
        plot_global_w=False,
        plot_node_w=False
    )


def test_exp_on_dual_average_svm(seed=None, n=100, distr='par', metrics_nodes='all', alert=True):
    if alert:
        print('test_exp_on_dual_average_svm()')
        print('n={}, distr={}, metrics_nodes={}'.format(n, distr, metrics_nodes))
        input("click [ENTER] to continue or [CTRL]+[C] to abort")

    time_distr_class = {
        'exp': statistics.ExponentialDistribution, 'unif': statistics.UniformDistribution,
        'par': statistics.Type2ParetoDistribution, 'real': statistics.SparkRealTimings
    }[distr]
    time_distr_param = {'exp': [[1]], 'unif': [[0, 2]], 'par': [[3, 2]], 'real': []}[distr]
    graphs = {
        100: ['2-expander', '3-expander', '4-expander', '8-expander', '10-expander', '20-expander', '50-expander',
            '80-expander', '99-clique', ],
        400: ['2-expander', '3-expander', '4-expander', '8-expander', '20-expander', '50-expander', '100-expander',
            '200-expander', '300-expander', '399-clique', ],
        1000: ['2-expander', '3-expander', '4-expander', '8-expander', '16-expander', '20-expander', '30-expander',
            '40-expander', '50-expander', '100-expander', '200-expander', '500-expander', '999-clique', ]
    }[n]
    metrics_type = {'worst': 2, 'all': 0}[metrics_nodes]

    simulator.run(
        seed=seed,
        n=n,
        graphs=graphs,
        n_samples=1000,
        n_features=100,
        dataset='svm',
        smv_label_flip_prob=0.05,
        starting_weights_domain=[-2, 2],
        max_iter=500,
        max_time=None,
        method='classic',
        alpha=1,
        learning_rate='constant',
        time_distr_class=time_distr_class,
        time_distr_param=time_distr_param,
        time_distr_param_rule=None,
        time_const_weight=0,
        obj_function='hinge_loss',
        spectrum_dependent_learning_rate=False,
        metrics=[],
        metrics_type=metrics_type,
        metrics_nodes=metrics_nodes,
        real_metrics_toggle=False,
        shuffle=True,
        save_test_to_file=True,
        test_folder_name_struct=[
            'conv_synt_svm',
            # 'dataset',
            'alpha',
            'nodes',
            # 'shuffle',
            'distr',
            'metrics',
            'time',
            'iter'
        ],
        test_parent_folder="",
        instant_plot=False,
        plots=['hinge_loss_iter', 'hinge_loss_time'],
        save_plot_to_file=False,
        plot_global_w=False,
        plot_node_w=False
    )


def test_exp_on_enereg_dataset(seed=None, n=100, distr='par', metrics_nodes='all', alert=True):
    if alert:
        print('test_exp_on_enereg_dataset()')
        print('n={}, distr={}, metrics_nodes={}'.format(n, distr, metrics_nodes))
        input("click [ENTER] to continue or [CTRL]+[C] to abort")

    time_distr_class = {
        'exp': statistics.ExponentialDistribution, 'unif': statistics.UniformDistribution,
        'par': statistics.Type2ParetoDistribution, 'real': statistics.SparkRealTimings
    }[distr]
    time_distr_param = {'exp': [[1]], 'unif': [[0, 2]], 'par': [[3, 2]], 'real': []}[distr]
    graphs = {
        100: ['2-expander', '3-expander', '4-expander', '8-expander', '10-expander', '20-expander', '50-expander',
            '80-expander', '99-clique', ],
        400: ['2-expander', '3-expander', '4-expander', '8-expander', '20-expander', '50-expander', '100-expander',
            '200-expander', '300-expander', '399-clique', ],
        1000: ['2-expander', '3-expander', '4-expander', '8-expander', '20-expander', '30-expander', '40-expander',
            '50-expander', '100-expander', '200-expander', '500-expander', '999-clique', ]
    }[n]
    metrics_type = {'worst': 2, 'all': 0}[metrics_nodes]

    # linear_regression result = 8658.025636439765
    simulator.run(
        seed=seed,
        n=n,
        graphs=graphs,
        n_samples=14000,
        dataset='enereg',
        starting_weights_domain=[0, 0],
        max_iter=1000,
        max_time=None,
        alpha=1e-6,
        learning_rate='constant',
        spectrum_dependent_learning_rate=True,
        time_distr_class=time_distr_class,
        time_distr_param=time_distr_param,
        obj_function='mse',
        metrics=[],
        metrics_type=metrics_type,
        metrics_nodes=metrics_nodes,
        shuffle=True,
        save_test_to_file=True,
        test_folder_name_struct=[
            're100',
            'dataset',
            # 'w_domain',
            'nodes',
            'distr',
            'metrics',
            'alpha',
            'samp',
            # 'feat',
            'time',
            'iter'
        ],
        test_parent_folder="",
        instant_plot=True,
        plots=['mse_iter', 'mse_time'],
        save_plot_to_file=True,
        plot_global_w=False,
        plot_node_w=False
    )


def test_exp_on_sloreg_dataset(seed=None, n=100, distr='par', metrics_nodes='all', alert=True):
    if alert:
        print('test_exp_on_enereg_dataset()')
        print('n={}, distr={}, metrics_nodes={}'.format(n, distr, metrics_nodes))
        input("click [ENTER] to continue or [CTRL]+[C] to abort")

    time_distr_class = {
        'exp': statistics.ExponentialDistribution, 'unif': statistics.UniformDistribution,
        'par': statistics.Type2ParetoDistribution, 'real': statistics.SparkRealTimings
    }[distr]
    time_distr_param = {'exp': [[1]], 'unif': [[0, 2]], 'par': [[3, 2]], 'real': []}[distr]
    graphs = {
        100: ['2-expander', '3-expander', '4-expander', '8-expander', '10-expander', '20-expander', '50-expander',
            '80-expander', '99-clique', ],
        400: ['2-expander', '3-expander', '4-expander', '8-expander', '20-expander', '50-expander', '100-expander',
            '200-expander', '300-expander', '399-clique', ],
        1000: ['2-expander', '3-expander', '4-expander', '8-expander', '16-expander', '20-expander', '30-expander',
            '40-expander', '50-expander', '100-expander', '200-expander', '500-expander', '999-clique', ]
    }[n]
    metrics_type = {'worst': 2, 'all': 0}[metrics_nodes]

    # linear_regression result = 67.30972320004327

    simulator.run(
        seed=seed,
        n=n,
        graphs=graphs,
        n_samples=52000,
        dataset='sloreg',
        starting_weights_domain=[2, 3],
        max_iter=1000,
        max_time=None,
        alpha=5e-6,
        learning_rate='constant',
        spectrum_dependent_learning_rate=False,
        time_distr_class=time_distr_class,
        time_distr_param=time_distr_param,
        obj_function='mse',
        metrics=[],
        metrics_type=metrics_type,
        metrics_nodes=metrics_nodes,
        shuffle=True,
        save_test_to_file=False,
        test_folder_name_struct=[
            'fixedseed_real_reg',
            # 'dataset',
            # 'w_domain',
            'nodes',
            'distr',
            'metrics',
            'alpha',
            'samp',
            # 'feat',
            'time',
            'iter'
        ],
        test_parent_folder="",
        instant_plot=False,
        plots=['mse_iter', 'mse_time'],
        save_plot_to_file=False,
        plot_global_w=False,
        plot_node_w=False
    )


def test_exp_on_susysvm_dataset(seed=None, n=100, distr='par', metrics_nodes='all', alert=True):
    if alert:
        print('test_exp_on_susysvm_dataset()')
        print('n={}, distr={}, metrics_nodes={}'.format(n, distr, metrics_nodes))
        input("click [ENTER] to continue or [CTRL]+[C] to abort")

    time_distr_class = {
        'exp': statistics.ExponentialDistribution, 'unif': statistics.UniformDistribution,
        'par': statistics.Type2ParetoDistribution, 'real': statistics.SparkRealTimings
    }[distr]
    time_distr_param = {'exp': [[1]], 'unif': [[0, 2]], 'par': [[3, 2]], 'real': []}[distr]
    graphs = {
        100: ['2-expander', '3-expander', '4-expander', '8-expander', '10-expander', '20-expander', '50-expander',
            '80-expander', '99-clique', ],
        400: ['2-expander', '3-expander', '4-expander', '8-expander', '20-expander', '50-expander', '100-expander',
            '200-expander', '300-expander', '399-clique', ],
        1000: ['2-expander', '3-expander', '4-expander', '8-expander', '16-expander', '20-expander', '30-expander',
            '40-expander', '50-expander', '100-expander', '200-expander', '500-expander', '999-clique', ]
    }[n]
    metrics_type = {'worst': 2, 'all': 0}[metrics_nodes]

    simulator.run(
        seed=seed,
        n=n,
        graphs=graphs,
        n_samples=500000,
        dataset='susysvm',
        starting_weights_domain=[-0.5, 2],
        max_iter=2000,
        max_time=None,
        alpha=5e-2,
        learning_rate='constant',
        spectrum_dependent_learning_rate=False,
        time_distr_class=time_distr_class,
        time_distr_param=time_distr_param,
        obj_function='hinge_loss',
        metrics=[],
        metrics_type=metrics_type,
        metrics_nodes=metrics_nodes,
        shuffle=True,
        save_test_to_file=True,
        test_folder_name_struct=[
            'fixedseed_real_svm',
            # 'dataset',
            # 'w_domain',
            'nodes',
            'distr',
            'metrics',
            'alpha',
            'samp',
            # 'feat',
            'time',
            'iter'
        ],
        test_parent_folder="",
        instant_plot=False,
        plots=['hinge_loss_iter', 'hinge_loss_time'],
        save_plot_to_file=False,
        plot_global_w=False,
        plot_node_w=False
    )


def test_exp_on_unisvm2_dataset(seed=None, n=100, distr='par', metrics_nodes='all', alert=True):
    if alert:
        print('test_exp_on_unisvm2_dataset()')
        print('n={}, distr={}, metrics_nodes={}'.format(n, distr, metrics_nodes))
        input("click [ENTER] to continue or [CTRL]+[C] to abort")

    time_distr_class = {
        'exp': statistics.ExponentialDistribution, 'unif': statistics.UniformDistribution,
        'par': statistics.Type2ParetoDistribution
    }[distr]
    time_distr_param = {'exp': [[1]], 'unif': [[0, 2]], 'par': [[3, 2]], }[distr]
    graphs = {
        100: ['2-expander', '3-expander', '4-expander', '8-expander', '10-expander', '20-expander', '50-expander',
            '80-expander', '99-clique', ],
        400: ['2-expander', '3-expander', '4-expander', '8-expander', '20-expander', '50-expander', '100-expander',
            '200-expander', '300-expander', '399-clique', ],
        1000: ['2-expander', '3-expander', '4-expander', '8-expander', '20-expander', '30-expander', '40-expander',
            '50-expander', '100-expander', '200-expander', '500-expander', '999-clique', ]
    }[n]
    metrics_type = {'worst': 2, 'all': 0}[metrics_nodes]

    simulator.run(
        seed=seed,
        n=n,
        graphs=graphs,
        dataset='unisvm2',
        starting_weights_domain=[1, 1],
        max_iter=1000,
        max_time=None,
        alpha=0.05,
        learning_rate='constant',
        spectrum_dependent_learning_rate=True,
        time_distr_class=time_distr_class,
        time_distr_param=time_distr_param,
        obj_function='hinge_loss',
        metrics=[],
        metrics_type=metrics_type,
        metrics_nodes=metrics_nodes,
        shuffle=True,
        save_test_to_file=True,
        test_folder_name_struct=[
            'us2000',
            'dataset',
            'alpha',
            'nodes',
            'shuffle',
            'distr',
            'metrics',
            'time',
            'iter'
        ],
        test_parent_folder="",
        instant_plot=False,
        plots=['hinge_loss_iter', 'hinge_loss_time'],
        save_plot_to_file=True,
        plot_global_w=False,
        plot_node_w=False
    )


def test_different_nodes_speed(seed=None, n=100, max_iter=1000, distr='exp', alert=True):
    if alert:
        print('test_different_nodes_timing()')
        print('seed={}, n={}, distr={}'.format(seed, n, distr))
        input("click [ENTER] to continue or [CTRL]+[C] to abort")

    time_distr_class = {
        'exp': statistics.ExponentialDistribution, 'unif': statistics.UniformDistribution,
        'par': statistics.Type2ParetoDistribution
    }[distr]
    time_distr_param = {'exp': [[1]], 'unif': [[0, 2]], 'par': [[3, 2]], }[distr]

    """100: ['2-expander', '3-expander', '4-expander', '8-expander', '10-expander', '20-expander', '50-expander',
        '80-expander', '99-clique', ],
    400: ['2-expander', '3-expander', '4-expander', '8-expander', '20-expander', '50-expander', '100-expander',
        '200-expander', '300-expander', '399-clique', ],
    1000: ['2-expander', '3-expander', '4-expander', '8-expander', '20-expander', '30-expander', '40-expander',
        '50-expander', '100-expander', '200-expander', '500-expander', '999-clique', ]
    """
    graphs = {
        # 10: ['2-cycle', '3-cycle', '5-cycle'],
        100: [
            '4-expander',
            '10-expander',
            '50-expander'
        ],
        300: [
            '5-expander',
            '17-expander',
            '150-expander'
        ],
        1000: [
            '6-expander',
            '32-expander',
            '500-expander'
        ]
    }[n]

    simulator.run(
        seed=seed,
        n=n,
        graphs=graphs,
        n_samples=1000,
        dataset='unireg',
        starting_weights_domain=[0, 0],
        max_iter=max_iter,
        max_time=None,
        method=None,
        alpha=0,
        learning_rate='constant',
        time_distr_class=time_distr_class,
        time_distr_param=time_distr_param,
        time_distr_param_rule=None,
        time_const_weight=0,
        obj_function='mse',
        real_metrics_toggle=False,
        save_test_to_file=True,
        test_folder_name_struct=[
            'speed_003',
            # 'shuffle',
            # 'w_domain',
            # 'metrics',
            # 'dataset',
            'nodes',
            'distr',
            # 'distr_rule',
            # 'error',
            # 'nodeserror',
            # 'alpha',
            # 'samp',
            # 'feat',
            'time',
            'iter',
            'c',
            # 'method',
        ],
        test_parent_folder="",
        instant_plot=False,
        plots=[],
        save_plot_to_file=False
    )


def test_classic_gd():
    input("sim_spectral_ratios()... click [ENTER] to continue or [CTRL]+[C] to abort")
    simulator.run(
        seed=22052010,
        n=100,
        graphs=[
            # "0-diagonal",
            # "1-cycle",
            # "2-uniform_edges",
            # "2-cycle",
            '2-expander',
            # "3-uniform_edges",
            # "3-cycle",
            # '3-expander',
            # "4-uniform_edges",
            # "4-cycle",
            '4-expander',
            # "5-uniform_edges",
            # "5-cycle",
            # '5-expander',
            # "8-uniform_edges",
            # "8-cycle",
            '8-expander',
            # "10-uniform_edges",
            # "10-cycle",
            # '10-expander',
            # "20-uniform_edges",
            # "20-cycle",
            # '18-expander',
            '20-expander',
            # '30-expander',
            # '40-expander',
            # "50-uniform_edges",
            # "50-cycle",
            # '50-expander',
            # "80-uniform_edges",
            # "80-cycle",
            # '80-expander',
            # '100-expander',
            # '200-expander',
            # '300-expander',
            # '500-expander',
            # '80-expander',
            "99-clique",
        ],
        n_samples=1000,
        n_features=100,
        dataset='reg2',
        error_std_dev=1.0,
        node_error_std_dev=0.0,
        starting_weights_domain=[-10, 50],
        max_iter=500,
        max_time=None,
        method='classic',
        alpha=1e-3,
        learning_rate='constant',
        time_distr_class=statistics.Type2ParetoDistribution,
        time_distr_param=[[3, 2]],
        time_distr_param_rule=None,
        time_const_weight=0,
        obj_function='mse',
        spectrum_dependent_learning_rate=True,
        metrics=[],
        metrics_type=2,
        metrics_nodes='worst',
        real_metrics_toggle=False,
        shuffle=False,
        save_test_to_file=True,
        test_folder_name_struct=[
            'n006',
            'dataset',
            'w_domain',
            'alpha',
            'shuffle',
            'metrics',
            'distr',
            'error',
            'nodeserror',
            'nodes',
            'samp',
            'feat',
            'time',
            'iter',
            'c',
            'method',
        ],
        test_parent_folder="",
        instant_plot=True,
        plots=('mse_iter', 'mse_time'),
        save_plot_to_file=True,
        plot_global_w=True,
        plot_node_w=False
    )


def test_unisvm2_dataset(distr):
    time_distr_class = [
        statistics.ExponentialDistribution,
        statistics.UniformDistribution,
        statistics.Type2ParetoDistribution
    ][distr]
    time_distr_param = [
        [[1]],
        [[0, 2]],
        [[3, 2]],
    ][distr]

    input("test_unisvm_dataset()... click [ENTER] to continue or [CTRL]+[C] to abort")
    simulator.run(
        seed=22052010,
        n=100,
        graphs=[
            # "0-diagonal",
            # "1-cycle",
            # "2-uniform_edges",
            # "2-cycle",
            '2-expander',
            # "3-uniform_edges",
            # "3-cycle",
            '3-expander',
            # "4-uniform_edges",
            # "4-cycle",
            '4-expander',
            # "5-uniform_edges",
            # "5-cycle",
            # '5-expander',
            # "8-uniform_edges",
            # "8-cycle",
            '8-expander',
            # "10-uniform_edges",
            # "10-cycle",
            '10-expander',
            # "20-uniform_edges",
            # "20-cycle",
            # '18-expander',
            '20-expander',
            # '30-expander',
            # '40-expander',
            # "50-uniform_edges",
            # "50-cycle",
            '50-expander',
            # "80-uniform_edges",
            # "80-cycle",
            # '80-expander',
            # '100-expander',
            # '200-expander',
            # '300-expander',
            # '500-expander',
            # '80-expander',
            "99-clique",
        ],
        dataset='unisvm2',
        smv_label_flip_prob=0.05,
        starting_weights_domain=[1, 1],
        max_time=None,
        max_iter=1000,
        alpha=0.05,
        learning_rate='constant',
        time_distr_class=time_distr_class,
        time_distr_param=time_distr_param,
        time_distr_param_rule=None,
        time_const_weight=0,
        obj_function='cont_hinge_loss',
        spectrum_dependent_learning_rate=False,
        metrics=[],
        metrics_type=2,
        metrics_nodes='worst',
        real_metrics_toggle=False,
        method='classic',
        shuffle=True,
        save_test_to_file=True,
        test_folder_name_struct=[
            'us2003',
            'dataset',
            # 'w_domain',
            'alpha',
            'shuffle',
            'metrics',
            'distr',
            # 'error',
            # 'nodeserror',
            'nodes',
            # 'samp',
            # 'feat',
            'time',
            'iter',
            # 'c',
            # 'method',
        ],
        test_parent_folder="",
        instant_plot=False,
        plots=('cont_hinge_loss_iter', 'hinge_loss_time'),
        save_plot_to_file=True,
        plot_global_w=False,
        plot_node_w=False
    )


def test_unisvm_dataset(distr):
    time_distr_class = [
        statistics.ExponentialDistribution,
        statistics.UniformDistribution,
        statistics.Type2ParetoDistribution
    ][distr]
    time_distr_param = [
        [[1]],
        [[0, 2]],
        [[3, 2]],
    ][distr]

    input("test_unisvm_dataset()... click [ENTER] to continue or [CTRL]+[C] to abort")
    simulator.run(
        seed=22052010,
        n=100,
        graphs=[
            # "0-diagonal",
            # "1-cycle",
            # "2-uniform_edges",
            # "2-cycle",
            '2-expander',
            # "3-uniform_edges",
            # "3-cycle",
            '3-expander',
            # "4-uniform_edges",
            # "4-cycle",
            '4-expander',
            # "5-uniform_edges",
            # "5-cycle",
            # '5-expander',
            # "8-uniform_edges",
            # "8-cycle",
            '8-expander',
            # "10-uniform_edges",
            # "10-cycle",
            '10-expander',
            # "20-uniform_edges",
            # "20-cycle",
            # '18-expander',
            '20-expander',
            # '30-expander',
            # '40-expander',
            # "50-uniform_edges",
            # "50-cycle",
            '50-expander',
            # "80-uniform_edges",
            # "80-cycle",
            '80-expander',
            # '100-expander',
            # '200-expander',
            # '300-expander',
            # '500-expander',
            # '80-expander',
            "99-clique",
        ],
        dataset='unisvm',
        starting_weights_domain=[1, 1],
        max_time=None,
        max_iter=800,
        alpha=0.1,  # * math.sqrt(2),
        learning_rate='constant',
        time_distr_class=time_distr_class,
        time_distr_param=time_distr_param,
        time_distr_param_rule=None,
        time_const_weight=0,
        obj_function='cont_hinge_loss',
        spectrum_dependent_learning_rate=False,
        metrics=[],
        metrics_type=2,
        metrics_nodes='worst',
        epsilon=-math.inf,
        real_metrics_toggle=False,
        method='classic',
        shuffle=False,
        save_test_to_file=True,
        test_folder_name_struct=[
            'us006',
            'dataset',
            # 'w_domain',
            'alpha',
            'shuffle',
            'metrics',
            'distr',
            # 'error',
            # 'nodeserror',
            'nodes',
            # 'samp',
            # 'feat',
            'time',
            'iter',
            # 'c',
            # 'method',
        ],
        test_parent_folder="",
        instant_plot=True,
        plots=('cont_hinge_loss_iter', 'cont_hinge_loss_time'),
        save_plot_to_file=True,
        plot_global_w=True,
        plot_node_w=False
    )


def test_over_c():
    for s in range(16):
        for c in [1, 0.999, 0.997, 0.995, 0.992, 0.985, 0.950, 0.9, 0.75, 0.5, 0.3, 0.1, 0]:
            try:
                simulator.run(
                    seed=s * 77,
                    n=100,
                    graphs=[
                        # "0-diagonal",
                        "1-cycle",
                        "2-uniform_edges",
                        "2-cycle",
                        "3-uniform_edges",
                        "3-cycle",
                        "4-uniform_edges",
                        "4-cycle",
                        # "5-uniform_edges",
                        # "5-cycle",
                        "8-uniform_edges",
                        "8-cycle",
                        # "10-uniform_edges",
                        # "10-cycle",
                        "20-uniform_edges",
                        "20-cycle",
                        "50-uniform_edges",
                        "50-cycle",
                        "80-uniform_edges",
                        "80-cycle",
                        "99-clique"
                    ],
                    n_samples=1000,
                    n_features=100,
                    dataset='reg',
                    error_std_dev=1.0,
                    starting_weights_domain=[-2.5, 10.60],
                    max_iter=None,
                    max_time=100,
                    method='classic',
                    alpha=1e-4,
                    learning_rate='root-decreasing',
                    time_distr_class=statistics.ExponentialDistribution,
                    time_distr_param=(1,),
                    time_const_weight=c,
                    save_test_to_file=True,
                    test_folder_name_struct=(
                        'y03',
                        'w_domain',
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
                        'c'
                    ),
                    test_parent_folder="y03",
                    instant_plot=False,
                    save_plot_to_file=False,
                    plot_global_w=False,
                    plot_node_w=False
                )
            except:
                continue


def test_spectral_ratios():
    input("test_spectral_ratios()... click [ENTER] to continue or [CTRL]+[C] to abort")
    simulator.run(
        seed=1531934976,
        n=100,
        graphs=[
            "0-diagonal",
            "1-cycle",
            # "2-uniform_edges",
            "2-cycle",
            '2-expander',
            # "3-uniform_edges",
            # "3-cycle",
            # '3-expander',
            # "4-uniform_edges",
            "4-cycle",
            '4-expander',
            # "5-uniform_edges",
            # "5-cycle",
            # '5-expander',
            # "8-uniform_edges",
            "8-cycle",
            '8-expander',
            # "10-uniform_edges",
            # "10-cycle",
            # '10-expander',
            # "20-uniform_edges",
            "20-cycle",
            '20-expander',
            # "50-uniform_edges",
            "50-cycle",
            '50-expander',
            # "80-uniform_edges",
            # "80-cycle",
            # '80-expander',
            "99-clique",
        ],
        n_samples=100,
        dataset='unireg',
        starting_weights_domain=[-70, -50],
        max_iter=100,
        alpha=0.01,
        learning_rate='constant',
        time_distr_class=statistics.ExponentialDistribution,
        time_distr_param=[[1]],
        time_distr_param_rule=None,
        time_const_weight=0,
        spectrum_dependent_learning_rate=False,
        metrics_type=0,
        metrics_nodes='all',
        method='classic',
        shuffle=False,
        save_test_to_file=True,
        test_folder_name_struct=[
            'ux004',
            'w_domain',
            'alpha',
            'shuffle',
            'dataset',
            'metrics',
            'distr',
            # 'error',
            # 'nodeserror',
            'nodes',
            # 'samp',
            # 'feat',
            # 'time',
            'iter',
            # 'c',
            # 'method',
        ],
        test_parent_folder="",
        instant_plot=True,
        plots=('mse_iter',),
        save_plot_to_file=True,
        plot_global_w=True,
        plot_node_w=[0, 5, 10, 20, 50, 55, 60, 70]
    )


def test_different_100nodes_timing_loop(index):
    S = [
        # [1, None],
        # [2, None],
        # [10, None],
        # [2, 'alternate'],
        # [10, 'alternate'],
        # [2, 'split'],
        # [10, 'split'],
        [3, 'alternate'],
        [4, 'alternate'],
        [8, 'alternate'],
        [15, 'alternate'],
        [20, 'alternate'],
        [3, 'split'],
        [4, 'split'],
        [8, 'split'],
        [15, 'split'],
        [20, 'split'],
    ]
    for mu_slow, rule in S:
        time_distr_class = [
            statistics.ExponentialDistribution,
            statistics.UniformDistribution,
            statistics.Type2ParetoDistribution
        ][index]
        time_distr_param = [
            [[1 / mu_slow], [1]],
            [[0, 2 * mu_slow], [0, 2]],
            [[3, 2 * mu_slow], [3, 2]],
        ][index]
        simulator.run(
            seed=18062018,
            n=100,
            graphs=[
                "0-diagonal",
                "1-cycle",
                "2-cycle",
                # '2-expander',
                "3-cycle",
                # '3-expander',
                "4-cycle",
                # '4-expander',
                "8-cycle",
                # '8-expander',
                "10-cycle",
                # '10-expander',
                "20-cycle",
                # '20-expander',
                "50-cycle",
                # '50-expander',
                "99-clique",
            ],
            n_samples=1000,
            dataset='unireg',
            starting_weights_domain=[0, 0],
            max_iter=None,
            max_time=10000,
            method=None,
            alpha=0,
            learning_rate='constant',
            time_distr_class=time_distr_class,
            time_distr_param=time_distr_param,
            time_distr_param_rule=rule,
            time_const_weight=0,
            obj_function='mse',
            real_metrics_toggle=False,
            save_test_to_file=True,
            test_folder_name_struct=[
                'hxx01_hetertime',
                # 'shuffle',
                # 'w_domain',
                # 'metrics',
                # 'dataset',
                'distr',
                'distr_rule',
                # 'error',
                # 'nodeserror',
                # 'alpha',
                'nodes',
                # 'samp',
                # 'feat',
                'time',
                'iter',
                'c',
                # 'method',
            ],
            test_parent_folder="",
            instant_plot=False,
            plots=[
                'iter_time',
                'avg_iter_time'
            ],
            save_plot_to_file=True
        )


def test_different_1000nodes_timing_loop(index):
    S = [
        [1, None],
        [2, None],
        [10, None],
        [2, 'alternate'],
        [10, 'alternate'],
        [2, 'split'],
        [10, 'split'],
    ]
    for mu_slow, rule in S:
        time_distr_class = [
            statistics.ExponentialDistribution,
            statistics.UniformDistribution,
            statistics.Type2ParetoDistribution
        ][index]
        time_distr_param = [
            [[1 / mu_slow], [1]],
            [[0, 2 * mu_slow], [0, 2]],
            [[3, 2 * mu_slow], [3, 2]],
        ][index]
        simulator.run(
            seed=18062018,
            n=1000,
            graphs=[
                "0-diagonal",
                "1-cycle",
                "2-cycle",
                '2-expander',
                "4-cycle",
                '4-expander',
                "20-cycle",
                '20-expander',
                "50-cycle",
                '50-expander',
                "100-cycle",
                '100-expander',
                "200-cycle",
                '200-expander',
                "500-cycle",
                '500-expander',
                "999-clique",
            ],
            n_samples=1000,
            dataset='unireg',
            starting_weights_domain=[0, 0],
            max_iter=None,
            max_time=10000,
            method=None,
            alpha=0,
            learning_rate='constant',
            time_distr_class=time_distr_class,
            time_distr_param=time_distr_param,
            time_distr_param_rule=rule,
            time_const_weight=0,
            obj_function='mse',
            real_metrics_toggle=False,
            save_test_to_file=True,
            test_folder_name_struct=[
                'hxx01_hetertime',
                # 'shuffle',
                # 'w_domain',
                # 'metrics',
                # 'dataset',
                'distr',
                'distr_rule',
                # 'error',
                # 'nodeserror',
                # 'alpha',
                'nodes',
                # 'samp',
                # 'feat',
                'time',
                'iter',
                'c',
                # 'method',
            ],
            test_parent_folder="",
            instant_plot=False,
            plots=[
                'iter_time',
                'avg_iter_time'
            ],
            save_plot_to_file=True
        )


def test_eigenvalue_computation_suite():
    for n in [1000, 500, 200, 100, 50, 10]:
        for d in [1, 2, 3, 4, 5, 8, 10, 20, 40, 50, 80, 99]:
            [v1, v2] = test_eigenvalue_computation(n, d)
            print("n={}, d={}, ({}, {}) diff={}".format(n, d, v1, v2, abs(v1 - v2)))


def test_eigenvalue_computation(N, d):
    A = graphs.generate_n_cycle_d_regular_graph_by_degree(N, d)
    NA = normalize(A, axis=1, norm='l1')
    v1 = uniform_weighted_Pn_second_eigenvalue_from_adjacency_matrix(A)
    v2 = (math.sin(math.pi * (d + 1) / N) / math.sin(math.pi / N)) / (d + 1)
    return [v1, v2]


def compute_velocities():
    test_folder_path = "./test_log/test_h002_hetertime_exp0.5_100n_1000time_INFiter_0c"
    logs, setup = load_test_logs(test_folder_path, return_setup=True)

    for graph in setup['graphs']:
        speed = statistics.single_iter_velocity_from_logs(
            logs['avg_iter_time'][graph],
            logs['avg_iter_time']['0-diagonal'],
            setup['max_time'],
            setup['n']
        )
        print("{} speed = {}".format(graph, speed))


"""
y = np.array([-1, 1, 1, 1, -1, 1])
y_hat_f = np.array([-1, -1, -1, 1, -1, 1])
X = np.array([[0, 1, 2, 3],
              [0, 1, 2, 3],
              [0, 1, 2, 3],
              [0, 1, 2, 3],
              [0, 1, 2, 3],
              [0, 1, 2, 3]])

print(mltoolbox.HingeLossFunction.f_gradient(y, y_hat_f, X))
print(mltoolbox.HingeLossFunction.f_gradient2(y, y_hat_f, X))
"""

"""
Ks = [0,1,2,3,4,8,20,50,99]
exp_str = "------------------------------\nExponential (lambda=1)\n------------------------------\n"
par_str = "------------------------------\nPareto Type 2 (alpha=3, sigma=2)\n------------------------------\n"
uni_str = "------------------------------\nUniform (a=0, b=2)\n------------------------------\n"
bounds = [
    ("memoryless lb", statistics.single_iteration_velocity_memoryless_lower_bound),
    ("residual time lb", statistics.single_iteration_velocity_residual_lifetime_lower_bound),
    ("don bound", statistics.single_iteration_velocity_don_bound),
    ("upper bound", statistics.single_iteration_velocity_upper_bound),
]

for bound in bounds:
    exp_str+="{}\n".format(bound[0])
    par_str+="{}\n".format(bound[0])
    uni_str+="{}\n".format(bound[0])
    for k in Ks:
        exp_str += "(k={}, {})\n".format(k, bound[1](
            k,
            statistics.ExponentialDistribution,
            [1]
        ))
        par_str += "(k={}, {})\n".format(k, bound[1](
            k,
            statistics.Type2ParetoDistribution,
            [3,2]
        ))
        uni_str += "(k={}, {})\n".format(k, bound[1](
            k,
            statistics.UniformDistribution,
            [0,2]
        ))
    exp_str+="\n"
    par_str+="\n"
    uni_str+="\n"

print(exp_str)
print(par_str)
print(uni_str)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run test simulations'
    )

    parser.add_argument(
        '-c', '--core',
        action='store',
        default=-1,
        required=False,
        help='Specify core test suite',
        dest='core'
    )

    args = parser.parse_args()

    run(int(args.core))
