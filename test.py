from src import statistics, graphs
from src.mltoolbox import *
from src.functions import *
import numpy as np
from src.plotter import plot_from_files
import main, time, math
from scripts import *
from sklearn.preprocessing import normalize


def run():
    """
    n_samples_list = [100, 200, 300, 400, 500, 600, 800, 1000, 2000, 5000, 8000, 10100, 20200]
    for n_features in [100]:
        for n_samples in n_samples_list:
            for _ in range(20):
                try:
                    main.main0(n_samples=n_samples, n_features=n_features)
                except:
                    continue
    """

    """ TEST VELOCITY OVER DEGREES
    nodes_list = [20, 50, 100, 200, 300, 400, 500, 600, 800, 1000]
    for N in nodes_list:
        for _ in range(2):
            try:
                main.main0(
                    n=N,
                    time_distr_class=statistics.ExponentialDistribution,
                    time_distr_param=[1],
                    seed=N*_
                )
                main.main0(
                    n=N,
                    time_distr_class=statistics.UniformDistribution,
                    time_distr_param=[0,2],
                    seed=N*_
                )
                main.main0(
                    n=N,
                    time_distr_class=statistics.ExponentialDistribution,
                    time_distr_param=[3,2],
                    seed=N*_
                )
            except:
                continue
    """

    # merge_tests.main()
    # plot_of_over_c.main()
    # plot_topologies_obj_func_at_time_over_c_comparison.main()
    # envelop.main()
    # test_different_nodes_timing()
    # test_different_nodes_timing_loop()
    # test_classic_gd()
    # test_spectral_ratios()
    # plot_spectral_gap_ratio_real_vs_prediction.main()
    print_topologies_velocity.main()


def test_classic_gd():
    input("sim_spectral_ratios()... click [ENTER] to continue or [CTRL]+[C] to abort")
    main.main(
        n=100,
        graphs=[
            # "0-diagonal",
            "1-cycle",
            # "2-uniform_edges",
            "2-cycle",
            # "3-uniform_edges",
            "3-cycle",
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
            "80-cycle",
            "99-clique"
        ],
        n_samples=1000,
        n_features=100,
        dataset='reg2',
        error_std_dev=1.0,
        node_error_std_dev=0.0,
        starting_weights_domain=[20, 30],
        max_iter=500,
        max_time=None,
        method='classic',
        alpha=1e-2,
        learning_rate='constant',
        time_distr_class=statistics.ExponentialDistribution,
        time_distr_param=(1,),
        obj_function='mse',
        real_metrics_toggle=False,
        metrics_type=2,
        metrics_nodes='worst',
        shuffle=True,
        save_test_to_file=True,
        test_folder_name_struct=(
            'n001',
            'dataset',
                # 'shuffle',
                # 'metrics',
            'w_domain',
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
        instant_plot=True,
        plots=('mse_iter',),
        save_plot_to_file=False
    )


def test_over_c():
    for s in range(16):
        for c in [1, 0.999, 0.997, 0.995, 0.992, 0.985, 0.950, 0.9, 0.75, 0.5, 0.3, 0.1, 0]:
            try:
                main.main(
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
    input("sim_spectral_ratios()... click [ENTER] to continue or [CTRL]+[C] to abort")
    main.main(
        n=100,
        graphs=[
            # "0-diagonal",
            "1-cycle",
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
            # '10-expander',
            # "20-uniform_edges",
            # "20-cycle",
            '20-expander',
            # "50-uniform_edges",
            # "50-cycle",
            '50-expander',
            # "80-uniform_edges",
            # "80-cycle",
            # '80-expander',
            "99-clique",
        ],
        n_samples=100,
        dataset='unireg',
        starting_weights_domain=[-70, -60],
        max_iter=500,
        alpha=0.5,
        learning_rate='constant',
        spectrum_dependent_learning_rate=True,
        metrics_type=0,
        metrics_nodes='all',
        method='classic',
        shuffle=False,
        save_test_to_file=True,
        test_folder_name_struct=[
            'uz043',
            'w_domain',
            'alpha',
            'shuffle',
            'dataset',
            'metrics',
            # 'distr',
            # 'error',
            # 'nodeserror',
            # 'nodes',
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
        plot_node_w=False
    )


def test_different_nodes_timing_loop():
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
        main.main(
            seed=17072017,
            n=100,
            graphs=[
                "0-diagonal",
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
            ],
            n_samples=100,
            dataset='unireg',
            starting_weights_domain=[0, 0],
            max_iter=None,
            max_time=10000,
            method=None,
            alpha=0,
            learning_rate='constant',
            time_distr_class=statistics.UniformDistribution,
            time_distr_param=[[0,2*mu_slow], [0,2]],
            time_distr_param_rule=rule,
            time_const_weight=0,
            obj_function='mse',
            real_metrics_toggle=False,
            save_test_to_file=True,
            test_folder_name_struct=[
                'h005_hetertime',
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
            test_parent_folder="h005",
            instant_plot=False,
            plots=tuple(),
            save_plot_to_file=True
        )


def test_different_nodes_timing():
    input("test_different_nodes_timing()... click [ENTER] to continue or [CTRL]+[C] to abort")
    main.main(
        seed=None,
        n=100,
        graphs=[
            "0-diagonal",
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
        ],
        n_samples=100,
        dataset='unireg',
        starting_weights_domain=[0, 0],
        max_iter=None,
        max_time=500,
        method=None,
        alpha=0,
        learning_rate='constant',
        time_distr_class=statistics.ExponentialDistribution,
        time_distr_param=[[0.1], [1]],
        time_distr_param_rule='alternate',
        time_const_weight=0,
        obj_function='mse',
        real_metrics_toggle=False,
        save_test_to_file=True,
        test_folder_name_struct=[
            'h006_hetertime',
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
        plots=tuple(),
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
    v1 = second_eigenvalue_approx_from_adjacency_matrix(A)
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
    run()
