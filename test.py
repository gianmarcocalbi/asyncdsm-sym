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
    for s in range(16):
        for c in [1, 0.999, 0.997, 0.995, 0.992, 0.985, 0.950, 0.9, 0.75, 0.5, 0.3, 0.1, 0]:
            try:
                main.main0(seed=s, time_const_weight=c)
            except:
                continue
    """

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
    plot_topologies_obj_func_at_time_over_c_comparison.main()
    # envelop.main()
    #sim_different_nodes_timing()
    # sim_spectral_ratios()
    #plot_spectral_gap_ratio_real_vs_prediction.main()


def sim_spectral_ratios():
    input("sim_spectral_ratios()... click [ENTER] to continue or [CTRL]+[C] to abort")
    main.main(
        seed=None,
        n=100,
        graphs=[
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
        ],
        n_samples=100,
        dataset='unireg',
        starting_weights_domain=[-100,-60],
        max_iter=400,
        alpha=1e-2,
        learning_rate='root-decreasing',
        metrics_type=2,
        metrics_nodes='worst',
        shuffle=False,
        save_test_to_file=True,
        test_folder_name_struct=(
            'u041',
            'shuffle',
            'w_domain',
            'metrics',
            'dataset',
            #'distr',
            #'error',
            #'nodeserror',
            'alpha',
            'nodes',
            'samp',
            #'feat',
            #'time',
            'iter',
            #'c',
            #'method',
        ),
        test_parent_folder="",
        instant_plot=True,
        plots=('mse_iter',),
        save_plot_to_file=True,
        plot_global_w=True,
        plot_node_w=False
    )


def sim_different_nodes_timing():
    input("sim_different_nodes_timing()... click [ENTER] to continue or [CTRL]+[C] to abort")
    main.main(
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
        n_samples=100,
        dataset='unireg',
        starting_weights_domain=[-70, -60],
        max_iter=None,
        max_time=1000,
        method='classic',
        alpha=1e-3,
        learning_rate='constant',
        time_distr_class=statistics.UniformDistribution,
        time_distr_param=[[0, 2], [1, 3]],
        time_distr_param_shuffle=False,
        time_const_weight=0,
        obj_function='mse',
        real_metrics_toggle=False,
        save_test_to_file=True,
        test_folder_name_struct=(
            'h001_hetertime',
                # 'shuffle',
                # 'w_domain',
                # 'metrics',
                # 'dataset',
            'distr',
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
        ),
        test_parent_folder="",
        instant_plot=True,
        plots=('iter_time',),
        save_plot_to_file=True,
        plot_global_w=False,
        plot_node_w=False,
        verbose_main=0,
        verbose_cluster=0,
        verbose_node=0,
        verbose_task=0,
        verbose_plotter=0
    )


def test_eigenvalue_computation_suite():
    for n in [1000, 500, 200, 100, 50, 10]:
        for d in [1, 2, 3, 4, 5, 8, 10, 20, 40, 50, 80, 99]:
            [v1, v2] = test_eigenvalue_computation(n, d)
            print("n={}, d={}, ({}, {}) diff={}".format(n, d, v1, v2, abs(v1 - v2)))


def test_eigenvalue_computation(N, d):
    A = graphs.generate_n_cycle_d_regular_graph_by_degree(N, d)
    NA = normalize(A, axis=1, norm='l1')
    v1 = compute_second_eigenvalue_from_adjacency_matrix(A)
    v2 = (math.sin(math.pi * (d + 1) / N) / math.sin(math.pi / N)) / (d + 1)
    return [v1, v2]


def compute_velocities():
    graphs = (
        ("0_diagonal", 0),
        ("1_cycle", 1),
        ("2_diam-expander", 2),
        ("2_root-expander", 2),
        ("3_regular", 3),
        ("4_regular", 4),
        ("5_regular", 5),
        ("6_regular", 6),
        ("7_regular", 7),
        ("8_regular", 8),
        ("9_regular", 9),
        ("20_regular", 20),
        ("50_regular", 50),
        ("n-1_clique", 99),
        ("n-1_star", 99),
    )

    for graph, degree in graphs:
        try:
            print(
                "{} speed = {}".format(graph, statistics.single_iteration_velocity_as_tot_iters_over_avg_diagonal_iter(
                    "./test_log/test_010_pareto3-2_0.5c_10ktime1e-4alpha_lowDegreeComparison_classic.conflict.0", graph,
                    100,
                    10000)))
        except IOError:
            pass
        except:
            raise


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
