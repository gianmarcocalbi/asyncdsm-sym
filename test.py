from src import statistics, mltoolbox
import numpy as np
from src.plotter import plot_from_files
import main, time
from scripts import plot_topologies_obj_func_over_samples_amount_comparison as plot_ofvsclique_over_n
from scripts import plot_topologies_obj_func_at_time_over_c_comparison as plot_of_over_c

"""
for _ in range(4):
    #seed = int(time.time())
    seed = [1530197457, 1530198096, 1530198755, 1530199411][_]
    for c in [0.999, 0.997, 0.995, 0.992, 0.985]:
        try:
            main.main0(seed=seed, time_const_weight=c)
        except:
            continue
"""

# """
n_samples_list = [100, 200, 500, 800, 1000, 2000, 5000, 8000, 11000, 50000, 100000]
for n_features in [100, 50, 10]:
    for n_samples in n_samples_list:
        for _ in range(8):
            try:
                main.main0(n_samples=n_samples, n_features=n_features)
            except:
                continue


# """

# plot_of_over_c.main()
# plot_ofvsclique_over_n.main()

# plot_from_files(plots=['real_mse_iter', 'mse_iter'])


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
