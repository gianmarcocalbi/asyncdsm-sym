import glob, os, pickle, datetime, time, re, pprint
import matplotlib.pyplot as plt
import numpy as np
from src import plotter, graphs
from src.mltoolbox.metrics import METRICS
from src.utils import *
from shutil import copyfile, rmtree
from src.plotter import Plotter


def main():
    # SETUP BEGIN
    test_folder_path = './test_log/test_rslo100_sloreg_100n_exp[1]_mtrT0all_sgC1e-05alpha_52000samp_INFtime_400iter'
    target_x0 = 100
    target_x = 300

    logs, setup = load_test_logs(test_folder_path, return_setup=True)
    objfunc = METRICS[setup['obj_function']]
    degrees = {}
    for graph in setup['graphs']:
        degrees[graph] = degree_from_adjacency_matrix(setup['graphs'][graph])
    graph_filter = [
        '*'
    ]

    pred_ratios = []
    real_slopes = []

    # equal to None so that if it will not be reassigned then it will raise exception
    clique_slope = None
    clique_spectral_gap = None

    for graph, graph_objfunc_log in dict(logs['metrics'][objfunc.id]).items():
        if graph not in graph_filter and '*' not in graph_filter:
            del logs['metrics'][objfunc.id][graph]
            continue

        y0 = logs['metrics'][objfunc.id][graph][target_x0]
        y = logs['metrics'][objfunc.id][graph][target_x]
        """if degrees[graph] == 2:
            y0 = logs['metrics'][objfunc.id][graph][3000]
            y = logs['metrics'][objfunc.id][graph][3600]"""

        slope = y - y0
        print(slope)

        real_slopes.append(slope)
        pred_ratios.append(1 / math.sqrt(uniform_weighted_Pn_spectral_gap_from_adjacency_matrix(setup['graphs'][graph])))

        if 'clique' in graph:
            clique_slope = slope
            clique_spectral_gap = uniform_weighted_Pn_spectral_gap_from_adjacency_matrix(setup['graphs'][graph])

    real_ratios = clique_slope / np.array(real_slopes)
    pred_ratios = np.array(pred_ratios)
    print(real_ratios)

    plt.figure(1, figsize=(12, 6))
    plt.suptitle(test_folder_path)
    plt.subplot(1, 2, 1)
    plt.title("Ratio comparison", loc='left')
    plt.title("({})".format(setup['time_distr_class'].name), loc='right')
    plt.xlabel("prediction")
    plt.ylabel("simulation")
    plt.yscale('linear')
    plt.plot(
        pred_ratios,
        real_ratios,
        color='blue',
        markersize=5,
        marker='o',
    )

    for i in range(len(pred_ratios)):
        plt.text(
            pred_ratios[i],
            real_ratios[i],
            'd={}'.format(degrees[list(logs['metrics'][objfunc.id].keys())[i]]),
            size='xx-small'
        )

    colors = Plotter.generate_rainbow_color_dict_from_graph_keys(
        list(setup['graphs'].keys()), setup['n']
    )

    # objfunc - AVG ITER SUBPLOT
    plt.subplot(1, 2, 2)
    plt.title("{} over iteration".format(objfunc.fullname), loc='left')
    plt.title("({})".format(setup['time_distr_class'].name), loc='right')
    plt.xlabel('iter')
    plt.ylabel(objfunc.fullname)
    plt.yscale('linear')
    for graph, graph_objfunc_log in dict(logs['metrics'][objfunc.id]).items():
        plt.plot(
            list(range(len(graph_objfunc_log))),
            graph_objfunc_log,
            label=graph,
            color=colors[graph]
        )
    plt.legend()
    plt.subplots_adjust(
        top=0.88,
        bottom=0.08,
        left=0.06,
        right=0.96,
        hspace=0.2,
        wspace=0.17
    )
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()