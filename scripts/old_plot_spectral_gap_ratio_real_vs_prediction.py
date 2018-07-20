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
    test_folder_path = './test_log/test_u043_Win[-70,-60]_sgC1alpha_!shuf_unireg_mtrT0all_500iter'
    logs, setup = load_test_logs(test_folder_path, return_setup=True)
    degrees = {}
    for graph in setup['graphs']:
        degrees[graph] = degree_from_adjacency_matrix(setup['graphs'][graph])
    graph_filter = [
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
        "99-clique",
    ]

    clique_mse_log = logs['metrics']['mse']['99-clique']

    opt = 858.5
    pred_ratios = []
    real_ratios = []

    for graph, graph_mse_log in dict(logs['metrics']['mse']).items():
        if graph not in graph_filter:
            del logs['metrics']['mse'][graph]
            continue

        ratio = []
        for x in range(len(graph_mse_log)):
            ry = (graph_mse_log[x] - opt) / (clique_mse_log[x] - opt + 1)
            ratio.append(ry)
        ratio = max(ratio)

        real_ratios.append(ratio)
        pred_ratios.append(1 / math.sqrt(mtm_spectral_gap_from_adjacency_matrix(setup['graphs'][graph])))

        # y = np.max(mse_log)
        # x = np.argmax(mse_log)
        # a[graph] = y * math.sqrt(x)

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
            pred_ratios[i] - 2,
            real_ratios[i] + 0.1,
            'd={}'.format(degrees[list(logs['metrics']['mse'].keys())[i]]),
            size='xx-small'
        )

    colors = Plotter.generate_color_dict_from_graph_keys(
        list(setup['graphs'].keys()), setup['n']
    )

    # MSE - AVG ITER SUBPLOT
    plt.subplot(1, 2, 2)
    plt.title("MSE over AVG iteration", loc='left')
    plt.title("({})".format(setup['time_distr_class'].name), loc='right')
    plt.xlabel('AVG iter')
    plt.ylabel('MSE')
    plt.yscale('linear')
    for graph, graph_mse_log in dict(logs['metrics']['mse']).items():
        plt.plot(
            list(range(len(graph_mse_log))),
            graph_mse_log,
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
