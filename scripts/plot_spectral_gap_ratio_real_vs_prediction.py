import glob, os, pickle, datetime, time, re, pprint
import matplotlib.pyplot as plt
import numpy as np
from src import plotter, graphs
from src.mltoolbox.metrics import METRICS
from src.functions import *
from shutil import copyfile, rmtree


def main():
    # SETUP BEGIN
    test_folder_path = './test_log/temp/test_u041_!shuf_Win[-50,-40]_mtrT2worst_unireg_r0.01alpha_100n_100samp_200iter'
    logs, setup = load_test_logs(test_folder_path, return_setup=True)
    degrees = {}
    for graph in setup['graphs']:
       degrees[graph] = compute_graph_degree_from_adjacency_matrix(setup['graphs'][graph])
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
        #"20-uniform_edges",
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
            ry = (graph_mse_log[x] - opt) / (clique_mse_log[x] - opt)
            ratio.append(ry)
        ratio = max(ratio)

        real_ratios.append(ratio)
        pred_ratios.append(1 / math.sqrt(1 - compute_second_eigenvalue_from_adjacency_matrix(
            setup['graphs'][graph]
        )))

        #y = np.max(mse_log)
        #x = np.argmax(mse_log)
        #a[graph] = y * math.sqrt(x)


    print(real_ratios)

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



    plt.legend()
    plt.show()
    plt.close()