import glob, os, pickle, datetime, time, re, pprint
import matplotlib.pyplot as plt
import numpy as np
from src import plotter, graphs
from src.mltoolbox.metrics import METRICS
from src.functions import *
from shutil import copyfile, rmtree

def main():
    # SETUP BEGIN
    test_folder_path = './test_log/test_u040_!shuf_Win[-20,-20]_mtrT2worst_unireg_exp1_0.0err_c0.01alpha_100n_INFtime_100iter_0c/'
    logs, setup = load_test_logs(test_folder_path, return_setup=True)
    rng = []
    points = [36]

    graph_filter = [
        # "0-diagonal",
        #"1-cycle",
        # "2-uniform_edges",
        #"2-cycle",
        # "3-uniform_edges",
        #"3-cycle",
        # "4-uniform_edges",
        #"4-cycle",
        # "5-uniform_edges",
        # "5-cycle",
        # "8-uniform_edges",
        #"8-cycle",
        # "10-uniform_edges",
        # "10-cycle",
        #"20-uniform_edges",
        #"20-cycle",
        # "50-uniform_edges",
        #"50-cycle",
        # "80-uniform_edges",
        #"80-cycle",
        "99-clique",
    ]

    a = {}

    for graph, mse_log in dict(logs['metrics']['mse']).items():
        if graph not in graph_filter:
            del logs['metrics']['mse'][graph]
            continue
        a[graph] = []
        if len(rng) == 2:
            x_list = range(max(0, rng[0]), min(rng[1], len(mse_log)))
        else:
            x_list = points

        for x in x_list:
            y = mse_log[x]
            a[graph].append(y * math.sqrt(x))

        a[graph] = sum(a[graph]) / len(a[graph])

    colors = plotter.Plotter.generate_color_dict_from_degrees(
        list(setup['graphs'].keys()), setup['n']
    )

    plt.title("MSE Envelops", loc='left')
    plt.title("({})".format(setup['time_distr_class'].name), loc='right')
    plt.xlabel("iter")
    plt.ylabel("mse")
    plt.yscale('linear')

    for graph, mse_log in logs['metrics']['mse'].items():
        lx = list(range(len(mse_log)))

        plt.plot(
            lx,
            mse_log,
            label=graph,
            color=colors[graph]
        )

        plt.plot(
            lx,
            [(a[graph] / math.sqrt(x)) for x in lx],
            #[(4.5 * 1271 / math.pow(x + 1, 0.5)) for x in lx],
            label=graph + ' Env',
            color=colors[graph],
            linestyle=(0, (1, 4))
        )


    plt.legend()
    plt.show()
    plt.close()
