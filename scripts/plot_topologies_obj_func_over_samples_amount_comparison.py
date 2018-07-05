import glob
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from src import plotter
from src.mltoolbox.metrics import METRICS
from src.functions import *


def main():
    global points, time_distr_name, obj_func_shortname, graphs_filter

    # SETUP BEGIN

    """
    x01 : reg only uniform edges avg on 8 tests
    x02 : reg cycles avg on 8 tests
    x03 : reg all with avg on 16 tests
    x04 : svm
    """

    it = 8
    test_root = './test_log'
    test_num = 'x04'  # 012, 013, 014, 015
    test_subroot = '/bulk/{}'.format(test_num)
    dataset = ''
    n_samples = ''
    n_features = '100'  # 10, 50, 100
    graphs_filter = ['cycle']

    # SETUP END

    test_folder_paths = list(glob.iglob(os.path.normpath("{}{}/test_{}*{}*{}samp*{}feat*".format(
        test_root,
        test_subroot,
        test_num,
        dataset,
        n_samples,
        n_features
    ))))

    """
    points = {
        {GRAPH_NAME} : {
            {NODES_AMOUNT} : {
                {x_value} : [{MSE_GRAPH/MSE_CLIQUE}]
            }
        }
    }
    """

    points = {}
    curves = {}
    time_distr_name = ''

    obj_func_shortname = ""

    if len(test_folder_paths) == 0:
        raise Exception("Empty test folder paths list")

    def extract_points_from_folder(test_folder_path):
        global points, time_distr_name, obj_func_shortname, graphs_filter
        try:
            with open("{}/.setup.pkl".format(test_folder_path), 'rb') as setup_file:
                setup = pickle.load(setup_file)
        except:
            print("No setup file to open")
            raise
        if time_distr_name == '':
            time_distr_name = setup['time_distr_class'].name
        elif time_distr_name != setup['time_distr_class'].name:
            raise Exception("Creating plot with tests using different time_distr_name, {} != {}".format(
                time_distr_name, setup['time_distr_class'].name
            ))

        if obj_func_shortname == '':
            obj_func_shortname = METRICS[setup['obj_function']].shortname

        N = setup['n']
        n_samples = setup['n_samples']
        n_features = setup['n_features']

        if setup['dataset'] != 'svm':
            x = n_samples / (N * (n_features + 1))
        else:
            x = n_samples / (N * n_features)

        for graph in setup['graphs']:
            try:
                for g in graphs_filter:
                    if g in graph:
                        raise Exception()
            except:
                continue

            degree = compute_graph_degree_from_adjacency_matrix(setup['graphs'][graph])
            if not graph in points:
                points[graph] = {}
            if not N in points[graph]:
                points[graph][N] = {}
            if not x in points[graph][N]:
                points[graph][N][x] = []
            graph_obj_func_log = np.loadtxt(
                os.path.normpath(
                    os.path.join(
                        test_folder_path,
                        '{}_{}_log.gz'.format(
                            graph, METRICS[setup['obj_function']].id
                        )
                    )
                )
            )
            points[graph][N][x].append(graph_obj_func_log[it])

    for test_folder_path in test_folder_paths:
        extract_points_from_folder(test_folder_path)

    for graph in points:
        for n in points[graph]:
            for x in points[graph][n]:
                points[graph][n][x] = np.average(points[graph][n][x])

    clique_name = [c for c in points if 'clique' in c][0]

    colors = plotter.Plotter.generate_color_dict_from_degrees(
        list(points.keys()), n
    )

    for graph in points:
        if graph not in curves:
            curves[graph] = []
        for n in points[graph]:
            for x in points[graph][n]:
                curves[graph].append((x, points[graph][n][x] / points[clique_name][n][x]))

    plt.title("{} comparison at iteration {}".format(obj_func_shortname, it), loc='left')
    plt.title("({})".format(time_distr_name), loc='right')
    plt.xlabel("#samples")
    plt.ylabel("{0} graph / {0} clique".format(obj_func_shortname))
    plt.yscale('linear')

    for graph in curves:
        curves[graph].sort(key=lambda _: _[0])
        lx = [float(p[0]) for p in curves[graph]]
        ly = [float(p[1]) for p in curves[graph]]

        plt.plot(
            lx,
            ly,
            label=graph,
            color=colors[graph],
            marker='o',
            markersize=2
            # **kwargs
        )

    plt.legend()
    plt.show()
    plt.close()
