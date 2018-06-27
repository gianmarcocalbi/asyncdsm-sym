from src import cluster, functions, graph_generator, node, plotter, statistics, tasks
from src.mltoolbox import *
import numpy as np
from importlib import reload
import math, os, pickle, glob
import matplotlib.pyplot as plt

def main():
    global points, time_distr_name

    it = 8
    test_folders = [

    ]

    """
    points = {
        {GRAPH_NAME} : {
            {NODES_AMOUNT} : {
                {x_value} : {MSE_GRAPH/MSE_CLIQUE}]
            }
        }
    }
    """

    points = {}
    curves = {}
    time_distr_name = ''
    test_folder_paths = list(glob.iglob("./test_log/test_012*"))


    def extract_points_from_folder(test_folder_path):
        global points, time_distr_name
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
        N = setup['n']
        n_samples = setup['n_samples']
        n_features = setup['n_features']
        x = N #1 / (N * n_features)

        for graph in setup['graphs']:
            if not graph in points:
                points[graph] = {}
            if not N in points[graph]:
                points[graph][N] = {}
            if not x in points[graph][N]:
                points[graph][N][x] = []
            graph_mse_log = np.loadtxt(os.path.normpath(os.path.join(test_folder_path, graph + '_mse_log.gz')))
            points[graph][N][x].append(graph_mse_log[it])


    for test_folder_path in test_folder_paths:
        extract_points_from_folder(test_folder_path)

    for graph in points:
        for n in points[graph]:
            for x in points[graph][n]:
                points[graph][n][x] = np.average(points[graph][n][x])

    for graph in points:
        if graph not in curves:
            curves[graph] = []
        for n in points[graph]:
            for x in points[graph][n]:
                curves[graph].append((x, points[graph][n][x] / points['n-1_clique'][n][x]))


    filename = "1_avg_iter_time"
    plt.title("MSE comparison at iteration {}".format(it), loc='left')
    plt.title("({})".format(time_distr_name), loc='right')
    plt.xlabel("#nodes")
    plt.ylabel("MSE graph / MSE clique")
    plt.yscale('linear')

    for graph in curves:
        curves[graph].sort(key=lambda _: _[0])
        lx = [float(p[0]) for p in curves[graph]]
        ly = [float(p[1]) for p in curves[graph]]

        plt.plot(
            lx,
            ly,
            label=graph,
            color=plotter.Plotter.get_graph_color(graph),
            #**kwargs
        )


    plt.legend()
    plt.show()
    plt.close()