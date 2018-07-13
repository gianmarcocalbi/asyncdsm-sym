import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from src import plotter
from src.mltoolbox.metrics import METRICS


def main():
    global points, time_distr_name, obj_func_shortname

    t = 8
    test_folders = [

    ]

    """
    points = {
        {GRAPH_NAME} : {
            {c} : [{obj_func_at_time_t_with_c}]
        }
    }
    """

    points = {}
    time_distr_name = ''
    test_folder_paths = list(glob.iglob("./test_log/bulk/y02/test_*"))
    obj_func_shortname = ""

    if len(test_folder_paths) == 0:
        raise Exception("Empty test folder paths list")

    def extract_points_from_folder(test_folder_path):
        global points, time_distr_name, obj_func_shortname
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

        c = setup['time_const_weight']

        for graph in setup['graphs']:
            if not graph in points:
                points[graph] = {}
            if not c in points[graph]:
                points[graph][c] = []
            graph_iter_time_log = np.loadtxt(
                os.path.normpath(os.path.join(test_folder_path, '{}_iter_time_log.gz'.format(graph)))
            )
            t_prev_index = max(np.searchsorted(graph_iter_time_log, t, side='left') - 1, 0)
            t_next_index = min(t_prev_index + 1, len(graph_iter_time_log) - 1)

            t_prev = graph_iter_time_log[t_prev_index]
            t_next = graph_iter_time_log[t_next_index]

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

            points[graph][c].append(
                (
                        abs(t - t_prev) * graph_obj_func_log[t_next_index]
                        + abs(t - t_next) * graph_obj_func_log[t_prev_index]
                ) / abs(t_next - t_prev)
            )

    for test_folder_path in test_folder_paths:
        extract_points_from_folder(test_folder_path)

    curves = {}
    for graph in points:
        if not graph in curves:
            curves[graph] = []
        for c in points[graph]:
            curves[graph].append((1-c, np.average(points[graph][c])))

    plt.title("{} comparison at time t={} over c".format(obj_func_shortname, t), loc='left')
    plt.title("({})".format(time_distr_name), loc='right')
    plt.xlabel("c")
    plt.ylabel("{} at time t={}".format(obj_func_shortname, t))
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
            marker='o',
            markersize=2
            # **kwargs
        )

    plt.legend()
    plt.show()
    plt.close()
