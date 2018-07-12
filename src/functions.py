import math, sys, random, cmath, pickle, os, warnings
import numpy as np
from termcolor import colored as col
from src.mltoolbox.metrics import METRICS

def load_test_logs(test_folder_path, return_setup=True):
    try:
        with open("{}/.setup.pkl".format(test_folder_path), 'rb') as setup_file:
            setup = pickle.load(setup_file)
    except:
        print("No setup file to open")
        raise

    graphs = list(setup['graphs'].keys())
    degrees = {}

    for graph in graphs:
        degrees[graph] = compute_graph_degree_from_adjacency_matrix(setup['graphs'][graph])

    logs = {
        "iter_time": {},
        "avg_iter_time": {},
        "max_iter_time": {},
        "metrics": {}
    }

    # Fill metrics with instances of metrics objects
    if not (isinstance(setup["metrics"], list) or isinstance(setup["metrics"], tuple)):
        if setup["metrics"] in METRICS:
            setup["metrics"] = [setup["metrics"]]
        elif setup["metrics"].lower() == 'all':
            setup["metrics"] = list(METRICS.keys())

    if setup['real_metrics_toggle']:
        if not (isinstance(setup["real_metrics"], list) or isinstance(setup["real_metrics"], tuple)):
            if setup["real_metrics"] in METRICS:
                setup["real_metrics"] = [setup["real_metrics"]]
            elif setup["real_metrics"].lower() == 'all':
                setup["real_metrics"] = list(METRICS.keys())

    if not setup['obj_function'] in setup['metrics']:
        setup['metrics'].insert(0, setup['obj_function'])

    if setup['real_metrics_toggle'] and not setup['obj_function'] in setup['real_metrics']:
        setup['real_metrics'].insert(0, setup['obj_function'])

    for m in setup["metrics"]:
        if m in METRICS:
            logs["metrics"][m] = {}
    if setup['real_metrics_toggle']:
        for rm in setup["real_metrics"]:
            if rm in METRICS:
                logs["metrics"]["real_" + rm] = {}

    # it's important to loop on a copy of graphs and not on the original one
    # since the original in modified inside the loop
    for graph in graphs[:]:
        iter_log_path = "{}/{}_iter_time_log".format(test_folder_path, graph)
        avg_iter_log_path = "{}/{}_avg_iter_time_log".format(test_folder_path, graph)
        max_iter_log_path = "{}/{}_max_iter_time_log".format(test_folder_path, graph)

        ext = ''
        if not os.path.isfile(iter_log_path):
            if os.path.isfile(iter_log_path + '.txt'):
                ext = '.txt'
            elif os.path.isfile(iter_log_path + '.gz'):
                ext = '.gz'
            elif os.path.isfile(iter_log_path + '.txt.gz'):
                ext = '.txt.gz'
            else:
                raise Exception('File not found in {}'.format(test_folder_path))

        iter_log_path += ext
        avg_iter_log_path += ext
        max_iter_log_path += ext

        try:
            logs["iter_time"][graph] = np.loadtxt(iter_log_path)
            logs["avg_iter_time"][graph] = [tuple(s.split(",")) for s in np.loadtxt(avg_iter_log_path, str)]
            logs["max_iter_time"][graph] = [tuple(s.split(",")) for s in np.loadtxt(max_iter_log_path, str)]
        except OSError:
            warnings.warn('Graph "{}" not found in folder {}'.format(graph, test_folder_path))
            graphs.remove(graph)
            continue

        for metrics_log in logs["metrics"]:
            metrics_log_path = "{}/{}_{}_log".format(test_folder_path, graph, metrics_log)
            metrics_log_path += ext

            try:
                logs["metrics"][metrics_log][graph] = np.loadtxt(metrics_log_path)
            except OSError:
                warnings.warn('Graph "{}" not found in folder {}'.format(graph, test_folder_path))
                graphs.remove(graph)
                continue

    if return_setup:
        return logs, setup
    return logs


def iteration_speed_lower_bound_new(distr_name, param):
    pass


def iteration_speed_lower_bound(l, k, time_arr):
    """
    Compute almost-lower bound.

    Parameters
    ----------
    l : float
        Lambda (exponential rate).
    k : int
        Graph's degree.
    time_arr : list of float
        #iter -> completion time fo such iteration.

    Returns
    -------
    list of float
    """

    lb = []
    for t in time_arr:
        lb.append(t * l / (1 + sum(1 / i for i in range(1, k + 1))))
    return lb


def compute_graph_degree_from_adjacency_matrix(adj_mat):
    degree = 0

    for i in range(len(adj_mat)):
        degree = max(degree, np.sum(adj_mat[i]) - 1)

    return int(degree)


def compute_spectral_gap_from_adjacency_matrix(adj_mat):
    return 1 - compute_second_eigenvalue_from_adjacency_matrix(adj_mat)


def compute_second_eigenvalue_from_adjacency_matrix(adj_mat):
    Ck = np.sign(adj_mat[0]).clip(min=0)
    n = len(adj_mat)
    d = np.sum(Ck)-1
    max_val = 0
    for m in range(1, n):
        s = 0.0
        for k in range(0, n):
            s += Ck[k] * cmath.exp(-1j * 2 * math.pi * k * m / n)
        s /= d + 1
        max_val = max(abs(s), max_val)
    return abs(max_val)


def progress(current_progress, total_progress, bar_length=50, text_before='', text_after=''):
    filled_len = int(round(bar_length * current_progress / float(total_progress)))
    percents = round(100.0 * current_progress / float(total_progress), 1)
    bar = '#' * filled_len + '-' * (bar_length - filled_len)

    if text_before != '':
        text_before += ' '

    sys.stdout.write('{}[{}] {}% {}\r'.format(text_before, bar, percents, text_after))
    sys.stdout.flush()


def print_verbose(level, msg, no_input=False):
    if level == 0:
        return
    elif level == 1 or (level == 2 and no_input):
        print(msg)
    elif level == 2:
        input(str(msg) + col(" [PRESS ENTER]", 'red'))
