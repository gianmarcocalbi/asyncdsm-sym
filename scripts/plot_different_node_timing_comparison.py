import glob
import matplotlib.pyplot as plt
from src import statistics
from src.utils import *


def main():
    # SETUP BEGIN

    it = 8
    test_root = './test_log'
    test_num = 'x04'  # 012, 013, 014, 015
    test_subroot = '/bulk/{}'.format(test_num)
    dataset = ''
    n_samples = ''
    n_features = '100'  # 10, 50, 100
    graph = ['4-cycle']

    # SETUP END

    test_folder_paths_pattern = os.path.normpath("{}{}/test_{}*{}*{}samp*{}feat*".format(
        test_root,
        test_subroot,
        test_num,
        dataset,
        n_samples,
        n_features
    ))

    test_folder_paths = list(glob.iglob(test_folder_paths_pattern))
    time_distr_class_name = None

    points = []
    for test_folder_path in test_folder_paths:
        logs, setup = load_test_logs(test_folder_path, return_setup=True)

        if time_distr_class_name is None:
            time_distr_class_name = setup['time_distr_class'].name

        if time_distr_class_name != setup['time_distr_class'].name:
            raise Exception('Logs come from simulations using different time_distr_class ({} and {})'.format(
                time_distr_class_name,
                setup['time_distr_class'].name
            ))

        p = []
        mu_slow = setup['time_distr_class'].mean(setup['time_distr_param'][0])
        mu_fast = setup['time_distr_class'].mean(setup['time_distr_param'][-1])
        p[0] = mu_slow / mu_fast
        p[1] = statistics.single_iter_velocity_from_logs(
            logs['avg_iter_time'][graph],
            logs['avg_iter_time']['0-diagonal'],
            setup['max_time'],
            setup['n']
        )
        points.append(p)

    plt.suptitle(test_folder_paths_pattern)
    plt.title("Iter speed comparison", loc='left')
    plt.title("({})".format(time_distr_class_name), loc='right')
    plt.xlabel("E[X]_slow / E[X]_fast")
    plt.ylabel("Iteration velocity")
    plt.yscale('linear')
    plt.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        color='green',
        markersize=5,
        marker='o',
        label=graph
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
