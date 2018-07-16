import glob
import matplotlib.pyplot as plt
from src import statistics
from src.functions import *


def main():
    # SETUP BEGIN

    test_folder_paths_pattern = os.path.normpath("./test_log/test_h002*")
    test_folder_paths = list(glob.iglob(test_folder_paths_pattern))
    for test_folder_path in test_folder_paths:
        logs, setup = load_test_logs(test_folder_path, return_setup=True)
        print('{} | {} | {}\n'.format(
            setup['time_distr_class'].name,
            setup['time_distr_param'][0],
            setup['time_distr_param'][-1]
        ))

        for graph in setup['graphs']:
            speed = statistics.single_iter_velocity_from_logs(
                logs['avg_iter_time'][graph],
                logs['avg_iter_time']['0-diagonal'],
                setup['max_time'],
                setup['n']
            )
            print("{} speed = {}".format(graph, speed))
        print('\n')