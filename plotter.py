import numpy as np
import matplotlib.pyplot as plt
import os, argparse


def plot_from_files(test_folder_path=None, save_to_test_folder=False):
    if test_folder_path is None:
        test_folder_path = os.path.join("test_name")
        test_folder_path = get_last_temp_test_path()

    plot_folder_path = os.path.join(test_folder_path, "plot")
    if save_to_test_folder:
        if not os.path.exists(plot_folder_path):
            os.makedirs(plot_folder_path)

    avg = None
    ymax = None
    yscale = 'log'  # linear or log
    scatter = False
    points_size = 0.5
    graphs = (
        "clique",
        "cycle",
        "diam-expander",
        "root-expander",
        "diagonal",
        #"star",
    )

    plots = (
        "iter_time",
        "mse_iter",
        "real-mse_iter",
        "mse_time",
        "real-mse_time",
    )

    mse_log = {}
    real_mse_log = {}
    iter_log = {}

    for graph in graphs:
        mse_log[graph] = np.loadtxt("{}/{}_global_mean_squared_error_log".format(test_folder_path, graph))
        real_mse_log[graph] = np.loadtxt("{}/{}_global_real_mean_squared_error_log".format(test_folder_path, graph))
        iter_log[graph] = np.loadtxt("{}/{}_iterations_time_log".format(test_folder_path, graph))

    if not avg is None and not avg is 0:
        avg_real_mse_log = {}
        avg_mse_log = {}
        for graph in graphs:
            n_iter = len(mse_log[graph])
            avg_mse_log[graph] = np.zeros(n_iter)
            avg_real_mse_log[graph] = np.zeros(n_iter)

            for i in range(0, int(n_iter / avg)):
                beg = i * avg
                end = min((i + 1) * avg, n_iter)
                avg_mse_log[graph] = np.concatenate([
                    avg_mse_log[graph][0:beg],
                    np.full(end - beg, float(np.mean(mse_log[graph][beg:end]))),
                    avg_mse_log[graph][end:]
                ])
                avg_real_mse_log[graph] = np.concatenate([
                    avg_real_mse_log[graph][0:beg],
                    np.full(end - beg, float(np.mean(real_mse_log[graph][beg:end]))),
                    avg_real_mse_log[graph][end:]
                ])

            mse_log[graph] = avg_mse_log[graph]
            real_mse_log[graph] = avg_real_mse_log[graph]

    if "iter_time" in plots:
        plt.title("Global iterations over cluster clock")
        plt.xlabel("Time (s)")
        plt.ylabel("Iteration")
        for graph in graphs:
            n_iter = len(mse_log[graph])
            if scatter:
                plt.scatter(
                    iter_log[graph],
                    list(range(0, n_iter)),
                    label=graph,
                    s=points_size
                )
            else:
                plt.plot(
                    iter_log[graph],
                    list(range(0, n_iter)),
                    label=graph
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "1_iter_time.png"))
            plt.close()
        else:
            plt.show()

    if "mse_iter" in plots:
        plt.title("MSE over global iterations")
        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.yscale(yscale)
        if not ymax is None:
            plt.ylim(ymax=50)
        for graph in graphs:
            if scatter:
                plt.scatter(
                    list(range(0, len(mse_log[graph]))),
                    mse_log[graph],
                    label=graph,
                    s=points_size
                )
            else:
                plt.plot(
                    list(range(0, len(mse_log[graph]))),
                    mse_log[graph],
                    label=graph
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "2_mse_iter.png"))
            plt.close()
        else:
            plt.show()

    if "real-mse_iter" in plots:
        plt.title("Real MSE over global iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Real MSE")
        plt.yscale(yscale)
        if not ymax is None:
            plt.ylim(ymax=50)
        for graph in graphs:
            if scatter:
                plt.scatter(
                    list(range(0, len(real_mse_log[graph]))),
                    real_mse_log[graph],
                    label=graph,
                    s=points_size
                )
            else:
                plt.plot(
                    list(range(0, len(real_mse_log[graph]))),
                    real_mse_log[graph],
                    label=graph
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "2_real-mse_iter.png"))
            plt.close()
        else:
            plt.show()

    if "mse_time" in plots:
        plt.title("MSE over time")
        plt.xlabel("Time (s)")
        plt.ylabel("MSE")
        plt.yscale(yscale)
        if not ymax is None:
            plt.ylim(ymax=50)
        for graph in graphs:
            if scatter:
                plt.scatter(
                    iter_log[graph],
                    mse_log[graph],
                    label=graph,
                    s=points_size
                )
            else:
                plt.plot(
                    iter_log[graph],
                    mse_log[graph],
                    label=graph
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "3_mse_time.png"))
            plt.close()
        else:
            plt.show()

    if "real-mse_time" in plots:
        plt.title("Real MSE over time")
        plt.xlabel("Time (s)")
        plt.yscale(yscale)
        if not ymax is None:
            plt.ylim(ymax=50)
        plt.ylabel("Real MSE")
        for graph in graphs:
            if scatter:
                plt.scatter(
                    iter_log[graph],
                    real_mse_log[graph],
                    label=graph,
                    s=points_size
                )
            else:
                plt.plot(
                    iter_log[graph],
                    real_mse_log[graph],
                    label=graph
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "3_real-mse_time.png"))
            plt.close()
        else:
            plt.show()


def get_last_temp_test_path():
    subdirs_list = os.listdir("./test_log/temp/")
    if len(subdirs_list) == 0:
        return ""
    return os.path.normpath(os.path.join("test_log", "temp", str(max(subdirs_list))))


if __name__ == "__main__":
    # argparse setup
    parser = argparse.ArgumentParser(
        description='Plotter'
    )

    parser.add_argument(
        '-t', '--test_path',
        action='store',
        default=None,
        required=False,
        help='Test from which load logs to',
        dest='test_path'
    )

    parser.add_argument(
        '-s', '--save',
        action='store_true',
        default=False,
        required=False,
        help='Specify whether save to file or not',
        dest='s_flag'
    )
    args = parser.parse_args()

    plot_from_files(
        args.test_path,
        args.s_flag
    )
