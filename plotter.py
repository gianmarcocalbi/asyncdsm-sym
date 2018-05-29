import numpy as np
import matplotlib.pyplot as plt
import os, argparse, warnings, glob, pickle
from src.functions import *
from src import statistics


def plot_from_files(test_folder_path=None, save_to_test_folder=False, instant_plot=True):
    if test_folder_path is None:
        test_folder_path = get_last_temp_test_path()

    if not os.path.exists(test_folder_path):
        raise Exception("Folder {} doesn't exist".format(test_folder_path))

    plot_folder_path = os.path.join(test_folder_path, "plot")
    if save_to_test_folder:
        if not os.path.exists(plot_folder_path):
            os.makedirs(plot_folder_path)

    avg = None
    ymax = None
    yscale = 'log'  # linear or log
    scatter = False
    points_size = 0.5

    old_graphs = [
        "diagonal",
        "cycle",
        "diam-expander",
        "3-regular",
        "4-regular",
        "8-regular",
        "20-regular",
        "50-regular",
        "clique",
        "star",
    ]

    plots = (
        #"iter_time",
        #"avg-iter_time",

        #"iter-lb_time",
        #"avg-iter-lb_time",

        #"iter-ub_time",
        "avg-iter-ub_time",

        #"mse_iter",
        #"real-mse_iter",

        #"mse_time",
        #"real-mse_time",

        #"iter-vel_degree",
        #"iter-vel-err_degree",
    )

    n = 100
    degrees = {
        #    "0_diagonal" : 0,
        #    "1_cycle" : 1,
        #    "2_diam-expander" : 2,
        #    "3_regular" : 3,
        #    "4_regular" : 4,
        #    "8_regular" : 8,
        #    "20_regular" : 20,
        #    "50_regular" : 50,
        #    "n-1_clique" : n-1,
        #    "n-1_star" : n-1,
    }

    colors = {
        "0_diagonal": '#ba2eff',  # pink/purple
        "1_cycle": '#ffc300',  # orange
        "2_diam-expander": '#009713',  # green
        "2_root-expander": '#BB0016',  # red
        "3_regular": '#7ec2a0ff',
        "4_regular": '#97a853ff',
        "8_regular": '#c7bd30ff',
        "20_regular": '#a57a00ff',
        "50_regular": '#6b5c32ff',
        "n-1_clique": '#1537dfff',  # blue
        "n-1_star": '#F8FE21',  # yellow
    }

    graphs = []
    # loop on that pattern to find all graphs in this test's folder
    for g in glob.glob("{}/*_global_mean_squared_error_log*".format(test_folder_path)):
        # take the degree and the name of the graph
        # since logs are formatted as {deg}_{name}_{*}
        g_deg, g_label = g.split("/")[-1].split("_")[0:2]
        g_name = g_deg + "_" + g_label

        # eval is needed to evaluate degrees containing "n"
        degrees[g_name] = eval(g_deg)
        graphs.append(g_name)

    if len(graphs) == 0:
        graphs = old_graphs[:]

    mse_log = {}
    real_mse_log = {}
    iter_log = {}
    avg_iter_log = {}

    try:
        with open("{}/.setup.pkl".format(test_folder_path), 'rb') as setup_file:
            setup = pickle.load(setup_file)
    except:
        raise

    for graph in graphs[:]:
        try:
            mse_log[graph] = np.loadtxt("{}/{}_global_mean_squared_error_log".format(test_folder_path, graph))
            real_mse_log[graph] = np.loadtxt("{}/{}_global_real_mean_squared_error_log".format(test_folder_path, graph))
            iter_log[graph] = np.loadtxt("{}/{}_iterations_time_log".format(test_folder_path, graph))
            avg_iter_log[graph] = [tuple(s.split(",")) for s in
                                   np.loadtxt("{}/{}_avg_iterations_time_log".format(test_folder_path, graph), str)]

        except OSError:
            warnings.warn('Graph "{}" not found in folder {}'.format(graph, test_folder_path))
            graphs.remove(graph)

    # moving average of plot's points
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
                    s=points_size,
                    color=colors[graph]
                )
            else:
                plt.plot(
                    iter_log[graph],
                    list(range(0, n_iter)),
                    label=graph,
                    color=colors[graph]
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "1_iter_time.png"))
        if instant_plot:
            plt.show()
        plt.close()

    if "avg-iter_time" in plots:
        plt.title("Global avg iterations over cluster clock")
        plt.xlabel("Time (s)")
        plt.ylabel("Iteration")
        for graph in graphs:
            lx = [float(p[0]) for p in avg_iter_log[graph]]
            ly = [float(p[1]) for p in avg_iter_log[graph]]
            n_iter = len(mse_log[graph])
            if scatter:
                plt.scatter(
                    lx,
                    ly,
                    label=graph,
                    s=points_size,
                    color=colors[graph]
                )
            else:
                plt.plot(
                    lx,
                    ly,
                    label=graph,
                    color=colors[graph]
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "1_avg-iter_time.png"))
        if instant_plot:
            plt.show()
        plt.close()

    if "iter-lb_time" in plots:
        plt.title("Global iterations over cluster clock with LB")
        plt.xlabel("Time (s)")
        plt.ylabel("Iteration")
        for graph in graphs:
            n_iter = len(mse_log[graph])
            if scatter:
                curves = plt.scatter(
                    iter_log[graph],
                    list(range(0, n_iter)),
                    label=graph,
                    s=points_size,
                    color=colors[graph]
                )
            else:
                curves = plt.plot(
                    iter_log[graph],
                    list(range(0, n_iter)),
                    label=graph,
                    color=colors[graph]
                )

            try:
                p_x = list(range(0, int(iter_log[graph][-1])))
                p_y = []
                lb_slope = statistics.single_iteration_velocity_residual_lifetime_lower_bound(
                    degrees[graph],
                    setup['time_distr_class'],
                    setup['time_distr_param']
                )
                for x in range(len(p_x)):
                    p_y.append(x * lb_slope)

                plt.plot(
                    p_x,
                    p_y,
                    # label="{} LB".format(graph),
                    color=curves[-1].get_color(),
                    linestyle=':'
                )
            except KeyError:
                plt.plot(
                    list(range(0, int(iter_log[graph][-1]))),
                    iteration_speed_lower_bound(1, degrees[graph], list(range(0, int(iter_log[graph][-1])))),
                    # label="{} LB".format(graph),
                    color=curves[-1].get_color(),
                    linestyle=':'
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "2_iter-lb_time.png"))
        if instant_plot:
            plt.show()
        plt.close()

    if "avg-iter-lb_time" in plots:
        plt.title("Global avg iterations over cluster clock with LB")
        plt.xlabel("Time (s)")
        plt.ylabel("Iteration")
        for graph in graphs:
            lx = [float(p[0]) for p in avg_iter_log[graph]]
            ly = [float(p[1]) for p in avg_iter_log[graph]]

            curves = None
            if scatter:
                curves = plt.scatter(
                    lx,
                    ly,
                    label=graph,
                    s=points_size,
                    color=colors[graph]
                )
            else:
                curves = plt.plot(
                    lx,
                    ly,
                    label=graph,
                    color=colors[graph]
                )

            try:
                p_x = list(range(0, int(iter_log[graph][-1])))
                p_y = []
                lb_slope = statistics.single_iteration_velocity_residual_lifetime_lower_bound(
                    degrees[graph],
                    setup['time_distr_class'],
                    setup['time_distr_param']
                )
                for x in range(len(p_x)):
                    p_y.append(x * lb_slope)

                plt.plot(
                    p_x,
                    p_y,
                    # label="{} LB".format(graph),
                    color=curves[-1].get_color(),
                    linestyle=':'
                )
            except KeyError:
                plt.plot(
                    list(range(0, int(iter_log[graph][-1]))),
                    iteration_speed_lower_bound(1, degrees[graph], list(range(0, int(iter_log[graph][-1])))),
                    # label="{} LB".format(graph),
                    color=curves[-1].get_color(),
                    linestyle=':'
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "2_avg-iter-lb_time.png"))
        if instant_plot:
            plt.show()
        plt.close()

    """if "iter-ub_time" in plots:
        plt.title("Global iterations over cluster clock with UB")
        plt.xlabel("Time (s)")
        plt.ylabel("Iteration")
        for graph in graphs:
            n_iter = len(mse_log[graph])
            if scatter:
                curves = plt.scatter(
                    iter_log[graph],
                    list(range(0, n_iter)),
                    label=graph,
                    s=points_size,
                    color=colors[graph]
                )
            else:
                curves = plt.plot(
                    iter_log[graph],
                    list(range(0, n_iter)),
                    label=graph,
                    color=colors[graph]
                )

            try:
                p_x = list(range(0, int(iter_log[graph][-1])))
                p_y = []
                lb_slope = statistics.single_iteration_velocity_residual_lifetime_lower_bound(
                    degrees[graph],
                    setup['time_distr_class'],
                    setup['time_distr_param']
                )
                for x in range(len(p_x)):
                    p_y.append(x * lb_slope)

                plt.plot(
                    p_x,
                    p_y,
                    # label="{} LB".format(graph),
                    color=curves[-1].get_color(),
                    linestyle=':'
                )
            except KeyError:
                plt.plot(
                    list(range(0, int(iter_log[graph][-1]))),
                    iteration_speed_lower_bound(1, degrees[graph], list(range(0, int(iter_log[graph][-1])))),
                    # label="{} LB".format(graph),
                    color=curves[-1].get_color(),
                    linestyle=':'
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "2_iter-lb_time.png"))
        if instant_plot:
            plt.show()
        plt.close()"""

    if "avg-iter-ub_time" in plots:
        plt.title("Global avg iterations over cluster clock with UB")
        plt.xlabel("Time (s)")
        plt.ylabel("Iteration")
        for graph in graphs:
            lx = [float(p[0]) for p in avg_iter_log[graph]]
            ly = [float(p[1]) for p in avg_iter_log[graph]]

            if scatter:
                curves = plt.scatter(
                    lx,
                    ly,
                    label=graph,
                    s=points_size,
                    color=colors[graph]
                )
            else:
                curves = plt.plot(
                    lx,
                    ly,
                    label=graph,
                    color=colors[graph]
                )

            try:
                p_x = list(range(0, int(iter_log[graph][-1])))
                p_y = []
                lb_slope = statistics.single_iteration_velocity_upper_bound(
                    degrees[graph],
                    setup['time_distr_class'],
                    setup['time_distr_param']
                )
                for x in range(len(p_x)):
                    p_y.append(x * lb_slope)

                plt.plot(
                    p_x,
                    p_y,
                    # label="{} LB".format(graph),
                    color=curves[-1].get_color(),
                    linestyle=(0, (3, 5, 1, 5, 1, 5))
                )
            except KeyError:
                plt.plot(
                    list(range(0, int(iter_log[graph][-1]))),
                    iteration_speed_lower_bound(1, degrees[graph], list(range(0, int(iter_log[graph][-1])))),
                    # label="{} LB".format(graph),
                    color=curves[-1].get_color(),
                    linestyle=':'
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "2_avg-iter-ub_time.png"))
        if instant_plot:
            plt.show()
        plt.close()


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
                    color=colors[graph],
                    s=points_size
                )
            else:
                plt.plot(
                    list(range(0, len(mse_log[graph]))),
                    mse_log[graph],
                    label=graph,
                    color=colors[graph]
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "3_mse_iter.png"))
        if instant_plot:
            plt.show()
        plt.close()

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
                    color=colors[graph],
                    s=points_size
                )
            else:
                plt.plot(
                    list(range(0, len(real_mse_log[graph]))),
                    real_mse_log[graph],
                    label=graph,
                    color=colors[graph]
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "3_real-mse_iter.png"))
        if instant_plot:
            plt.show()
        plt.close()

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
                    color=colors[graph],
                    s=points_size
                )
            else:
                plt.plot(
                    iter_log[graph],
                    mse_log[graph],
                    label=graph,
                    color=colors[graph]
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "4_mse_time.png"))
        if instant_plot:
            plt.show()
        plt.close()

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
                    color=colors[graph],
                    s=points_size
                )
            else:
                plt.plot(
                    iter_log[graph],
                    real_mse_log[graph],
                    label=graph,
                    color=colors[graph]
                )
        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "4_real-mse_time.png"))
        if instant_plot:
            plt.show()
        plt.close()

    if "iter-vel-err_degree"in plots:
        plt.title("Velocity error over topology's degree")
        plt.xlabel("Degree k")
        plt.ylabel("Error (Velocity LB for k / Real Velocity for k)")

        p_x = []
        p_y = []
        for graph in graphs:
            lx = [float(p[0]) for p in avg_iter_log[graph]]
            ly = [float(p[1]) for p in avg_iter_log[graph]]
            v_lb = statistics.single_iteration_velocity_residual_lifetime_lower_bound(
                degrees[graph],
                setup['time_distr_class'],
                setup['time_distr_param']
            )
            v_real = ly[-1] / lx[-1]

            p_x.append(degrees[graph])
            p_y.append(v_lb / v_real)

        plt.plot(
            p_x,
            p_y,
            markersize=5,
            marker='o'
        )

        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "5_iter-vel-err_degree.png"))
        if instant_plot:
            plt.show()
        plt.close()

    if "iter-vel_degree" in plots:
        plt.title("Real VS LB Velocity over topology's degree")
        plt.xlabel("Degree k")
        plt.ylabel("Velocity for k (#iter / time)")

        v_real_y = []
        v_lb_y = []
        x = []
        for graph in graphs:
            lx = [float(p[0]) for p in avg_iter_log[graph]]
            ly = [float(p[1]) for p in avg_iter_log[graph]]
            v_lb = statistics.single_iteration_velocity_residual_lifetime_lower_bound(
                degrees[graph],
                setup['time_distr_class'],
                setup['time_distr_param']
            )
            v_real = ly[-1] / lx[-1]

            x.append(degrees[graph])
            v_real_y.append(v_real)
            v_lb_y.append(v_lb)


        if scatter:
            plt.scatter(
                x,
                v_real_y,
                label="Real velocity (from tests)",
                color='g',
                s=4,
                marker='_'
            )
            plt.scatter(
                x,
                v_lb_y,
                label="Velocity LB",
                color='b',
                s=4,
                marker='_'
            )
        else:
            plt.plot(
                x,
                v_real_y,
                markersize=3,
                marker='o',
                label="Real velocity (from tests)",
                color='g'
            )
            plt.plot(
                x,
                v_lb_y,
                markersize=3,
                marker='o',
                label="Velocity LB",
                color='b'
            )

        plt.legend()
        if save_to_test_folder:
            plt.savefig(os.path.join(plot_folder_path, "5_iter-vel_degree.png"))
        if instant_plot:
            plt.show()
        plt.close()


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
        args.s_flag,
        not args.s_flag
    )
