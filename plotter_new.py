import numpy as np
import matplotlib.pyplot as plt
import os, argparse, warnings, glob, pickle
from src.functions import *
from src import statistics


class Plotter:
    def _get_last_temp_test_folder_name(self):
        subdirs_list = os.listdir("./test_log/temp/")
        if len(subdirs_list) == 0:
            return ""
        return str(max(subdirs_list))

    def __init__(self,
                 test_folder_root="./test_log/",
                 test_folder_name=None,
                 test_folder_path=None,
                 save_plots_to_test_folder=False,
                 instant_plot=True,
                 plots=(),
                 moving_average_window=0,
                 ymax=None,
                 yscale='log',  # linear or log
                 scatter=False,
                 points_size=0.5,
                 ):

        self.test_folder_root = test_folder_root
        self.test_folder_name = test_folder_name
        self.save_plots_to_test_folder = save_plots_to_test_folder
        self.instant_plot = instant_plot
        self.last_temp_test_folder_name = self._get_last_temp_test_folder_name()
        self.moving_average_window = moving_average_window
        self.ymax = ymax
        self.yscale = yscale
        self.scatter = scatter
        self.points_size = points_size

        self.available_plots = (
            "iter_time",
            "avg-iter_time",
            "avg-iter_time-memoryless-lb",
            "avg-iter_time-residual-lifetime-lb",
            "avg-iter_time-ub",
            "avg-iter_time-don-bound",
            "mse_iter",
            "real-mse_iter",
            "mse_time",
            "real-mse_time",
            "iter-vel_degree",
            "iter-vel-err_degree",
            "memoryless-lb-error_degree",
            "residual-lifetime-lb-error_degree",
            "ub-error_degree"
        )

        if self.test_folder_name is None:
            if self.last_temp_test_folder_name == "":
                raise Exception("No temp test to plot")
            self.test_folder_path = os.path.normpath(
                os.path.join(
                    self.test_folder_root,
                    "temp",
                    self._get_last_temp_test_folder_name()
                )
            )
        else:
            if test_folder_path is None:
                self.test_folder_path = os.path.normpath(
                    os.path.join(
                        self.test_folder_root,
                        self.test_folder_name
                    )
                )
            else:
                self.test_folder_path = os.path.normpath(test_folder_path)

        if not os.path.exists(self.test_folder_path):
            raise Exception("Folder {} doesn't exist".format(self.test_folder_path))

        self.plot_folder_path = os.path.join(self.test_folder_path, "plot")

        if self.save_plots_to_test_folder:
            if not os.path.exists(self.plot_folder_path):
                os.makedirs(self.plot_folder_path)

        if len(plots) == 0:
            self.plots = self.available_plots
        else:
            self.plots = [graph for graph in plots if graph in self.available_plots]

        self.colors = {
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

        try:
            with open("{}/.setup.pkl".format(self.test_folder_path), 'rb') as setup_file:
                self.setup = pickle.load(setup_file)
        except:
            raise

        self.time_distr_name = self.setup['time_distr_class'].name

        self.graphs = list(self.setup['graphs'].keys())
        self.degrees = {}

        for graph in self.graphs:
            self.degrees[graph] = compute_graph_degree_from_adjacency_matrix(self.setup['graphs'][graph])

        self.mse_log = {}
        self.real_mse_log = {}
        self.iter_log = {}
        self.avg_iter_log = {}
        self.max_iter_log = {}

        for graph in self.graphs[:]:
            try:
                self.mse_log[graph] = np.loadtxt(
                    "{}/{}_global_mean_squared_error_log".format(self.test_folder_path, graph)
                )
                self.real_mse_log[graph] = np.loadtxt(
                    "{}/{}_global_real_mean_squared_error_log".format(self.test_folder_path, graph)
                )
                self.iter_log[graph] = np.loadtxt(
                    "{}/{}_iterations_time_log".format(self.test_folder_path, graph)
                )
                self.avg_iter_log[graph] = [
                    tuple(s.split(",")) for s in
                    np.loadtxt("{}/{}_avg_iterations_time_log".format(self.test_folder_path, graph), str)
                ]
                self.max_iter_log[graph] = [
                    tuple(s.split(",")) for s in
                    np.loadtxt("{}/{}_max_iterations_time_log".format(self.test_folder_path, graph), str)
                ]

            except OSError:
                warnings.warn('Graph "{}" not found in folder {}'.format(graph, self.test_folder_path))
                self.graphs.remove(graph)

        if self.moving_average_window != 0:
            avg_real_mse_log = {}
            avg_mse_log = {}
            for graph in self.graphs:
                n_iter = len(self.mse_log[graph])
                avg_mse_log[graph] = np.zeros(n_iter)
                avg_real_mse_log[graph] = np.zeros(n_iter)

                for i in range(0, int(n_iter / self.moving_average_window)):
                    beg = i * self.moving_average_window
                    end = min((i + 1) * self.moving_average_window, n_iter)
                    avg_mse_log[graph] = np.concatenate([
                        avg_mse_log[graph][0:beg],
                        np.full(end - beg, float(np.mean(self.mse_log[graph][beg:end]))),
                        avg_mse_log[graph][end:]
                    ])
                    avg_real_mse_log[graph] = np.concatenate([
                        avg_real_mse_log[graph][0:beg],
                        np.full(end - beg, float(np.mean(self.real_mse_log[graph][beg:end]))),
                        avg_real_mse_log[graph][end:]
                    ])

                self.mse_log[graph] = avg_mse_log[graph]
                self.real_mse_log[graph] = avg_real_mse_log[graph]

    def plot(self):
        if "iter_time" in self.plots:
            self.plot_iter_over_time()

        if "avg-iter_time" in self.plots:
            self.plot_avg_iter_over_time()

        if "avg-iter_time-memoryless-lb" in self.plots:
            self.plot_avg_iter_over_time_with_memoryless_lower_bound()

        if "avg-iter_time-residual-lifetime-lb" in self.plots:
            self.plot_avg_iter_over_time_with_residual_lifetime_lower_bound()

        if "avg-iter_time-ub" in self.plots:
            self.plot_avg_iter_over_time_with_upper_bound()

        if "mse_iter" in self.plots:
            self.plot_mse_over_iter()

        if "real-mse_iter" in self.plots:
            self.plot_real_mse_over_iter()

        if "mse_time" in self.plots:
            self.plot_mse_over_time()

        if "real-mse_time" in self.plots:
            self.plot_real_mse_over_time()

        if "iter-vel_degree" in self.plots:
            pass

        if "iter-vel-err_degree" in self.plots:
            pass

    def _plot_init(self,
                   plot_filename,
                   title_center="",
                   title_left="",
                   title_right="",
                   xlabel="",
                   ylabel=""
                   ):
        plt.title(title_center)
        plt.title(title_left, loc='left')
        plt.title("({})".format(title_right), loc='right')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def _plot_subroutine(self, lx, ly, **kwargs):
        if self.scatter:
            plot_func = plt.scatter
        else:
            plot_func = plt.plot

        plot_func(
            lx,
            ly,
            **kwargs
        )

    def _plot_close(self, plot_filename, legend=True):
        if legend:
            plt.legend()
        if self.save_plots_to_test_folder:
            plt.savefig(os.path.join(self.plot_folder_path, plot_filename + ".png"))
        if self.instant_plot:
            plt.show()
        plt.close()

    def _plot_iter_over_time_lines(self, **kwargs):
        for graph in self.graphs:
            self._plot_subroutine(
                self.iter_log[graph],
                list(range(len(self.iter_log[graph]))),
                label=graph,
                color=self.colors[graph],
                **kwargs)

    def _plot_avg_iter_over_time_lines(self, **kwargs):
        for graph in self.graphs:
            lx = [float(p[0]) for p in self.avg_iter_log[graph]]
            ly = [float(p[1]) for p in self.avg_iter_log[graph]]

            self._plot_subroutine(
                lx,
                ly,
                label=graph,
                color=self.colors[graph],
                **kwargs)

    def _plot_max_iter_over_time_lines(self, **kwargs):
        for graph in self.graphs:
            lx = [float(p[0]) for p in self.max_iter_log[graph]]
            ly = [float(p[1]) for p in self.max_iter_log[graph]]

            self._plot_subroutine(
                lx,
                ly,
                label=graph,
                color=self.colors[graph],
                **kwargs)

    def _plot_iter_over_time_bound_lines_subroutine(self, statistics_lb_function, **kwargs):
        for graph in self.graphs:
            slope = statistics_lb_function(
                self.degrees[graph],
                self.setup['time_distr_class'],
                self.setup['time_distr_param']
            )
            lx = list(range(int(self.iter_log[graph][-1])))
            ly = [slope * x for x in lx]

            self._plot_subroutine(
                lx,
                ly,
                color=self.colors[graph],
                **kwargs)

    def _plot_iter_over_time_residual_lifetime_lower_bound_lines(self, **kwargs):
        self._plot_iter_over_time_bound_lines_subroutine(
            statistics.single_iteration_velocity_residual_lifetime_lower_bound,
            **kwargs
        )

    def _plot_iter_over_time_memoryless_lower_bound_lines(self, **kwargs):
        self._plot_iter_over_time_bound_lines_subroutine(
            statistics.single_iteration_velocity_memoryless_lower_bound,
            **kwargs
        )

    def _plot_iter_over_time_upper_bound_lines(self, **kwargs):
        self._plot_iter_over_time_bound_lines_subroutine(
            statistics.single_iteration_velocity_upper_bound,
            **kwargs
        )

    def _plot_iter_over_time_don_bound_lines(self, **kwargs):
        self._plot_iter_over_time_bound_lines_subroutine(
            statistics.single_iteration_velocity_don_bound,
            **kwargs
        )

    # ERROR over ITERATIONS BEGIN

    def _plot_mse_over_iter_lines(self, **kwargs):
        for graph in self.graphs:
            self._plot_subroutine(
                self.mse_log[graph],
                list(range(len(self.mse_log[graph]))),
                label=graph,
                color=self.colors[graph],
                **kwargs)

    def _plot_real_mse_over_iter_lines(self, **kwargs):
        for graph in self.graphs:
            self._plot_subroutine(
                self.real_mse_log[graph],
                list(range(len(self.real_mse_log[graph]))),
                label=graph,
                color=self.colors[graph],
                **kwargs)

    def _plot_mse_over_time_lines(self, **kwargs):
        for graph in self.graphs:
            self._plot_subroutine(
                self.iter_log[graph],
                self.mse_log[graph],
                label=graph,
                color=self.colors[graph],
                **kwargs)

    def _plot_real_mse_over_time_lines(self, **kwargs):
        for graph in self.graphs:
            self._plot_subroutine(
                self.iter_log[graph],
                self.real_mse_log[graph],
                label=graph,
                color=self.colors[graph],
                **kwargs)

    # ERROR over ITERATIONS END

    # BOUND ERROR over DEGREE BEGIN

    def _plot_bound_error_over_degree_lines_subroutine(self, bound_func, **kwargs):
        p_x = []
        p_y = []
        for graph in self.graphs:
            lx = [float(p[0]) for p in self.avg_iter_log[graph]]
            ly = [float(p[1]) for p in self.avg_iter_log[graph]]
            v_lb = bound_func(
                self.degrees[graph],
                self.setup['time_distr_class'],
                self.setup['time_distr_param']
            )
            v_real = ly[-1] / lx[-1]

            p_x.append(self.degrees[graph])
            p_y.append(v_lb / v_real)

        self._plot_subroutine(
            p_x,
            p_y,
            markersize=5,
            marker='o',
            **kwargs)

    def _plot_memoryless_lower_bound_error_over_degree_lines(self, **kwargs):
        self._plot_bound_error_over_degree_lines_subroutine(
            statistics.single_iteration_velocity_memoryless_lower_bound,
            **kwargs
        )

    def _plot_residual_lifetime_lower_bound_error_over_degree_lines(self, **kwargs):
        self._plot_bound_error_over_degree_lines_subroutine(
            statistics.single_iteration_velocity_residual_lifetime_lower_bound,
            **kwargs
        )

    def _plot_upper_bound_error_over_degree_lines(self, **kwargs):
        self._plot_bound_error_over_degree_lines_subroutine(
            statistics.single_iteration_velocity_upper_bound,
            **kwargs
        )

    def _plot_don_bound_error_over_degree_lines(self, **kwargs):
        self._plot_bound_error_over_degree_lines_subroutine(
            statistics.single_iteration_velocity_don_bound,
            **kwargs
        )

    # BOUND ERROR over DEGREE END

    #

    def plot_iter_over_time(self):
        filename = "1_iter_time"
        self._plot_init(filename,
                        title_center="",
                        title_left="Smallest iteration over time",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="Iteration")
        self._plot_iter_over_time_lines()
        self._plot_close(filename)

    def plot_avg_iter_over_time(self):
        filename = "1_avg-iter_time"
        self._plot_init(filename,
                        title_center="",
                        title_left="Average iteration at time",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="Average iteration")
        self._plot_avg_iter_over_time_lines()
        self._plot_close(filename)

    # PLOTS WITH BOUNDS BEGIN

    def plot_iter_over_time_memoryless_lower_bound_only(self):
        filename = "1_iter_time-memoryless-lb-only"
        self._plot_init(filename,
                        title_center="",
                        title_left="Iterations over time memoryless lower bound",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="Iteration")
        self._plot_iter_over_time_memoryless_lower_bound_lines()
        self._plot_close(filename)

    def plot_iter_over_time_residual_lifetime_lower_bound_only(self):
        filename = "1_iter_time-residual-lifetime-lb-only"
        self._plot_init(filename,
                        title_center="",
                        title_left="Iterations over time residual time lower bound",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="Iteration")
        self._plot_iter_over_time_residual_lifetime_lower_bound_lines()
        self._plot_close(filename)

    def plot_iter_over_time_upper_bound_only(self):
        filename = "1_iter_time-ub-only"
        self._plot_init(filename,
                        title_center="",
                        title_left="Iterations over time upper bound",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="Iteration")
        self._plot_iter_over_time_upper_bound_lines()
        self._plot_close(filename)

    def plot_avg_iter_over_time_with_memoryless_lower_bound(self):
        filename = "1_avg-iter_time-memoryless-lb"
        self._plot_init(filename,
                        title_center="",
                        title_left="Average iteration at time with memoryless lower bound",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="Iteration")
        self._plot_iter_over_time_memoryless_lower_bound_lines(
            linestyle=(0, (1, 8))
        )
        self._plot_avg_iter_over_time_lines()
        self._plot_close(filename)

    def plot_avg_iter_over_time_with_residual_lifetime_lower_bound(self):
        filename = "1_avg-iter_time-residual-lifetime-lb"
        self._plot_init(filename,
                        title_center="",
                        title_left="Average iteration at time with residual lifetime lower bound",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="Iteration")
        self._plot_iter_over_time_residual_lifetime_lower_bound_lines(
            linestyle=(0, (1, 4))
        )
        self._plot_avg_iter_over_time_lines()
        self._plot_close(filename)

    def plot_avg_iter_over_time_with_upper_bound(self):
        filename = "1_avg-iter_time-ub"
        self._plot_init(filename,
                        title_center="",
                        title_left="Average iteration at time with upper bound",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="Iteration")
        self._plot_iter_over_time_upper_bound_lines(
            linestyle=(0, (3, 5, 1, 5))
        )
        self._plot_avg_iter_over_time_lines()
        self._plot_close(filename)

    def plot_avg_iter_over_time_with_don_bound(self):
        filename = "1_avg-iter_time-don-bound"
        self._plot_init(filename,
                        title_center="",
                        title_left="Average iteration at time with \"Don\" bound for the fastest node",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="Iteration")
        self._plot_iter_over_time_don_bound_lines(
            linestyle=(0, (1, 4))
        )
        self._plot_max_iter_over_time_lines()
        self._plot_close(filename)

    # PLOTS WITH BOUNDS BEGIN

    def plot_mse_over_iter(self):
        filename = "2_mse_iter"
        self._plot_init(filename,
                        title_center="",
                        title_left="MSE over iterations",
                        title_right=self.time_distr_name,
                        xlabel="Iteration",
                        ylabel="MSE")
        self._plot_mse_over_iter_lines()
        self._plot_close(filename)

    def plot_real_mse_over_iter(self):
        filename = "2_real-mse_iter"
        self._plot_init(filename,
                        title_center="",
                        title_left="RMSE over iterations",
                        title_right=self.time_distr_name,
                        xlabel="Iteration",
                        ylabel="RMSE")
        self._plot_real_mse_over_iter_lines()
        self._plot_close(filename)

    def plot_mse_over_time(self):
        filename = "3_mse_time"
        self._plot_init(filename,
                        title_center="",
                        title_left="MSE over time",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="MSE")
        self._plot_mse_over_time_lines()
        self._plot_close(filename)

    def plot_real_mse_over_time(self):
        filename = "3_real-mse_time"
        self._plot_init(filename,
                        title_center="",
                        title_left="RMSE over time",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="RMSE")
        self._plot_real_mse_over_time_lines()
        self._plot_close(filename)

    def plot_memoryless_lower_bound_error_over_degree(self):
        filename = "4_memoryless-lb-error_degree"
        self._plot_init(filename,
                        title_center="",
                        title_left="Memoryless lower bound error over degree",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="LB Error (Velocity LB for k / Real Velocity for k)")
        self._plot_memoryless_lower_bound_error_over_degree_lines()
        self._plot_close(filename)

    def plot_residual_lifetime_lower_bound_error_over_degree(self):
        filename = "4_residual-lifetime-lb-error_degree"
        self._plot_init(filename,
                        title_center="",
                        title_left="Residual lifetime lower bound error over degree",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="LB Error (Velocity LB for k / Real Velocity for k)")
        self._plot_residual_lifetime_lower_bound_error_over_degree_lines()
        self._plot_close(filename)

    def plot_upper_bound_error_over_degree(self):
        filename = "4_ub-error_degree"
        self._plot_init(filename,
                        title_center="",
                        title_left="Upper bound error over degree",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="LB Error (Velocity UB for k / Real Velocity for k)")
        self._plot_upper_bound_error_over_degree_lines()
        self._plot_close(filename)

    def plot_don_bound_error_over_degree(self):
        filename = "4_don-b-error_degree"
        self._plot_init(filename,
                        title_center="",
                        title_left="\"Don\" bound error over degree",
                        title_right=self.time_distr_name,
                        xlabel="Time (s)",
                        ylabel="Bound Error (Velocity Bound for k / Real Velocity for k)")
        self._plot_don_bound_error_over_degree_lines()
        self._plot_close(filename)


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

    Plotter(
        test_folder_path=args.test_path,
        save_plots_to_test_folder=args.s_flag,
        instant_plot=not args.s_flag,
        plots=(
            "iter_time",
            "avg-iter_time",
            "avg-iter_time-memoryless-lb",
            "avg-iter_time-residual-lifetime-lb",
            "avg-iter_time-ub",
            "mse_iter",
            "real-mse_iter",
            "mse_time",
            "real-mse_time",
            "iter-vel_degree",
            "iter-vel-err_degree",
        ),
        moving_average_window=0,
        ymax=None,
        yscale='log',  # linear or log
        scatter=False,
        points_size=0.5
    )
