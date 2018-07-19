import numpy as np
import matplotlib.pyplot as plt
import os, warnings, glob, pickle
from src.utils import *
from src import statistics
from src.mltoolbox.metrics import METRICS


def plot_from_files(**kwargs):
    Plotter(**kwargs).plot()


class Plotter:
    def __init__(
            self,
            test_folder_root="./test_log/",
            test_folder_name=None,
            test_folder_path=None,
            temp_index=0,
            save_plots_to_test_folder=False,
            instant_plot=True,
            plots=(),
            moving_average_window=0,
            ymax=None,
            yscale='log',  # linear or log
            scatter=False,
            points_size=0.5,
            verbose=False
    ):

        self.verbose = verbose
        self.test_folder_root = test_folder_root
        self.test_folder_name = test_folder_name
        self.test_folder_path = test_folder_path
        self.save_plots_to_test_folder = save_plots_to_test_folder
        self.instant_plot = instant_plot
        self.temp_test_folder_name = Plotter.get_temp_test_folder_name_by_index(temp_index)
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

            "iter-memoryless-lb-error_degree",
            "iter-residual-lifetime-lb-error_degree",
            "iter-ub-error_degree",
            "iter-all-bounds-error_degree",

            "iter-memoryless-lb-velocity_degree",
            "iter-residual-lifetime-lb-velocity_degree",
            "iter-ub-velocity_degree",
            "iter-all-bounds-velocity_degree",
        )

        if self.test_folder_name is None and test_folder_path is None:
            if self.temp_test_folder_name == "":
                raise Exception("No temp test to plot")
            self.test_folder_name = Plotter.get_temp_test_folder_name_by_index(temp_index)
            self.test_folder_path = os.path.normpath(
                os.path.join(
                    self.test_folder_root,
                    "temp",
                    self.test_folder_name
                )
            )
        elif test_folder_path is None:
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

        # if no plots array has been specified then consider all of them
        if len(plots) == 0:
            self.plots = self.available_plots
        else:
            # take only specified plots by making intersection between available plots and
            # passed parameter array
            self.plots = [graph for graph in plots if graph in self.available_plots]

        self.colors = {
            "0_diagonal": '#ba2eff',  # pink/purple
            "1_cycle": '#ffc300',  # orange
            "2_cycle-bi": '#ffea01',
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
            print("No setup file to open")
            raise

        self.time_distr_name = self.setup['time_distr_class'].name

        self.graphs = list(self.setup['graphs'].keys())
        self.degrees = {}

        for graph in self.graphs:
            self.degrees[graph] = degree_from_adjacency_matrix(self.setup['graphs'][graph])

        self.logs = {
            "iter_time": {},
            "avg_iter_time": {},
            "max_iter_time": {},
            "metrics" : {}
        }

        # Fill self.metrics with instances of metrics objects
        if not (isinstance(self.setup["metrics"], list) or isinstance(self.setup["metrics"], tuple)):
            if self.setup["metrics"] in METRICS:
                self.setup["metrics"] = [self.setup["metrics"]]
            elif self.setup["metrics"].lower() == 'all':
                self.setup["metrics"] = list(METRICS.keys())
        if not (isinstance(self.setup["real_metrics"], list) or isinstance(self.setup["real_metrics"], tuple)):
            if self.setup["real_metrics"] in METRICS:
                self.setup["real_metrics"] = [self.setup["real_metrics"]]
            elif self.setup["real_metrics"].lower() == 'all':
                self.setup["real_metrics"] = list(METRICS.keys())

        for m in self.setup["metrics"]:
            self.logs["metrics"][m] = {}
        for rm in self.setup["real_metrics"]:
            self.logs["metrics"]["real_" + rm] = {}

        # it's important to loop on a copy of self.graphs and not on the original one
        # since the original in modified inside the loop
        for graph in self.graphs[:]:
            iter_log_path = "{}/{}_iter_time_log".format(self.test_folder_path, graph)
            avg_iter_log_path = "{}/{}_avg_iter_time_log".format(self.test_folder_path, graph)
            max_iter_log_path = "{}/{}_max_iter_time_log".format(self.test_folder_path, graph)

            if os.path.isfile(iter_log_path + ".gz"):
                iter_log_path += '.gz'
            if os.path.isfile(avg_iter_log_path + ".gz"):
                avg_iter_log_path += '.gz'
            if os.path.isfile(max_iter_log_path + ".gz"):
                max_iter_log_path += '.gz'

            try:
                self.logs["iter_time"][graph] = np.loadtxt(iter_log_path)
                self.logs["avg_iter_time"][graph] = [tuple(s.split(",")) for s in np.loadtxt(avg_iter_log_path, str)]
                self.logs["max_iter_time"][graph] = [tuple(s.split(",")) for s in np.loadtxt(max_iter_log_path, str)]
            except OSError:
                warnings.warn('Graph "{}" not found in folder {}'.format(graph, self.test_folder_path))
                self.graphs.remove(graph)
                continue

            for metrics_log in self.logs["metrics"]:
                metrics_log_path = "{}/{}_{}_log".format(self.test_folder_path, graph, metrics_log)
                if os.path.isfile(metrics_log_path + ".gz"):
                    metrics_log_path += '.gz'

                try:
                    self.logs["metrics"][metrics_log][graph] = np.loadtxt(metrics_log_path)
                except OSError:
                    warnings.warn('Graph "{}" not found in folder {}'.format(graph, self.test_folder_path))
                    self.graphs.remove(graph)
                    continue

        # TEMPORARY BACKWARD COMPATIBILITY
        self.mse_log = self.logs["metrics"]["mse"]
        self.real_mse_log = self.logs["metrics"]["real_mse"]
        self.iter_log = self.logs["iter_time"]
        self.avg_iter_log = self.logs["avg_iter_time"]
        self.max_iter_log = self.logs["max_iter_time"]

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

    @staticmethod
    def get_temp_test_folder_name_by_index(index=0):
        subdirs_list = os.listdir("./test_log/temp/")
        if len(subdirs_list) == 0:
            return ""
        subdirs_list.sort(reverse=True)
        if abs(index) > len(subdirs_list) - 1:
            index = 0
        return str(subdirs_list[index])

    @staticmethod
    def get_temp_test_folder_path_by_index(index=0):
        return os.path.normpath(os.path.join("./test_log/temp/", Plotter.get_temp_test_folder_name_by_index(index)))

    def get_graph_color(self, graph):
        if graph in self.colors:
            return self.colors[graph]
        else:
            h = hex(int(np.random.uniform(0, 0xffffff)))[2:] + "000000"
            h = "#" + h[0:6]
            warnings.warn("Graph '{}' has no own color specified, used {} instead".format(graph, h))
            return h

    # PLOT MAIN UTILS METHODS - BEGIN

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

        if "iter-memoryless-lb-error_degree" in self.plots:
            self.plot_iter_memoryless_lower_bound_error_over_degree()

        if "iter-residual-lifetime-lb-error_degree" in self.plots:
            self.plot_iter_residual_lifetime_lower_bound_error_over_degree()

        if "iter-ub-error_degree" in self.plots:
            self.plot_iter_upper_bound_error_over_degree()

        if "iter-all-bounds-error_degree" in self.plots:
            self.plot_iter_all_bounds_error_over_degree()

        if "iter-memoryless-lb-velocity_degree" in self.plots:
            self.plot_iter_memoryless_lower_bound_velocity_over_degree()

        if "iter-residual-lifetime-lb-velocity_degree" in self.plots:
            self.plot_iter_residual_lifetime_lower_bound_velocity_over_degree()

        if "iter-ub-velocity_degree" in self.plots:
            self.plot_iter_upper_bound_velocity_over_degree()

        if "iter-all-bounds-velocity_degree" in self.plots:
            self.plot_iter_all_bounds_velocity_over_degrees_with_real_velocity()

    def _plot_init(self,
                   title_center="",
                   title_left="",
                   title_right="",
                   xlabel="",
                   ylabel="",
                   ymax=None,
                   yscale=None
                   ):
        plt.title(title_center)
        plt.title(title_left, loc='left')
        plt.title("({})".format(title_right), loc='right')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if ymax is None:
            ymax = self.ymax
        if yscale is None:
            yscale = self.yscale

        if not ymax is None:
            plt.ylim(ymax)
        if not yscale is None:
            plt.yscale(yscale)

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

    # PLOT MAIN UTILS METHODS - END

    # PLOT ITER over TIME LINES - BEGIN

    def _plot_iter_over_time_lines(self, **kwargs):
        for graph in self.graphs:
            self._plot_subroutine(
                self.iter_log[graph],
                list(range(len(self.iter_log[graph]))),
                label=graph,
                color=self.get_graph_color(graph),
                **kwargs
            )

    def _plot_avg_iter_over_time_lines(self, **kwargs):
        for graph in self.graphs:
            lx = [float(p[0]) for p in self.avg_iter_log[graph]]
            ly = [float(p[1]) for p in self.avg_iter_log[graph]]

            self._plot_subroutine(
                lx,
                ly,
                label=graph,
                color=self.get_graph_color(graph),
                **kwargs
            )

    def _plot_max_iter_over_time_lines(self, **kwargs):
        for graph in self.graphs:
            lx = [float(p[0]) for p in self.max_iter_log[graph]]
            ly = [float(p[1]) for p in self.max_iter_log[graph]]

            self._plot_subroutine(
                lx,
                ly,
                label=graph,
                color=self.get_graph_color(graph),
                **kwargs
            )

    def _plot_iter_over_time_bound_lines_subroutine(self, statistics_lb_function, **kwargs):
        for graph in self.graphs:
            slope = statistics_lb_function(
                self.degrees[graph],
                self.setup['time_distr_class'],
                self.setup['time_distr_param']
            )

            # iter_log[graph] is an array indexed as "iter#" -> time of such iter completion
            # so int(self.iter_log[graph][-1]) take the rightmost x value of iter_over_time line
            lx = list(range(int(self.iter_log[graph][-1])))
            ly = [slope * x for x in lx]

            self._plot_subroutine(
                lx,
                ly,
                color=self.get_graph_color(graph),
                **kwargs
            )

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

    # PLOT ITER over TIME LINES - BEGIN

    # PLOT ERROR over ITER LINES - BEGIN

    def _plot_mse_over_iter_lines(self, **kwargs):
        for graph in self.graphs:
            self._plot_subroutine(
                list(range(len(self.mse_log[graph]))),
                self.mse_log[graph],
                label=graph,
                color=self.get_graph_color(graph),
                **kwargs)

    def _plot_real_mse_over_iter_lines(self, **kwargs):
        for graph in self.graphs:
            self._plot_subroutine(
                list(range(len(self.real_mse_log[graph]))),
                self.real_mse_log[graph],
                label=graph,
                color=self.get_graph_color(graph),
                **kwargs)

    def _plot_mse_over_time_lines(self, **kwargs):
        for graph in self.graphs:
            self._plot_subroutine(
                self.iter_log[graph],
                self.mse_log[graph],
                label=graph,
                color=self.get_graph_color(graph),
                **kwargs)

    def _plot_real_mse_over_time_lines(self, **kwargs):
        for graph in self.graphs:
            self._plot_subroutine(
                self.iter_log[graph],
                self.real_mse_log[graph],
                label=graph,
                color=self.get_graph_color(graph),
                **kwargs)

    # PLOT ERROR over ITER LINES - END

    # PLOT ITER BOUND ERROR over DEGREE LINES - BEGIN

    def _plot_iter_bound_error_over_degree_lines_subroutine(self, bound_func, **kwargs):
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

    def _plot_iter_real_velocity_error_over_degree_lines(self, **kwargs):
        p_x = list(self.degrees.values())
        p_y = [1 for _ in p_x]

        self._plot_subroutine(
            p_x,
            p_y,
            markersize=5,
            marker='o',
            color='g',
            label='Best bound error',
            **kwargs)

    def _plot_iter_memoryless_lower_bound_error_over_degree_lines(self, **kwargs):
        self._plot_iter_bound_error_over_degree_lines_subroutine(
            statistics.single_iteration_velocity_memoryless_lower_bound,
            label="Memoryless lb",
            color='m',
            **kwargs
        )

    def _plot_iter_residual_lifetime_lower_bound_error_over_degree_lines(self, **kwargs):
        self._plot_iter_bound_error_over_degree_lines_subroutine(
            statistics.single_iteration_velocity_residual_lifetime_lower_bound,
            label='Residual lifetime lb',
            color='r',
            **kwargs
        )

    def _plot_iter_upper_bound_error_over_degree_lines(self, **kwargs):
        self._plot_iter_bound_error_over_degree_lines_subroutine(
            statistics.single_iteration_velocity_upper_bound,
            label='Upper bound',
            color='b',
            **kwargs
        )

    def _plot_iter_don_bound_error_over_degree_lines(self, **kwargs):
        self._plot_iter_bound_error_over_degree_lines_subroutine(
            statistics.single_iteration_velocity_don_bound,
            label='Don bound',
            color='y',
            **kwargs
        )

    # PLOT ITER BOUND ERROR over DEGREE LINES - END

    # PLOT ITER BOUND VELOCITY over DEGREES LINES - BEGIN

    def _plot_iter_bound_velocity_over_degree_lines_subroutine(self, bound_func, **kwargs):
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
            p_y.append(v_lb)

        self._plot_subroutine(
            p_x,
            p_y,
            markersize=3,
            marker='o',
            **kwargs)

    def _plot_iter_real_velocity_over_degree_lines(self, **kwargs):
        p_x = []
        p_y = []
        for graph in self.graphs:
            lx = [float(p[0]) for p in self.avg_iter_log[graph]]
            ly = [float(p[1]) for p in self.avg_iter_log[graph]]
            v_real = ly[-1] / lx[-1]

            p_x.append(self.degrees[graph])
            p_y.append(v_real)

        self._plot_subroutine(
            p_x,
            p_y,
            markersize=3,
            marker='o',
            color='g',
            label='Real velocity',
            **kwargs)

    def _plot_iter_memoryless_lower_bound_velocity_over_degree_lines(self, **kwargs):
        self._plot_iter_bound_velocity_over_degree_lines_subroutine(
            statistics.single_iteration_velocity_memoryless_lower_bound,
            label="Memoryless lb",
            color='m',
            **kwargs
        )

    def _plot_iter_residual_lifetime_lower_bound_velocity_over_degree_lines(self, **kwargs):
        self._plot_iter_bound_velocity_over_degree_lines_subroutine(
            statistics.single_iteration_velocity_residual_lifetime_lower_bound,
            label='Residual lifetime lb',
            color='r',
            **kwargs
        )

    def _plot_iter_upper_bound_velocity_over_degree_lines(self, **kwargs):
        self._plot_iter_bound_velocity_over_degree_lines_subroutine(
            statistics.single_iteration_velocity_upper_bound,
            label='Upper bound',
            color='b',
            **kwargs
        )

    def _plot_iter_don_bound_velocity_over_degree_lines(self, **kwargs):
        self._plot_iter_bound_velocity_over_degree_lines_subroutine(
            statistics.single_iteration_velocity_don_bound,
            label='Don bound',
            color='y',
            **kwargs
        )

    # PLOT ITER BOUND VELOCITY over DEGREES LINES - END

    # PUBLIC INTERFACE - BEGIN

    def plot_iter_over_time(self):
        filename = "1_iter_time"
        self._plot_init(
            title_center="",
            title_left="Smallest iteration over time",
            title_right=self.time_distr_name,
            xlabel="Time (s)",
            ylabel="Iteration",
            yscale='linear'
        )
        self._plot_iter_over_time_lines()
        self._plot_close(filename)

    def plot_avg_iter_over_time(self):
        filename = "1_avg-iter_time"
        self._plot_init(
            title_center="",
            title_left="Average iteration at time",
            title_right=self.time_distr_name,
            xlabel="Time (s)",
            ylabel="Average iteration",
            yscale='linear')
        self._plot_avg_iter_over_time_lines()
        self._plot_close(filename)

    # PLOTS WITH BOUNDS BEGIN

    def plot_iter_over_time_memoryless_lower_bound_only(self):
        filename = "1_iter_time-memoryless-lb-only"
        self._plot_init(
            title_center="",
            title_left="Iterations over time memoryless lower bound",
            title_right=self.time_distr_name,
            xlabel="Time (s)",
            ylabel="Iteration",
            yscale='linear')
        self._plot_iter_over_time_memoryless_lower_bound_lines()
        self._plot_close(filename)

    def plot_iter_over_time_residual_lifetime_lower_bound_only(self):
        filename = "1_iter_time-residual-lifetime-lb-only"
        self._plot_init(
            title_center="",
            title_left="Iterations over time residual time lower bound",
            title_right=self.time_distr_name,
            xlabel="Time (s)",
            ylabel="Iteration",
            yscale='linear')
        self._plot_iter_over_time_residual_lifetime_lower_bound_lines()
        self._plot_close(filename)

    def plot_iter_over_time_upper_bound_only(self):
        filename = "1_iter_time-ub-only"
        self._plot_init(
            title_center="",
            title_left="Iterations over time upper bound",
            title_right=self.time_distr_name,
            xlabel="Time (s)",
            ylabel="Iteration",
            yscale='linear')
        self._plot_iter_over_time_upper_bound_lines()
        self._plot_close(filename)

    def plot_avg_iter_over_time_with_memoryless_lower_bound(self):
        filename = "1_avg-iter_time-memoryless-lb"
        self._plot_init(
            title_center="",
            title_left="Average iteration at time with memoryless lower bound",
            title_right=self.time_distr_name,
            xlabel="Time (s)",
            ylabel="Iteration",
            yscale='linear'
        )
        self._plot_iter_over_time_memoryless_lower_bound_lines(
            linestyle=(0, (1, 8))
        )
        self._plot_avg_iter_over_time_lines()
        self._plot_close(filename)

    def plot_avg_iter_over_time_with_residual_lifetime_lower_bound(self):
        filename = "1_avg-iter_time-residual-lifetime-lb"
        self._plot_init(
            title_center="",
            title_left="Average iteration at time with residual lifetime lower bound",
            title_right=self.time_distr_name,
            xlabel="Time (s)",
            ylabel="Iteration",
            yscale='linear')
        self._plot_iter_over_time_residual_lifetime_lower_bound_lines(
            linestyle=(0, (1, 4))
        )
        self._plot_avg_iter_over_time_lines()
        self._plot_close(filename)

    def plot_avg_iter_over_time_with_upper_bound(self):
        filename = "1_avg-iter_time-ub"
        self._plot_init(
            title_center="",
            title_left="Average iteration at time with upper bound",
            title_right=self.time_distr_name,
            xlabel="Time (s)",
            ylabel="Iteration",
            yscale='linear')
        self._plot_iter_over_time_upper_bound_lines(
            linestyle=(0, (3, 5, 1, 5))
        )
        self._plot_avg_iter_over_time_lines()
        self._plot_close(filename)

    def plot_avg_iter_over_time_with_don_bound(self):
        filename = "1_avg-iter_time-don-bound"
        self._plot_init(
            title_center="",
            title_left="Average iteration at time with \"Don\" bound for the fastest node",
            title_right=self.time_distr_name,
            xlabel="Time (s)",
            ylabel="Iteration",
            yscale='linear')
        self._plot_iter_over_time_don_bound_lines(
            linestyle=(0, (1, 4))
        )
        self._plot_max_iter_over_time_lines()
        self._plot_close(filename)

    # PLOTS WITH BOUNDS BEGIN

    def plot_mse_over_iter(self):
        filename = "2_mse_iter"
        self._plot_init(
            title_center="",
            title_left="MSE over iterations",
            title_right=self.time_distr_name,
            xlabel="Iteration",
            ylabel="MSE")
        self._plot_mse_over_iter_lines()
        self._plot_close(filename)

    def plot_real_mse_over_iter(self):
        filename = "2_real-mse_iter"
        self._plot_init(
            title_center="",
            title_left="RMSE over iterations",
            title_right=self.time_distr_name,
            xlabel="Iteration",
            ylabel="RMSE")
        self._plot_real_mse_over_iter_lines()
        self._plot_close(filename)

    def plot_mse_over_time(self):
        filename = "3_mse_time"
        self._plot_init(
            title_center="",
            title_left="MSE over time",
            title_right=self.time_distr_name,
            xlabel="Time (s)",
            ylabel="MSE")
        self._plot_mse_over_time_lines()
        self._plot_close(filename)

    def plot_real_mse_over_time(self):
        filename = "3_real-mse_time"
        self._plot_init(
            title_center="",
            title_left="RMSE over time",
            title_right=self.time_distr_name,
            xlabel="Time (s)",
            ylabel="RMSE")
        self._plot_real_mse_over_time_lines()
        self._plot_close(filename)

    def plot_iter_memoryless_lower_bound_error_over_degree(self):
        filename = "4_iter-memoryless-lb-error_degree"
        self._plot_init(
            title_center="",
            title_left="Memoryless lower bound error over degree",
            title_right=self.time_distr_name,
            yscale='linear',
            xlabel="Degree",
            ylabel="LB Error (Velocity LB for k / Real Velocity for k)")
        self._plot_iter_memoryless_lower_bound_error_over_degree_lines()
        self._plot_close(filename)

    def plot_iter_residual_lifetime_lower_bound_error_over_degree(self):
        filename = "4_iter-residual-lifetime-lb-error_degree"
        self._plot_init(
            title_center="",
            title_left="Residual lifetime lower bound error over degree",
            title_right=self.time_distr_name,
            yscale='linear',
            xlabel="Degree",
            ylabel="LB Error (Velocity LB for k / Real Velocity for k)")
        self._plot_iter_residual_lifetime_lower_bound_error_over_degree_lines()
        self._plot_close(filename)

    def plot_iter_upper_bound_error_over_degree(self):
        filename = "4_iter-ub-error_degree"
        self._plot_init(
            title_center="",
            title_left="Upper bound error over degree",
            title_right=self.time_distr_name,
            yscale='linear',
            xlabel="Degree",
            ylabel="LB Error (Velocity UB for k / Real Velocity for k)")
        self._plot_iter_upper_bound_error_over_degree_lines()
        self._plot_close(filename)

    def plot_iter_don_bound_error_over_degree(self):
        filename = "4_iter-don-b-error_degree"
        self._plot_init(
            title_center="",
            title_left="\"Don\" bound error over degree",
            title_right=self.time_distr_name,
            yscale='linear',
            xlabel="Degree",
            ylabel="Bound Error (Velocity Bound for k / Real Velocity for k)")
        self._plot_iter_don_bound_error_over_degree_lines()
        self._plot_close(filename)

    def plot_iter_all_bounds_error_over_degree(self):
        filename = "4_iter-all-b-error_degree"
        self._plot_init(
            title_center="",
            title_left="All bounds' errors over degree",
            title_right=self.time_distr_name,
            yscale='linear',
            xlabel="Degree",
            ylabel="Bound Error (Velocity Bound for k / Real Velocity for k)")
        self._plot_iter_memoryless_lower_bound_error_over_degree_lines()
        self._plot_iter_residual_lifetime_lower_bound_error_over_degree_lines()
        self._plot_iter_upper_bound_error_over_degree_lines()
        self._plot_iter_don_bound_error_over_degree_lines()
        self._plot_close(filename)

    def plot_iter_all_bounds_error_over_degree_with_real_velocity(self):
        filename = "4_iter-all-b-error_degree"
        self._plot_init(
            title_center="",
            title_left="All bounds' errors over degree",
            title_right=self.time_distr_name,
            yscale='linear',
            xlabel="Degree",
            ylabel="Bound Error (Velocity Bound for k / Real Velocity for k)")
        self._plot_iter_memoryless_lower_bound_error_over_degree_lines()
        self._plot_iter_residual_lifetime_lower_bound_error_over_degree_lines()
        self._plot_iter_upper_bound_error_over_degree_lines()
        self._plot_iter_don_bound_error_over_degree_lines()
        self._plot_iter_real_velocity_error_over_degree_lines()
        self._plot_close(filename)

    def plot_iter_memoryless_lower_bound_velocity_over_degree(self):
        filename = "5_iter-memoryless-lb-velocity_degree"
        self._plot_init(
            title_center="",
            title_left="Memoryless lower bound velocity over degree",
            title_right=self.time_distr_name,
            yscale='linear',
            xlabel="Degree",
            ylabel="Bound Velocity")
        self._plot_iter_memoryless_lower_bound_velocity_over_degree_lines()
        self._plot_close(filename)

    def plot_iter_residual_lifetime_lower_bound_velocity_over_degree(self):
        filename = "5_iter-residual_lifetime-lb-velocity_degree"
        self._plot_init(
            title_center="",
            title_left="Residual Lifetime lower bound velocity over degree",
            title_right=self.time_distr_name,
            yscale='linear',
            xlabel="Degree",
            ylabel="Bound Velocity")
        self._plot_iter_residual_lifetime_lower_bound_velocity_over_degree_lines()
        self._plot_close(filename)

    def plot_iter_upper_bound_velocity_over_degree(self):
        filename = "5_iter-ub-velocity_degree"
        self._plot_init(
            title_center="",
            title_left="Upper bound velocity over degree",
            title_right=self.time_distr_name,
            yscale='linear',
            xlabel="Degree",
            ylabel="Bound Velocity")
        self._plot_iter_upper_bound_velocity_over_degree_lines()
        self._plot_close(filename)

    def plot_iter_don_bound_velocity_over_degree(self):
        filename = "5_iter-don-b-velocity_degree"
        self._plot_init(
            title_center="",
            title_left="Don bound velocity over degree",
            title_right=self.time_distr_name,
            yscale='linear',
            xlabel="Degree",
            ylabel="Bound Velocity")
        self._plot_iter_don_bound_velocity_over_degree_lines()
        self._plot_close(filename)

    def plot_iter_all_bounds_velocity_over_degrees_with_real_velocity(self):
        filename = "5_iter-all-b-velocity_degree"
        self._plot_init(
            title_center="",
            title_left="All bounds' velocities over degree",
            title_right=self.time_distr_name,
            yscale='linear',
            xlabel="Degree",
            ylabel="Velocity")
        self._plot_iter_memoryless_lower_bound_velocity_over_degree_lines()
        self._plot_iter_residual_lifetime_lower_bound_velocity_over_degree_lines()
        self._plot_iter_upper_bound_velocity_over_degree_lines()
        self._plot_iter_don_bound_velocity_over_degree_lines()
        self._plot_iter_real_velocity_over_degree_lines()
        self._plot_close(filename)

    # PUBLIC INTERFACE - END
