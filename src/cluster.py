import copy, types, warnings
from src import mltoolbox, statistics
from src.mltoolbox.metrics import METRICS
from src.mltoolbox import functions
from src.functions import *
from src.node import Node


class Cluster:
    """

    """

    def __init__(self, adjacency_matrix, graph_name="undefined"):
        """
        Parameters
        ----------
        adjacency_matrix : ndarray
            Adjacency matrix of the dependency graph.
        """
        self.graph_name = graph_name
        self.future_event_list = {}  # FEL of the discrete event simulator
        self.nodes = []  # computational unites (nodes) list
        self.adjacency_matrix = adjacency_matrix
        self.max_iter = None
        self.max_time = None
        self.clock = 0
        self.iteration = 0
        self.X = None  # keep whole training set examples
        self.y = None  # keep whole training set target function values
        self.real_y = None
        self.real_w = None
        self.w = []  # log 'iteration i-th' => value of weight vector at iteration i-th
        self.obj_function = None
        self.metrics = {}
        self.real_metrics = {}
        self.metrics_type = 0
        self.metrics_nodes_id = []
        self.metrics_nodes = []

        self.logs = {
            "obj_function": [],
            "real_obj_function": [],
            "dynamics": [],
            "iter_time": [0.0],
            "avg_iter_time": [(0.0, 0.0)],
            "max_iter_time": [(0.0, 0.0)],
            "metrics": {},
            "real_metrics": {}
        }

        self.epsilon = 0.0  # acceptance threshold
        self.linear_regression_beta = None

    def setup(self, X, y, real_w,
              obj_function=METRICS["mse"],
              method="classic",
              max_iter=None,
              max_time=None,
              batch_size=5,
              dual_averaging_radius=10,
              epsilon=0.0,
              alpha=1e-4,
              learning_rate="constant",
              metrics="all",
              real_metrics="all",
              metrics_type=0,
              metrics_nodes='all',
              shuffle=True,
              verbose=False,
              time_distr_class=statistics.ExponentialDistribution,
              time_distr_param=(),
              time_const_weight=0,
              node_error_mean=0,
              node_error_std_dev=1,
              starting_weights_domain=(0, 5)):
        """Cluster setup.

        Parameters
        ----------
        X : ndarray of float
            Training set samples.

        y : array of float
            Training set sample target function values.

        real_w : array of float
            Real weight vector used to generate the training set.

        obj_function : class
            Class of the objective function to minimize.

        method : 'classic', 'stochastic', 'batch' or 'linear_regression', optional
            Method used to minimize error function.

        max_iter : int, None or math.inf, optional
            Maximum number of iteration after which the cluster terminates.
            None and math.inf indicates that the cluster will never stops due to
            iterations number.

        max_time : int, None or math.inf, optional
            Maximum clock time value after which the cluster terminates.
            None and math.inf indicates that the cluster will never stops due to
            time.

        batch_size : int, optional
            Batch size only taken into account for method = 'batch'.

        dual_averaging_radius: float, optional
            Radius r of circle used in dual averaging method.

        epsilon : float, optional
            Error acceptance threshold under which the cluster terminates the execution.

        alpha : float, optional
            Gradient descent step size alpha coefficient.

        learning_rate : str, optional
            Learning rate type (actually not exploited yet).

        metrics : str or list of str, optional
            Choose any in ['all', 'score', 'mse', 'mae', rmse']
            or 'all'.

        real_metrics : str or list of str, optional
            Choose any in ['all', 'score', 'mse', 'mae', rmse']
            or 'all'.

        metrics_type : int, optional
            0 : normal metric, 1 : alt metric, 2 : new alt metric.

        metrics_nodes : int or list of int or anything else, optional
            Node considered in the computation of metrics.
            If anything else then all nodes will be taken into account.

        shuffle : bool, optional
            True if the cluster may shuffle the training set before splitting it (always suggested).

        verbose : bool, optional
            Not exploited yet.

        time_distr_class : class, optional
            Statistic distribution class from src.statistics.

        time_distr_param : list, optional
            Params' list to pass to distribution sample method.

        time_const_weight : float, optional
            Weight of constant part in computation time of nodes Time(t) = c * E[X] + (1-c) * X(t).

        node_error_mean : float, optional
            Mean of the error inside nodes.

        node_error_std_dev : float, optional
            Standard deviation of error inside nodes.

        starting_weights_domain : tuple of two floats
            Domain extremes of nodes' starting weights.

        Returns
        -------
        None

        """

        N = self.adjacency_matrix.shape[0]

        # if X and y have different sizes then the training set is bad formatted
        if len(y) != X.shape[0]:
            raise Exception("X has different amount of rows than y ({} != {})".format(X.shape[0], len(y)))

        if epsilon is max_iter is max_time is None:
            warnings.warn("None of epsilon, max_iter and max_time is set, the Cluster will never stop but due to a"
                          "KeyboardInterrupt (CTRL+C). Be careful!")

        if epsilon is None:
            epsilon = 0.0

        if max_iter is None:
            max_iter = math.inf

        if max_time is None:
            max_time = math.inf

        self.max_iter = max_iter
        self.max_time = max_time
        self.epsilon = epsilon
        self.metrics_type = metrics_type

        if not obj_function in METRICS:
            raise Exception("'{}' is not a viable objective function")

        self.obj_function = METRICS[obj_function]

        # Setup of nodes to take into account for metrics calculations
        if not (isinstance(metrics_nodes, list) or isinstance(metrics_nodes, tuple)):
            if isinstance(metrics_nodes, int):
                metrics_nodes = [metrics_nodes]
            else:
                metrics_nodes = list(range(N))
        self.metrics_nodes_id = metrics_nodes

        # Fill self.metrics with instances of metrics objects
        if not (isinstance(metrics, list) or isinstance(metrics, tuple)):
            if metrics in METRICS:
                self.metrics[metrics] = METRICS[metrics]
            elif metrics.lower() == 'all':
                self.metrics = METRICS
        else:
            for m in metrics:
                if m in METRICS:
                    self.metrics[m] = METRICS[m]
                else:
                    warnings.warn("Metric {} does not exists".format(m))
        if len(self.metrics) == 0:
            warnings.warn("No metrics specified")

        # Fill self.real_metrics with instances of metrics objects
        if not (isinstance(real_metrics, list) or isinstance(real_metrics, tuple)):
            if real_metrics in METRICS:
                self.real_metrics[real_metrics] = METRICS[real_metrics]
            elif real_metrics.lower() == 'all':
                self.real_metrics = METRICS
        else:
            for m in real_metrics:
                if m in METRICS:
                    self.real_metrics[m] = METRICS[m]
                else:
                    warnings.warn("Metric {} does not exists".format(m))
        if len(self.real_metrics) == 0:
            warnings.warn("No real_metrics specified")

        # add objective function to metrics and real metrics
        if not self.obj_function.id in self.metrics:
            self.metrics[self.obj_function.id] = self.obj_function
        if not self.obj_function.id in self.real_metrics:
            self.real_metrics[self.obj_function.id] = self.obj_function

        # instantiate logs list for each metrics and real_metrics
        for mk in self.metrics.keys():
            self.logs["metrics"][mk] = []
        for rmk in self.real_metrics.keys():
            self.logs["real_metrics"][rmk] = []

        self.logs["obj_function"] = self.logs["metrics"][self.obj_function.id]
        self.logs["real_obj_function"] = self.logs["real_metrics"][self.obj_function.id]

        self.X = X
        self.y = y
        self.real_w = real_w

        del y, X

        if shuffle:
            Xy = np.c_[self.X, self.y]
            np.random.shuffle(Xy)
            self.X = np.delete(Xy, -1, 1)
            self.y = np.take(Xy, -1, 1)
            del Xy

        self.real_y = self.obj_function.y_hat_func.compute_value(self.X, self.real_w)

        nodes_errors = np.random.normal(node_error_mean, node_error_std_dev, N)
        subX_size = int(math.floor(self.X.shape[0] / N))
        subX_size_res = self.X.shape[0] % N
        prev_end = 0
        for i in range(N):
            # size of the subsample of the training set that will be assigned to
            # this node
            node_X_size = subX_size
            if subX_size_res > 0:
                node_X_size += 1
                subX_size_res -= 1
            beg = prev_end
            end = beg + node_X_size

            if not (node_error_mean == node_error_std_dev == 0):
                for j in range(beg, end):
                    self.y[j] += nodes_errors[i]

            # assign the correct subsample to this node
            node_X = copy.deepcopy(self.X[beg:end])  # instances
            node_y = copy.deepcopy(self.y[beg:end])  # oracle outputs

            """if np.sum(self.adjacency_matrix[0]) == 1:
                method = "linear_regression"
            """
            # instantiate new node for the just-selected subsample
            self.nodes.append(
                Node(
                    i,
                    node_X,
                    node_y,
                    self.real_w,
                    self.obj_function,
                    method,
                    batch_size,
                    dual_averaging_radius,
                    alpha,
                    learning_rate,
                    self.metrics,
                    self.real_metrics,
                    shuffle,
                    verbose,
                    time_distr_class,
                    time_distr_param,
                    time_const_weight,
                    starting_weights_domain
                ))
            self.logs["dynamics"].append([])

            prev_end = end

        prev_size = self.nodes[0].training_task.X.shape[0]
        for u in range(1, len(self.nodes)):
            size = self.nodes[u].training_task.X.shape[0]
            if not (size == prev_size or size == prev_size - 1):
                raise Exception(
                    "Wrong dataset allocation among nodes: at least one node has a training subset "
                    "too big or too small wrt the others")

        if prev_end != self.X.shape[0]:
            raise Exception(
                "Wrong dataset allocation among nodes: the total amount of samples in nodes "
                "is different from the total size of the training set")

        # set up all nodes' dependencies following the adjacency_matrix
        for i in range(N):
            for j in range(self.adjacency_matrix.shape[1]):
                if i != j and self.adjacency_matrix[i, j] == 1:
                    self.nodes[j].add_dependency(self.nodes[i])
                    self.nodes[i].add_recipient(self.nodes[j])

        for i in self.metrics_nodes_id:
            self.metrics_nodes.append(self.nodes[i])

        self.linear_regression_beta = functions.estimate_linear_regression_beta(self.X, self.y)
        self._compute_all_metrics()

    def get_w_at_iteration(self, iteration):
        return np.copy(self.w[iteration])

    def get_w(self):
        return np.copy(self.w[-1])

    def get_obj_function_value(self):
        return self.logs["obj_function"][-1]

    def get_metrics_value(self, met, real=False):
        if real:
            return self.logs["real_metrics"][met][-1]
        return self.logs["metrics"][met][-1]

    def compute_avg_w(self):
        w = np.zeros(len(self.nodes[0].training_task.get_w()))
        for node in self.nodes:
            w += node.training_task.get_w_at_iteration(self.iteration)
        w /= len(self.nodes)

        if len(self.w) == self.iteration:
            self.w.append(w)
        elif len(self.w) == self.iteration + 1:
            self.w[self.iteration] = w
        else:
            raise Exception('Unexpected w(t) size')

    def _compute_all_metrics(self):
        for m in self.metrics:
            self._compute_metrics(m, real=False)
        for rm in self.real_metrics:
            self._compute_metrics(rm, real=True)

    def _compute_metrics(self, m, real=False):
        if not real:
            y = self.y
            metrics = self.metrics
            metrics_log = self.logs["metrics"]
        else:
            y = self.real_y
            metrics = self.real_metrics
            metrics_log = self.logs["real_metrics"]

        val = 0
        if self.metrics_type == 1:
            # average of local metrics in nodes
            for node in self.metrics_nodes:
                val += node.training_task.get_metrics_value_at_iteration(m, self.iteration, real=real)
            val /= len(self.metrics_nodes)
        elif self.metrics_type == 2:
            # average of metrics computed used local w_u of nodes in self.metrics_nodes
            # on the whole training set
            for node in self.metrics_nodes:
                val += metrics[m].compute_value(
                    self.X,
                    y,
                    node.training_task.get_w_at_iteration(self.iteration)
                )
            val /= len(self.metrics_nodes)
        else:
            val = metrics[m].compute_value(
                self.X,
                y,
                self.get_w_at_iteration(self.iteration)
            )

        if len(metrics_log[m]) == self.iteration:
            metrics_log[m].append(val)
        elif len(metrics_log[m]) == self.iteration + 1:
            metrics_log[m][self.iteration] = val
            warnings.warn("Unexpected behaviour: metrics {} already computed in cluster".format(m))
        else:
            raise Exception('Unexpected metrics {} log size'.format(m))

        if math.isnan(val) or math.isinf(val):
            raise Exception("Computation has diverged to infinite")

        return val

    def enqueue_event(self, e):
        """
        Enqueue a new event in the future_event_list.
        :param e: Event dict
        :return: None
        """
        if not e["time"] in self.future_event_list:
            self.future_event_list[e["time"]] = []
        self.future_event_list[e["time"]].append(e)

    def dequeue_event(self):
        """
        Dequeue the event with the highest priority in the future_event_list,
        that is the one with the smallest time field value.
        :return: Event
        :rtype: dict
        """
        keys = sorted(self.future_event_list.keys())

        if len(keys) == 0:
            return None

        key = keys[0]

        e = self.future_event_list[key].pop(0)
        if len(self.future_event_list[key]) == 0:
            del self.future_event_list[key]
        return e

    def run(self):
        """
        Run the cluster (distributed computation simulation).
        :return: None
        """
        if len(self.nodes) == 0:
            raise Exception("Cluster has not been set up before calling run method")

        # enqueue one step event for each node in the cluster
        for _node in self.nodes:
            self.enqueue_event({
                'time': _node.local_clock,
                'type': 'node_step',
                'node': _node
            })

        stop_condition = False  # todo: stop condition (tolerance)
        event = self.dequeue_event()
        while not stop_condition and not event is None:

            prev_clock = self.clock
            self.clock = event["time"]

            if event["type"] in ("node_step", "node_endstep"):
                node = event["node"]

                if node.can_run() and event["type"] == "node_step":
                    self.logs["dynamics"][node.get_id()].append(node.step())

                    # when this node finishes iteration "i", it checks if all the others
                    # have already performed the iteration i-th, if so then the global
                    # iteration i-th has been completed and the completion time for such
                    # iteration is the actual local_clock of this node

                    min_iter = math.inf
                    max_iter = -1

                    for _node in self.nodes:
                        if _node.iteration < min_iter:
                            min_iter = _node.iteration
                        if _node.iteration > max_iter:
                            max_iter = _node.iteration

                    if max_iter > self.logs["max_iter_time"][-1][1]:
                        self.logs["max_iter_time"].append((self.clock, max_iter))

                    if min_iter == self.iteration + 1:
                        self.iteration += 1

                        last_to_complete_iteration_clock = -1
                        for __node in self.nodes:
                            node_clock_at_iter = __node.get_local_clock_by_iteration(self.iteration)
                            if node_clock_at_iter > last_to_complete_iteration_clock:
                                last_to_complete_iteration_clock = node_clock_at_iter

                        self.logs["iter_time"].append(last_to_complete_iteration_clock)

                        avg_iter = 0
                        for __node in self.nodes:
                            avg_iter += __node.get_iteration_at_local_clock(self.clock)
                        avg_iter /= len(self.nodes)

                        self.logs["avg_iter_time"].append((self.clock, avg_iter))

                        self._compute_all_metrics()

                        """max_error = -math.inf
                        for _node in self.nodes:
                            node_error = _node.training_task.mean_squared_error_log[self.iteration]
                            if node_error > max_error:
                                max_error = node_error
                        print("MAX ERROR: {}".format(max_error))"""

                    elif min_iter > self.iteration + 1:
                        raise Exception("Unexpected behaviour of cluster distributed dynamics")
                else:
                    # node cannot run computation because it lacks some
                    # dependencies' informations or it is in the endstep event
                    # so it is not supposed to do anything else
                    max_local_clock = node.local_clock
                    for dep in node.dependencies:
                        if dep.get_local_clock_by_iteration(node.iteration) > max_local_clock:
                            max_local_clock = max(max_local_clock, dep.local_clock)

                    # set the local_clock of the node equal to the value of the
                    # local_clock of the last dependency that ended the computation
                    # this node needs
                    node.set_local_clock(max_local_clock)

                output = "[{}] >>> ".format(self.graph_name.upper())

                if not self.max_time is None:
                    output += "Time: ({}/{}) ".format(int(self.clock * 100) / 100, self.max_time)

                if not self.max_iter is None:
                    output += "Iter: ({}/{}) ".format(self.iteration, self.max_iter)

                for m in self.metrics:
                    try:
                        mval = int(self.get_metrics_value(m) * 100) / 100
                    except OverflowError:
                        mval = self.get_metrics_value(m)
                    output += "{}={} ".format(m, mval)

                for rm in self.real_metrics:
                    try:
                        rmval = int(self.get_metrics_value(rm, real=True) * 100) / 100
                    except OverflowError:
                        rmval = self.get_metrics_value(rm, real=True)
                    output += "r{}={} ".format(rm, rmval)

                sys.stdout.write('\x1b[2K')
                sys.stdout.write(output + "\r")
                sys.stdout.flush()

                # check for the stop condition
                stop_condition = False
                if self.iteration >= self.max_iter:
                    stop_condition = True
                    print(output)
                    print("Cluster stopped due to global iteration (={}) being equal to max_iter (={})".format(
                        self.iteration, self.max_iter))

                if self.clock >= self.max_time:
                    stop_condition = True
                    print(output)
                    print("Cluster stopped due to global clock (={}) being grater than or equal to max_time (={})".
                          format(self.clock, self.max_time))

                if self.get_obj_function_value() <= self.epsilon:
                    stop_condition = True
                    print(output)
                    print("Cluster stopped due to error (={}) being less than or equal to epsilon (={})".format(
                        self.get_obj_function_value(),
                        self.epsilon
                    ))

                # enqueue the next event for current node
                if not stop_condition and event['type'] != 'node_endstep':
                    new_event_type = 'node_endstep'
                    if node.iteration < self.max_iter and node.local_clock < self.max_time:
                        new_event_type = 'node_step'

                    new_event = {
                        'time': node.local_clock,
                        'type': new_event_type,
                        'node': node
                    }
                    self.enqueue_event(new_event)

            elif event["type"] == "":
                pass
            else:
                pass

            event = self.dequeue_event()
