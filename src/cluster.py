import copy

from src import statistics
from src.mltoolbox import functions
from src.node import Node
from src.utils import *


class Cluster:
    """

    """

    def __init__(self, adjacency_matrix, graph_name="undefined", verbose=False):
        """
        Parameters
        ----------
        adjacency_matrix : ndarray
            Adjacency matrix of the dependency graph.
        """
        self.verbose = verbose
        self.verbose_node = False
        self.verbose_task = False
        self.graph_name = graph_name
        self.future_event_list = {}  # FEL of the discrete event simulator
        self.nodes = []  # computational unites (nodes) list
        self.adjacency_matrix = adjacency_matrix
        self.max_iter = None
        self.max_time = None
        self.clock = 0.0
        self.iteration = 0
        self.X = None  # keep whole training set examples
        self.y = None  # keep whole training set target function values
        self.real_y = None
        self.real_w = None
        self.w = []  # log 'iteration i-th' => value of weight vector at iteration i-th
        self.method = None
        self.obj_function = None
        self.average_model_toggle = None
        self.metrics = {}
        self.real_metrics = {}
        self.real_metrics_toggle = True
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
            real_y_activation_function,
            obj_function=METRICS["mse"],
            average_model_toggle=False,
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
            real_metrics_toggle=True,
            metrics_type=0,
            metrics_nodes='all',
            shuffle=True,
            time_distr_class=statistics.ExponentialDistribution,
            time_distr_param=(),
            time_const_weight=0,
            node_error_mean=0,
            node_error_std_dev=1,
            starting_weights_domain=(0, 5),
            verbose_node=False,
            verbose_task=False
    ):
        """Cluster setup.

        Parameters
        ----------
        X : ndarray of float
            Training set samples.

        y : array of floats
            Training set sample target function values.

        real_w : array of floats
            Real weight vector used to generate the training set.

        real_y_activation_function : function
            Activation function to call over the output of the prediction model (no more used).

        obj_function : class
            Class of the objective function to minimize.

        average_model_toggle : bool
            If True then the average model over time is used rather than just x(k) to compute metrics.

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

        real_metrics_toggle : bool


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

        verbose_node : bool or int
            Verbose policy in node class.
            - <0 : no print at all except from errors (unsafe).
            -  0 or False : default messages;
            -  1 : verbose + default messages
            -  2 : verbose + default messages + input required to continue after each message (simulation will be paused
                after each message and will require to press ENTER to go on, useful for debugging).

        verbose_task : bool or int
            Verbose policy in task class.

        Returns
        -------
        None
        """

        self.verbose_node = verbose_node
        self.verbose_task = verbose_task
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
        self.method = method
        self.metrics_type = metrics_type
        self.real_metrics_toggle = real_metrics_toggle
        self.average_model_toggle = average_model_toggle

        if not obj_function in METRICS:
            raise Exception("'{}' is not a viable objective function")

        self.obj_function = METRICS[obj_function]

        # Setup of nodes to take into account for metrics calculations
        if (metrics_nodes == 'worst' or metrics_nodes == 'best') and self.metrics_type != 2:
            metrics_nodes = 'all'

        if metrics_nodes != 'worst' and metrics_nodes != 'best':
            if not (isinstance(metrics_nodes, list) or isinstance(metrics_nodes, tuple)):
                if isinstance(metrics_nodes, int):
                    metrics_nodes = [metrics_nodes]
                else:
                    metrics_nodes = list(range(N))
            self.metrics_nodes_id = metrics_nodes
        else:
            self.metrics_nodes = metrics_nodes

        # METRICS SETUP BEGIN
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
            pass
            # todo: temp warnings.warn("No metrics specified")
            # warnings.warn("No metrics specified")

        # add objective function to metrics
        if not self.obj_function.id in self.metrics:
            self.metrics[self.obj_function.id] = self.obj_function

        # instantiate logs list for each metrics
        for mk in self.metrics.keys():
            self.logs["metrics"][mk] = []

        # METRICS SETUP END

        # REAL METRICS SETUP BEGIN
        if self.real_metrics_toggle:
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

            # add objective function to real metrics
            if not self.obj_function.id in self.real_metrics:
                self.real_metrics[self.obj_function.id] = self.obj_function

            # instantiate logs list for each real metrics
            for rmk in self.real_metrics.keys():
                self.logs["real_metrics"][rmk] = []

        # REAL METRICS SETUP END

        self.logs["obj_function"] = self.logs["metrics"][self.obj_function.id]
        if self.real_metrics_toggle:
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

        self.real_y = self.X.dot(self.real_w)

        if not real_y_activation_function is None:
            self.real_y = real_y_activation_function(self.real_y)

        time_distr_params_list = []
        if isinstance(time_distr_param[0], tuple) or isinstance(time_distr_param[0], list):
            for i in range(N):
                time_distr_params_list = time_distr_param
        else:
            for i in range(N):
                time_distr_params_list.append(time_distr_param)

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
            # todo: embody this permanently somewhere
            node_self_weight = 1 / sum(self.adjacency_matrix[i])
            # if 'expander' in self.graph_name or 'clique' in self.graph_name:
            #    node_self_weight = 0.5

            # instantiate new node for the just-selected subsample
            self.nodes.append(
                Node(
                    i,
                    node_self_weight,
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
                    self.real_metrics_toggle,
                    shuffle,
                    time_distr_class,
                    time_distr_params_list[i],
                    time_const_weight,
                    starting_weights_domain,
                    verbose_node,
                    verbose_task
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

    def _compute_avg_w(self):
        w = np.zeros(len(self.nodes[0].training_task.get_w()))
        for node in self.nodes:
            w += node.training_task.get_w_at_iteration(self.iteration, self.average_model_toggle)
        w /= len(self.nodes)

        if len(self.w) == self.iteration:
            self.w.append(w)
        elif len(self.w) == self.iteration + 1:
            self.w[self.iteration] = w
        else:
            raise Exception('Unexpected w(t) size')

    def _compute_all_metrics(self):
        if self.method is None:
            return
        self._compute_avg_w()
        for m in self.metrics:
            self._compute_metrics(m, real=False)
        if self.real_metrics_toggle:
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
            if self.metrics_nodes == 'worst':
                worst_val = metrics[m].compute_value(
                    self.X,
                    y,
                    self.nodes[0].training_task.get_w_at_iteration(self.iteration, self.average_model_toggle)
                )
                worst_node = 0
                for i in range(1, len(self.nodes)):
                    new_val = metrics[m].compute_value(
                        self.X,
                        y,
                        self.nodes[i].training_task.get_w_at_iteration(self.iteration, self.average_model_toggle)
                    )
                    if new_val > worst_val:
                        worst_val = new_val
                        worst_node = i
                val = worst_val
            elif self.metrics_nodes == 'best':
                best_val = metrics[m].compute_value(
                    self.X,
                    y,
                    self.nodes[0].training_task.get_w_at_iteration(self.iteration, self.average_model_toggle)
                )
                for i in range(1, len(self.nodes)):
                    best_val = min(best_val, metrics[m].compute_value(
                        self.X,
                        y,
                        self.nodes[i].training_task.get_w_at_iteration(self.iteration, self.average_model_toggle)
                    ))
                val = best_val
            else:
                for node in self.metrics_nodes:
                    val += metrics[m].compute_value(
                        self.X,
                        y,
                        node.training_task.get_w_at_iteration(self.iteration, self.average_model_toggle)
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

    def _top_event(self):
        keys = sorted(self.future_event_list.keys())

        if len(keys) == 0:
            return None

        key = keys[0]

        prev_len = len(self.future_event_list[key])
        e = self.future_event_list[key][0]

        if len(self.future_event_list[key]) != prev_len:
            raise Exception("Top operation on Cluster Future Event List unexpectedly removed an event from it")
        return e

    def get_next_event_time(self):
        next_event = self._top_event()
        if next_event is None:
            return math.inf
        return next_event['time']

    def run(self):
        """
        Run the cluster (distributed computation simulation).
        :return: None
        """

        for node in self.nodes:
            node.broadcast_weight_to_recipients()

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

        # dequeue the first event
        event = self.dequeue_event()

        # loop until future event list gets empty or upon reaching a stop condition
        while not stop_condition and not event is None:
            # the starting time of the event dequeued from the FEL is always the least starting
            # time among all events still in the queue so the cluster clock can align to such time
            self.clock = event["time"]

            print_verbose(self.verbose, "Event type='{}' for node=[{}] starting at CLOCK={}".format(
                event["type"], col(event["node"].get_id(), 'cyan'), col(np.around(self.clock, 4), 'green')
            ))

            if event["type"] in ("node_step", "node_endstep"):
                node = event["node"]

                if event['type'] == 'node_step':
                    if node.can_run():
                        # node can run, then do it
                        t0, tf = node.step()

                        # append (t0,tf) pair in dynamics log at index=node._id
                        self.logs["dynamics"][node.get_id()].append((t0, tf))

                        print_verbose(self.verbose, "Node [{}] run from {} to {}".format(
                            col(node.get_id(), 'cyan'),
                            col(np.around(t0, 4), 'green'),
                            col(np.around(tf, 4), 'green')
                        ))
                    else:
                        # node cannot run computation because it lacks some
                        # dependencies' information or it is in the end-step event
                        # so it is not supposed to do anything else
                        max_local_clock = node.local_clock

                        # todo perf-boost:
                        # - get the row of local_clocks matrix (iter x nodes) as an array
                        # - set to zero indices not in node.dependencies
                        # - get the max in the array

                        for dep in node.dependencies:
                            if dep.get_local_clock_by_iteration(node.iteration) > max_local_clock:
                                max_local_clock = max(max_local_clock, dep.local_clock)

                        # set the local_clock of the node equal to the value of the
                        # local_clock of the last dependency that ended the computation
                        # this node needs
                        node.set_local_clock(max_local_clock)

                        print_verbose(self.verbose, "Node [{}] cannot run, so node.clock is set from {} to {}".format(
                            col(node.get_id(), 'cyan'),
                            col(np.around(event['time'], 4), 'green'),
                            col(np.around(max_local_clock, 4), 'green')
                        ))

                    # Evaluate stop conditions of node
                    new_event_type = 'node_endstep'
                    if node.iteration < self.max_iter and node.local_clock < self.max_time:
                        # if none of the stop conditions is met then reset the event type to 'node_step'
                        new_event_type = 'node_step'
                    else:
                        # if at least one of the stop conditions is verified then the node
                        # have to stop running
                        node.is_running = False

                        # generate and print verbose output
                        stop_reason = ""
                        if node.iteration >= self.max_iter:
                            stop_reason += "node.iteration = {} >= {} = self.max_iter".format(
                                node.iteration, self.max_iter
                            )

                        if node.iteration >= self.max_iter and node.local_clock >= self.max_time:
                            stop_reason += " | "

                        if node.local_clock >= self.max_time:
                            stop_reason += "node.local_clock = {} >= {} = self.max_time".format(
                                node.local_clock, self.max_time
                            )

                        print_verbose(self.verbose, "Node [{}] has finished computation due to {}".format(
                            col(node.get_id(), 'cyan'),
                            stop_reason))

                    # create next event for this node
                    # this happens only if not in endstep event type
                    new_event = {
                        'time': node.local_clock,
                        'type': new_event_type,
                        'node': node
                    }

                    # enqueue the new event in the future event list
                    self.enqueue_event(new_event)

                # local clock of node has been updated, so now update the cluster clock
                # set it equal to the next event starting time
                next_event_time = self.get_next_event_time()

                if not math.isinf(next_event_time):
                    self.clock = next_event_time
                else:
                    # if next event time is math.inf then it means that the future event list is empty
                    # e.g. the simulation will end right after this loop

                    # todo performance boost:
                    # run max on last row of local clocks matrix
                    max_clock = -1
                    for _node in self.nodes:
                        if _node.local_clock > max_clock:
                            max_clock = _node.local_clock

                    # set cluster clock equal to the biggest local_clock among nodes
                    self.clock = max_clock

                # set local_clocks of all not-running nodes such that they follow
                # the cluster clock (to avoid strange behaviours)
                for _node in self.nodes:
                    if not _node.is_running and _node.local_clock < self.clock:
                        _node.local_clock = self.clock

                # todo: remove following snippet because it just slows down the computation
                # check whether one node's local_clock has been left behind the cluster clock
                # that is completely unexpected, so in such case raise an exception
                """min_clock = math.inf
                for _node in self.nodes:
                    if _node.local_clock < min_clock:
                        min_clock = _node.local_clock
                if min_clock < self.clock:
                    raise Exception("Cluster clock is higher than clock of some node and this is a "
                                    "completely unexpected behaviour")
                """

                # the node step has been almost completed, now check if all the others
                # have already performed the iteration i-th, if so then the global
                # iteration i-th has been completed and the completion time for such
                # iteration is the actual local_clock of this node

                min_iter = math.inf

                for _node in self.nodes:
                    _node_iter = _node.iteration
                    if _node_iter < min_iter:
                        min_iter = _node_iter

                if min_iter == self.iteration + 1:
                    self.iteration += 1

                    last_to_complete_iteration_clock = -1
                    for __node in self.nodes:
                        node_clock_at_iter = __node.get_local_clock_by_iteration(self.iteration)
                        if node_clock_at_iter > last_to_complete_iteration_clock:
                            last_to_complete_iteration_clock = node_clock_at_iter

                    self.logs["iter_time"].append(last_to_complete_iteration_clock)
                    self.logs["max_iter_time"].append((self.clock, min_iter))
                    self.logs["avg_iter_time"].append((self.clock, min_iter))

                    self._compute_all_metrics()

                    print_verbose(
                        self.verbose,
                        "Cluster ITER++. min_iter={}, clock={}".format(
                            min_iter, np.around(self.clock, 2)
                        )
                    )

                    if self.verbose <= 0:
                        sys.stdout.write('\x1b[2K')
                        sys.stdout.write(self._step_output() + "\r")
                        sys.stdout.flush()
                    else:
                        print_verbose(self.verbose, self._step_output())

                    """max_error = -math.inf
                    for _node in self.nodes:
                        node_error = _node.training_task.mean_squared_error_log[self.iteration]
                        if node_error > max_error:
                            max_error = node_error
                    print("MAX ERROR: {}".format(max_error))"""

                elif min_iter > self.iteration + 1:
                    raise Exception("Unexpected behaviour of cluster distributed dynamics")

                # check for the stop condition
                stop_condition = False
                if self.iteration >= self.max_iter:
                    stop_condition = True
                    print(self._step_output())
                    print("Cluster stopped due to global iteration (={}) being equal to max_iter (={})".format(
                        self.iteration, self.max_iter))
                elif self.clock >= self.max_time:
                    stop_condition = True
                    print(self._step_output())
                    print("Cluster stopped due to global clock (={}) being grater than or equal to max_time (={})".
                        format(self.clock, self.max_time))
                elif (not self.method is None) and self.get_obj_function_value() <= self.epsilon:
                    stop_condition = True
                    print(self._step_output())
                    print("Cluster stopped due to error (={}) being less than or equal to epsilon (={})".format(
                        self.get_obj_function_value(),
                        self.epsilon
                    ))

            elif event["type"] == "":
                pass
            else:
                pass

            event = self.dequeue_event()

        min_iter = math.inf
        max_iter = -1
        avg_iter = 0

        if self.max_time is None or math.isinf(self.max_time):
            last_time = self.clock
        else:
            last_time = self.max_time

        for _node in self.nodes:
            _node_iter = _node.get_iteration_at_local_clock(last_time)
            if _node_iter < min_iter:
                min_iter = _node_iter
            if _node_iter > max_iter:
                max_iter = _node_iter
            avg_iter += _node_iter
        avg_iter /= len(self.nodes)

        self.logs["max_iter_time"][-1] = (last_time, max_iter)
        self.logs["avg_iter_time"][-1] = (last_time, avg_iter)

        # todo: temp
        print("Cluster simulation run ended at iter={} and clock={}".format(
            self.iteration, self.clock
        ))

        print_verbose(self.verbose, "Cluster simulation run ended at iter={} and clock={}".format(
            self.iteration, self.clock
        ))

    def run_stale(self):
        """
        Run the cluster (distributed computation simulation).
        :return: None
        """

        for node in self.nodes:
            node.broadcast_weight_to_recipients()

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

        # dequeue the first event
        event = self.dequeue_event()

        # loop until future event list gets empty or upon reaching a stop condition
        while not stop_condition and not event is None:
            # the starting time of the event dequeued from the FEL is always the least starting
            # time among all events still in the queue so the cluster clock can align to such time
            self.clock = event["time"]

            print_verbose(self.verbose, "Event type='{}' for node=[{}] starting at CLOCK={}".format(
                event["type"], col(event["node"].get_id(), 'cyan'), col(np.around(self.clock, 4), 'green')
            ))

            if event["type"] in ("node_step", "node_endstep"):
                node = event["node"]

                if event['type'] == 'node_step':
                    if node.can_run():
                        # node can run, then do it
                        t0, tf = node.step()

                        # append (t0,tf) pair in dynamics log at index=node._id
                        self.logs["dynamics"][node.get_id()].append((t0, tf))

                        print_verbose(self.verbose, "Node [{}] run from {} to {}".format(
                            col(node.get_id(), 'cyan'),
                            col(np.around(t0, 4), 'green'),
                            col(np.around(tf, 4), 'green')
                        ))
                    else:
                        # node cannot run computation because it lacks some
                        # dependencies' informations or it is in the endstep event
                        # so it is not supposed to do anything else
                        max_local_clock = node.local_clock

                        # todo perf-boost:
                        # - get the row of local_clocks matrix (iter x nodes) as an array
                        # - set to zero indices not in node.dependencies
                        # - get the max in the array

                        for dep in node.dependencies:
                            if dep.get_local_clock_by_iteration(node.iteration) > max_local_clock:
                                max_local_clock = max(max_local_clock, dep.local_clock)

                        # set the local_clock of the node equal to the value of the
                        # local_clock of the last dependency that ended the computation
                        # this node needs
                        node.set_local_clock(max_local_clock)

                        print_verbose(self.verbose, "Node [{}] cannot run, so node.clock is set from {} to {}".format(
                            col(node.get_id(), 'cyan'),
                            col(np.around(event['time'], 4), 'green'),
                            col(np.around(max_local_clock, 4), 'green')
                        ))

                    # Evaluate stop conditions of node
                    new_event_type = 'node_endstep'
                    if node.iteration < self.max_iter and node.local_clock < self.max_time:
                        # if none of the stop conditions is met then reset the event type to 'node_step'
                        new_event_type = 'node_step'
                    else:
                        # if at least one of the stop conditions is verified then the node
                        # have to stop running
                        node.is_running = False

                        # generate and print verbose output
                        stop_reason = ""
                        if node.iteration >= self.max_iter:
                            stop_reason += "node.iteration = {} >= {} = self.max_iter".format(
                                node.iteration, self.max_iter
                            )

                        if node.iteration >= self.max_iter and node.local_clock >= self.max_time:
                            stop_reason += " | "

                        if node.local_clock >= self.max_time:
                            stop_reason += "node.local_clock = {} >= {} = self.max_time".format(
                                node.local_clock, self.max_time
                            )

                        print_verbose(self.verbose, "Node [{}] has finished computation due to {}".format(
                            col(node.get_id(), 'cyan'),
                            stop_reason))

                    # create next event for this node
                    # this happens only if not in endstep event type
                    new_event = {
                        'time': node.local_clock,
                        'type': new_event_type,
                        'node': node
                    }

                    # enqueue the new event in the future event list
                    self.enqueue_event(new_event)

                # local clock of node has been updated, so now update the cluster clock
                # set it equal to the next event starting time
                next_event_time = self.get_next_event_time()

                if not math.isinf(next_event_time):
                    self.clock = next_event_time
                else:
                    # if next event time is math.inf then it means that the future event list is empty
                    # e.g. the simulation will end right after this loop

                    # todo performance boost:
                    # rum max on last row of local clocks matrix
                    max_clock = -1
                    for _node in self.nodes:
                        if _node.local_clock > max_clock:
                            max_clock = _node.local_clock

                    # set cluster clock equal to the biggest local_clock among nodes
                    self.clock = max_clock

                # set local_clocks of all not-running nodes such that they follow
                # the cluster clock (to avoid strange behaviours)
                for _node in self.nodes:
                    if not _node.is_running and _node.local_clock < self.clock:
                        _node.local_clock = self.clock

                # todo: remove following snippet because it just slows down the computation
                # check whether one node's local_clock has been left behind the cluster clock
                # that is completely unexpected, so in such case raise an exception
                """min_clock = math.inf
                for _node in self.nodes:
                    if _node.local_clock < min_clock:
                        min_clock = _node.local_clock
                if min_clock < self.clock:
                    raise Exception("Cluster clock is higher than clock of some node and this is a "
                                    "completely unexpected behaviour")
                """

                # the node step has been almost completed, now check if all the others
                # have already performed the iteration i-th, if so then the global
                # iteration i-th has been completed and the completion time for such
                # iteration is the actual local_clock of this node

                min_iter = math.inf
                max_iter = -1
                avg_iter = 0

                for _node in self.nodes:
                    _node_iter = _node.get_iteration_at_local_clock(self.clock)
                    if _node_iter < min_iter:
                        min_iter = _node_iter
                    if _node_iter > max_iter:
                        max_iter = _node_iter
                    avg_iter += _node_iter
                avg_iter /= len(self.nodes)

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

                    print_verbose(
                        self.verbose,
                        "Cluster ITER++. min_iter={}, avg_iter={}, max_iter={}, clock={}".format(
                            min_iter, avg_iter, max_iter, np.around(self.clock, 2)
                        )
                    )

                    self.logs["avg_iter_time"].append((self.clock, avg_iter))

                    self._compute_all_metrics()

                    if self.verbose <= 0:
                        sys.stdout.write('\x1b[2K')
                        sys.stdout.write(self._step_output() + "\r")
                        sys.stdout.flush()
                    else:
                        print_verbose(self.verbose, self._step_output())

                    """max_error = -math.inf
                    for _node in self.nodes:
                        node_error = _node.training_task.mean_squared_error_log[self.iteration]
                        if node_error > max_error:
                            max_error = node_error
                    print("MAX ERROR: {}".format(max_error))"""

                elif min_iter > self.iteration + 1:
                    raise Exception("Unexpected behaviour of cluster distributed dynamics")

                # check for the stop condition
                stop_condition = False
                if self.iteration >= self.max_iter:
                    stop_condition = True
                    print(self._step_output())
                    print("Cluster stopped due to global iteration (={}) being equal to max_iter (={})".format(
                        self.iteration, self.max_iter))
                elif self.clock >= self.max_time:
                    stop_condition = True
                    print(self._step_output())
                    print("Cluster stopped due to global clock (={}) being grater than or equal to max_time (={})".
                        format(self.clock, self.max_time))
                elif (not self.method is None) and self.get_obj_function_value() <= self.epsilon:
                    stop_condition = True
                    print(self._step_output())
                    print("Cluster stopped due to error (={}) being less than or equal to epsilon (={})".format(
                        self.get_obj_function_value(),
                        self.epsilon
                    ))

            elif event["type"] == "":
                pass
            else:
                pass

            event = self.dequeue_event()

        print_verbose(self.verbose, "Cluster simulation run ended at iter={} and clock={}".format(
            self.iteration, self.clock
        ))

    def _step_output(self):
        output = "[{}] >>> ".format(self.graph_name.upper())

        if not self.max_time is None:
            output += "Time: ({}/{}) ".format(
                col(np.around(self.clock, 2), 'yellow'),
                col(self.max_time, 'blue')
            )

        if not self.max_iter is None:
            output += "Iter: ({}/{}) ".format(
                col(self.iteration, 'yellow'),
                col(self.max_iter, 'blue')
            )

        if not self.method is None:
            for m in self.metrics:
                try:
                    mval = np.around(self.get_metrics_value(m), 2)
                except OverflowError:
                    mval = self.get_metrics_value(m)
                output += "{}={} ".format(
                    m,
                    col(mval, 'red')
                )

            for rm in self.real_metrics:
                try:
                    rmval = np.around(self.get_metrics_value(rm, real=True), 2)
                except OverflowError:
                    rmval = self.get_metrics_value(rm, real=True)
                output += "real-{}={} ".format(
                    rm,
                    col(rmval, 'red')
                )

        return output
