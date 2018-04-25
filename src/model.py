import copy, math, random, time, types, sys, warnings
import numpy as np
from src import mltoolbox


class Cluster:
    """

    """

    def __init__(self, adjacency_matrix):
        """
        Parameters
        ----------
        adjacency_matrix : ndarray
            Adjacency matrix of the dependency graph.
        """
        self.future_event_list = {}  # FEL of the discrete event simulator
        self.nodes = []  # computational unites (nodes) list
        self.adjacency_matrix = adjacency_matrix
        self.max_iter = None
        self.max_time = None
        self.clock = 0
        self.iteration = 0
        self.X = None  # keep whole training set examples
        self.y = None  # keep whole training set target function values
        self.dynamic_log = []  # log containing for each iteration i a pair (t0_i, tf_i)
        self.W_log = []  # log 'iteration i-th' => value of weight vector at iteration i-th
        self.iterations_time_log = []  # 'iteration i-th' => clock value at which i-th iteration has been completed
        self.global_mean_absolute_error_log = []  # 'iteration' => global MAE
        self.global_mean_squared_error_log = []  # 'iteration' => global MSE
        self.global_real_mean_squared_error_log = []  # 'iteration' => global RMSE
        self.epsilon = 0.0  # acceptance threshold
        self.alt_metrics = False  # use alternative metrics (almost obsolete)

    def setup(self, X, y, y_hat, method="stochastic", max_iter=None, max_time=None, batch_size=5, activation_func=None,
              loss=mltoolbox.SquaredLossFunction, penalty='l2', epsilon=0.0, alpha=0.0001, learning_rate="constant",
              metrics="all", alt_metrics=False, shuffle=True, verbose=False):
        """Cluster setup.

        Parameters
        ----------
        X : ndarray of float
            Training set samples.

        y : array of float
            Training set sample target function values.

        y_hat : class inheriting from YHatFunctionAbstract
            Class inherited from YHatFunctionAbstract.

        method : 'classic', 'stochastic' or 'batch', optional
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

        activation_func : None or function, optional
            Activation function applied to normalized output of target function or y hat.

        loss : class inheriting from LossFunctionAbstract, optional

        penalty : str, optional
            Penalty applied to avoid overfitting (actually not used at the moment).

        epsilon : float, optional
            Error acceptance threshold under which the cluster terminates the execution.

        alpha : float, optional
            Gradient descent step size alpha coefficient.

        learning_rate : str, optional
            Learning rate type (actually not exploited yet).

        metrics : str or list of str, optional
            Choose any in ['all', 'score', 'mean_absolute_error', 'mean_squared_error', real_mean_squared_error']
            or 'all'.

        alt_metrics : bool, optional
            Use old MSE computation method.

        shuffle : bool, optional
            True if the cluster may shuffle the training set before splitting it (always suggested).

        verbose : bool, optional
            Not exploited yet.

        Returns
        -------
        None

        """
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
        self.alt_metrics = alt_metrics
        self.X = X
        self.X = np.c_[np.ones((X.shape[0])), X]
        self.y = y

        if shuffle:
            Xy = np.c_[X, y]
            np.random.shuffle(Xy)
            X = np.delete(Xy, -1, 1)
            y = np.take(Xy, -1, 1)
            del Xy

        N = self.adjacency_matrix.shape[0]
        for i in range(N):
            # size of the subsample of the training set that will be assigned to
            # this node
            node_X_size = math.floor(X.shape[0] / (N - i))

            # assign the correct subsample to this node
            node_X = copy.deepcopy(X[0:node_X_size])  # instances
            node_y = copy.deepcopy(y[0:node_X_size])  # oracle outputs

            # instantiate new node for the just-selected subsample
            self.nodes.append(Node(i, node_X, node_y, y_hat, method, batch_size, activation_func, loss, penalty, alpha,
                                   learning_rate, metrics, shuffle, verbose))
            self.dynamic_log.append([])

            # evict the just-already-assigned samples of the training-set
            X = X[node_X_size:]
            y = y[node_X_size:]

        # set up all nodes' dependencies following the adjacency_matrix
        for i in range(N):
            for j in range(self.adjacency_matrix.shape[1]):
                if i != j and self.adjacency_matrix[i, j] == 1:
                    self.nodes[j].add_dependency(self.nodes[i])
                    self.nodes[i].add_recipient(self.nodes[j])

    def get_avg_W(self, index=-1):
        if index < len(self.W_log) > 0:
            return self.W_log[index]
        else:
            raise Exception("No weight vector defined")

    def get_global_mean_absolute_error(self, index=-1):
        if index < len(self.global_mean_absolute_error_log) > 0:
            return self.global_mean_absolute_error_log[index]
        else:
            return math.inf

    def get_global_mean_squared_error(self, index=-1):
        if index < len(self.global_mean_squared_error_log) > 0:
            return self.global_mean_squared_error_log[index]
        else:
            return math.inf

    def get_global_real_mean_squared_error(self, index=-1):
        if index < len(self.global_real_mean_squared_error_log) > 0:
            return self.global_real_mean_squared_error_log[index]
        else:
            return math.inf

    def compute_global_score(self):
        # todo
        pass

    def compute_avg_W(self):
        W = np.zeros(len(self.nodes[0].training_task.W))
        for node in self.nodes:
            W += node.training_task.W_log[self.iteration]
        W /= len(self.nodes)

        if len(self.W_log) == self.iteration:
            self.W_log.append(W)
        elif len(self.W_log) == self.iteration + 1:
            self.W_log[self.iteration] = W
        else:
            raise Exception('Unexpected W_log size')

    def compute_global_mean_absolute_error(self):
        gmae = 0
        if not self.alt_metrics:
            W = self.W_log[self.iteration]
            N = self.X.shape[0]
            mltask = self.nodes[0].training_task
            predictions = mltask.activation_func(mltask.y_hat.f(self.X, W))
            linear_error = np.absolute(self.y - predictions)
            gmae = np.sum(linear_error) / N
        else:
            for node in self.nodes:
                gmae += node.training_task.mean_absolute_error_log[self.iteration]
            gmae /= len(self.nodes)

        if len(self.global_mean_absolute_error_log) == self.iteration:
            self.global_mean_absolute_error_log.append(gmae)
        elif len(self.global_mean_absolute_error_log) == self.iteration + 1:
            self.global_mean_absolute_error_log[self.iteration] = gmae
        else:
            raise Exception('Unexpected global_mean_absolute_error_log size')

        if math.isnan(gmae) or math.isinf(gmae):
            raise Exception("Computation has diverged to infinite")

    def compute_global_mean_squared_error(self):
        gmse = 0
        if not self.alt_metrics:
            W = self.W_log[self.iteration]
            N = self.X.shape[0]
            mltask = self.nodes[0].training_task
            predictions = mltask.activation_func(mltask.y_hat.f(self.X, W))
            linear_error = np.absolute(self.y - predictions)
            gmse = np.sum(np.power(linear_error, 2)) / N

        else:
            for node in self.nodes:
                gmse += node.training_task.mean_squared_error_log[self.iteration]
            gmse /= len(self.nodes)

        if len(self.global_mean_squared_error_log) == self.iteration:
            self.global_mean_squared_error_log.append(gmse)
        elif len(self.global_mean_squared_error_log) == self.iteration + 1:
            self.global_mean_squared_error_log[self.iteration] = gmse
        else:
            raise Exception('Unexpected global_mean_squared_error_log size')

        if math.isnan(gmse) or math.isinf(gmse):
            raise Exception("Computation has diverged to infinite")

    def compute_global_real_mean_squared_error(self):
        grmse = 0
        if not self.alt_metrics:
            W = self.W_log[self.iteration]
            real_W = np.ones(len(W))
            N = self.X.shape[0]
            mltask = self.nodes[0].training_task
            real_values = mltask.activation_func(mltask.y_hat.f(self.X, real_W))
            predictions = mltask.activation_func(mltask.y_hat.f(self.X, W))
            linear_error = np.absolute(real_values - predictions)
            grmse = np.sum(np.power(linear_error, 2)) / N

        else:
            for node in self.nodes:
                grmse += node.training_task.real_mean_squared_error_log[self.iteration]
            grmse /= len(self.nodes)

        if len(self.global_real_mean_squared_error_log) == self.iteration:
            self.global_real_mean_squared_error_log.append(grmse)
        elif len(self.global_real_mean_squared_error_log) == self.iteration + 1:
            self.global_real_mean_squared_error_log[self.iteration] = grmse
        else:
            raise Exception('Unexpected global_mean_squared_error_log size')

        if math.isnan(grmse) or math.isinf(grmse):
            raise Exception("Computation has diverged to infinite")

    def _compute_metrics(self):
        self.compute_avg_W()
        for metric in self.nodes[0].training_task.metrics:
            # todo: remove eval!!!
            eval("self.compute_global_" + metric + "()")

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
            # console.stdout.screen.clrtoeol()

            self.clock = event["time"]

            if event["type"] == "node_step":
                node = event["node"]

                if node.can_run():
                    self.dynamic_log[node._id].append(node.gradient_step())

                    # when this node finishes iteration "i", it checks if all the others
                    # have already performed the iteration i-th, if so then the global
                    # iteration i-th has been completed and the completion time for such
                    # iteration is the actual local_clock of this node
                    min_iter = math.inf
                    for _node in self.nodes:
                        if _node.iteration < min_iter:
                            min_iter = _node.iteration
                    if min_iter == self.iteration + 1:
                        self.iterations_time_log.append(-1)
                        self.iterations_time_log[self.iteration] = node.local_clock
                        self._compute_metrics()
                        self.iteration = min_iter
                    elif min_iter > self.iteration + 1:
                        raise Exception("Unexpected behaviour of cluster distributed dynamics")
                else:
                    # node cannot run computation because it lacks some
                    # dependencies' informations
                    max_local_clock = node.local_clock
                    for dep in node.dependencies:
                        if dep.get_local_clock_by_iteration(node.iteration) > max_local_clock:
                            max_local_clock = max(max_local_clock, dep.local_clock)

                    # set the local_clock of the node equal to the value of the
                    # local_clock of the last dependency that ended the computation
                    # this node needs
                    node.set_local_clock(max_local_clock)

                # Print node's informations
                print("Node: {} | iter: {} | meanSqError: {}".format(
                    node._id,
                    node.iteration,
                    node.training_task.get_mean_squared_error())
                )

                # check for the stop condition
                stop_condition = False
                if self.iteration >= self.max_iter:
                    stop_condition = True
                    print("Cluster stopped due to global iteration (={}) being equal to max_iter (={})".format(
                        self.iteration, self.max_iter))

                if self.clock >= self.max_time:
                    stop_condition = True
                    print("Cluster stopped due to global clock (={}) being grater than or equal to max_time (={})".
                          format(self.clock, self.max_time))

                if self.get_global_mean_squared_error() <= self.epsilon:
                    stop_condition = True
                    print("Cluster stopped due to error (={}) being less than or equal to epsilon (={})".format(
                        self.get_global_mean_squared_error(),
                        self.epsilon
                    ))

                # enqueue the next event for current node
                if not stop_condition:
                    new_event = {
                        'time': node.local_clock,
                        'type': 'node_step',
                        'node': node
                    }
                    self.enqueue_event(new_event)

                """
                _depstr = ""
                for _dep in node.dependencies:
                    if _dep.get_local_clock_by_iteration(node.iteration) > node.iteration:
                        _depstr += str(_dep._id)

                console.stdout.screen.addstr(node._id, 0,
                                             "Node: {} | iter: {} | error: {} | score: {} | wait for: {}".format(
                                                 node._id,
                                                 node.iteration,
                                                 node.training_task.squared_loss(),
                                                 node.training_task.score(), _depstr))
                """
            elif event["type"] == "":
                pass
            else:
                pass

            event = self.dequeue_event()
            # console.stdout.screen.refresh()


class Node:
    """
    Represent a computational node.
    """

    def __init__(self, _id, X, y, y_hat, method, batch_size, activation_func, loss, penalty, alpha,
                 learning_rate, metrics, shuffle, verbose):
        self._id = _id  # id number of the node
        self.dependencies = []  # list of node dependencies
        self.recipients = []
        self.local_clock = 0.0  # local internal clock (float)
        self.iteration = 0  # current iteration
        self.log = [0.0]  # log indexed as "iteration" -> "completion clock"

        # buffer of incoming weights from dependencies
        # it store a queue for each dependency. Such queue can be accessed by
        # addressing the id of the node: "dependency_id" -> dep_queue.
        self.buffer = {}

        if activation_func is None:
            activation_func = "identity"

        if not activation_func is types.FunctionType:
            if activation_func == "sigmoid":
                activation_func = mltoolbox.sigmoid
            elif activation_func == "sign":
                activation_func = np.sign
            elif activation_func == "tanh":
                activation_func = np.tanh
            else:
                activation_func = lambda x: x

        # instantiate training model for the node
        if method == "stochastic":
            self.training_task = mltoolbox.StochasticGradientDescentTrainer(
                X, y, y_hat,
                activation_func,
                loss,
                penalty,
                alpha,
                learning_rate,
                metrics,
                shuffle,
                verbose
            )
        elif method == "batch":
            self.training_task = mltoolbox.BatchGradientDescentTrainer(
                batch_size,
                X, y, y_hat,
                activation_func,
                loss,
                penalty,
                alpha,
                learning_rate,
                metrics,
                shuffle,
                verbose
            )
        else:
            if method != "classic":
                warnings.warn('Method "{}" does not exist, using classic gradient descent instead'.format(method))

            self.training_task = mltoolbox.GradientDescentTrainer(
                X, y, y_hat,
                activation_func,
                loss,
                penalty,
                alpha,
                learning_rate,
                metrics,
                shuffle,
                verbose
            )

    def set_dependencies(self, dependencies):
        """
        Set the node's dependencies.
        :param dependencies: list of nodes
        :return: None
        """
        for dependency in dependencies:
            self.add_dependency(dependency)

    def add_dependency(self, dependency):
        """
        Add a new dependency for the node.
        :param dependency: node
        :return: None
        """
        self.dependencies.append(dependency)
        self.buffer[dependency._id] = []

    def add_recipient(self, recipient):
        """
        Add a new recipient for the node.
        :param recipient: node
        :return: None
        """
        self.recipients.append(recipient)

    def set_local_clock(self, new_local_clock):
        self.local_clock = new_local_clock

    def get_local_clock_by_iteration(self, iteration):
        """
        Given the iteration number, return the value of the local_clock when
        such iteration had been completed. If the iteration has not been completed
        yet then return the constant math.inf (INFINITE), due to comparison
        reasons.
        :param iteration: iteration value
        :return: local_clock when iteration had been completed
        """
        if len(self.log) > iteration:
            return self.log[iteration]
        return math.inf

    def enqueue_weight(self, dependency_node_id, weight):
        """
        Enqueue a weight in the buffer.
        :param dependency_node_id: node that perform the enqueue operation
        :param weight: weight vector to enqueue
        :return: None
        """
        self.buffer[dependency_node_id].append(weight)

    def dequeue_weight(self, dependency_node_id):
        """
        Remove and return the head of the buffer for a certain dependency.
        :param dependency_node_id: id of the dependency
        :return: weight vector from such dependency (for the current iteration)
        """
        return self.buffer[dependency_node_id].pop(0)

    def step(self):
        """
        Perform a single step (iteration) of the computation task. Actually this
        method just performs a time.sleep() that lasts for a time distributed
        following Exp(0.5).
        :return: None
        """

        """
        t0 = time.perf_counter()
        time.sleep(random.expovariate(0.5))
        t = time.perf_counter()
        self.local_clock += t - t0
        self.iteration += 1
        """
        t0 = self.local_clock
        tf = t0 + random.expovariate(0.5)
        self.local_clock = tf
        self.iteration += 1
        self.log.append(self.local_clock)
        print("node ({0}) advanced to iteration #{1}".format(self._id, self.iteration))
        return [t0, tf]

    def gradient_step(self):
        """
        Perform a single step of the gradient descent method.
        :return: a list containing [clock_before, clock_after] w.r.t. the computation
        """
        # useful vars for estimate the time taken by the computation
        t0 = self.local_clock
        # get the counter before the computation starts
        # c0 = time.perf_counter()

        # avg internal self.W vector with W incoming from dependencies
        if self.iteration > 0:
            self.avg_weight_with_dependencies()

        # compute the gradient descent step
        self.training_task.step()

        # broadcast the obtained value to all node's recipients
        self.broadcast_weight_to_recipients()

        # get the counter after the computation has ended
        # cf = time.perf_counter()
        dt = random.expovariate(1)  # todo: temp

        # computes the clock when the computation has finished
        # tf = t0 + cf - c0
        tf = t0 + dt
        # update the local_clock
        self.local_clock = tf

        self.iteration += 1
        self.log.append(self.local_clock)

        return [t0, tf]

    def avg_weight_with_dependencies(self):
        """
        Average self.W vector with weights W from dependencies.
        :return: None
        """
        if len(self.dependencies) > 0:
            W = self.training_task.W
            for dep in self.dependencies:
                W = W + self.dequeue_weight(dep._id)
            self.training_task.W = W / (len(self.dependencies) + 1)

    def broadcast_weight_to_recipients(self):
        """
        Broadcast the just computed self.W vector to recipients by enqueuing
        it on their buffers.
        :return: None
        """
        for recipient in self.recipients:
            recipient.enqueue_weight(self._id, self.training_task.W)

    def can_run(self):
        """
        Return whether the node can go further computing a new iteration, thus
        can proceed to the next step or not.
        :return: Boolean
        """
        for dep in self.dependencies:
            if dep.get_local_clock_by_iteration(self.iteration) > self.local_clock:
                return False
        return True
