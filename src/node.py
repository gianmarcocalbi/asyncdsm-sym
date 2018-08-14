import warnings
from src import tasks
from src.utils import *
from termcolor import colored as col


class Node:
    """
    Represent a computational node.
    """

    def __init__(self, _id, self_weight, X, y, real_w, obj_function, method, batch_size, dual_averaging_radius,
            alpha, learning_rate, metrics, real_metrics, real_metrics_toggle, shuffle, time_distr_class,
            time_distr_param, time_const_weight, starting_weights_domain, verbose, verbose_task):
        self.verbose = verbose
        self._id = _id  # id number of the node
        self.dependencies = []  # list of node dependencies
        self.recipients = []
        self.local_clock = 0.0  # local internal clock (float)
        self.iteration = 0  # current iteration
        self.log = [0.0]  # log indexed as "iteration" -> "completion clock"
        self.self_weight = self_weight
        self.time_distr_class = time_distr_class
        if not type(time_distr_param) in (list, tuple,):
            time_distr_param = [time_distr_param]
        self.time_distr_param = time_distr_param
        self.time_const_weight = time_const_weight
        self.is_running = True

        # buffer of incoming weights from dependencies
        # it store a queue for each dependency. Such queue can be accessed by
        # addressing the id of the node: "dependency_id" -> dep_queue.
        self.buffer = {}
        self.method = method

        # instantiate training model for the node
        if method == "stochastic":
            self.training_task = tasks.StochasticGradientDescentTrainer(
                X, y, real_w, obj_function,
                starting_weights_domain,
                alpha,
                learning_rate,
                metrics,
                real_metrics,
                real_metrics_toggle,
                shuffle,
                verbose_task
            )
        elif method == "batch":
            self.training_task = tasks.BatchGradientDescentTrainer(
                batch_size,
                starting_weights_domain,
                X, y, real_w, obj_function,
                alpha,
                learning_rate,
                metrics,
                real_metrics,
                real_metrics_toggle,
                shuffle,
                verbose_task
            )
        elif method == "linear_regression":
            self.training_task = tasks.LinearRegressionGradientDescentTrainer(
                X, y, real_w, obj_function,
                starting_weights_domain,
                alpha,
                learning_rate,
                metrics,
                real_metrics,
                real_metrics_toggle,
                shuffle,
                verbose_task
            )
        elif method == "dual_averaging":
            self.training_task = tasks.DualAveragingGradientDescentTrainer(
                dual_averaging_radius,
                X, y, real_w, obj_function,
                starting_weights_domain,
                alpha,
                learning_rate,
                metrics,
                real_metrics,
                real_metrics_toggle,
                shuffle,
                verbose_task
            )
        elif method == "subgradient":
            self.training_task = tasks.SubgradientDescentTrainer(
                dual_averaging_radius,
                X, y, real_w, obj_function,
                starting_weights_domain,
                alpha,
                learning_rate,
                metrics,
                real_metrics,
                real_metrics_toggle,
                shuffle,
                verbose_task
            )
        else:
            #warnings.warn('Method "{}" does not exist, nodes will compute nothing!'.format(method))

            self.training_task = tasks.GradientDescentTrainer(
                X, y, real_w, obj_function,
                starting_weights_domain,
                alpha,
                learning_rate,
                metrics,
                real_metrics,
                real_metrics_toggle,
                shuffle,
                verbose_task
            )

    def get_id(self):
        return self._id

    def set_dependencies(self, dependencies):
        """
        Set the node's dependencies.
        :param dependencies: list of nodes
        :return: None
        """
        dep_ids = "["
        for dependency in dependencies:
            self.add_dependency(dependency)
            dep_ids += "{}, ".format(dependency.get_id())
        dep_ids = dep_ids[:-2]
        dep_ids += "]"
        print_verbose(self.verbose, "Set Node [{}] dependencies = {}".format(col(self._id, 'cyan'), dep_ids))

    def add_dependency(self, dependency):
        """
        Add a new dependency for the node.
        :param dependency: node
        :return: None
        """
        self.dependencies.append(dependency)
        self.buffer[dependency.get_id()] = []

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

    def get_iteration_at_local_clock(self, local_clock):
        i = np.searchsorted(self.log, local_clock, side='right') - 1
        if i < 0:
            return 0
        return i

    def enqueue_outgoing_data(self, dependency_node_id, data):
        self.buffer[dependency_node_id].append(data)

    def dequeue_incoming_data(self, dependency_node_id):
        return self.buffer[dependency_node_id].pop(0)

    def step(self):
        # useful vars for estimate the time taken by the computation
        t0 = self.local_clock
        # get the counter before the computation starts
        # c0 = time.perf_counter()

        if self.method == "dual_averaging":
            self.dual_averaging_step()
        elif self.method == "linear_regression":
            self.linear_regression_step()
        elif self.method in ['classic','subgradient']:
            self.gradient_step()

        # get the counter after the computation has ended
        # cf = time.perf_counter()
        mean = self.time_distr_class.mean(*self.time_distr_param)
        c = self.time_const_weight
        x = self.time_distr_class.sample(*self.time_distr_param)
        dt = c * mean + (1 - c) * x
        # dt = random.uniform(0,2)

        # computes the clock when the computation has finished
        # tf = t0 + cf - c0
        tf = t0 + dt
        # update the local_clock
        self.local_clock = tf

        self.iteration += 1
        self.log.append(self.local_clock)

        return t0, tf

    def linear_regression_step(self):
        self.training_task.step()

    def dual_averaging_step(self):
        avg_z = np.zeros(len(self.training_task.get_w()))
        if self.iteration > 0:
            avg_z = self.avg_z_with_dependencies()

        self.training_task.step(avg_z)

        # broadcast the obtained value to all node's recipients
        self.broadcast_z_to_recipients()

    def gradient_step(self):
        """
        Perform a single step of the gradient descent method.
        :return: a list containing [clock_before, clock_after] w.r.t. the computation
        """
        # avg internal self.w vector with w incoming from dependencies
        """if self.iteration > 0:
            avg_w = self.avg_weight_with_dependencies()
        else:
            avg_w = self.training_task.get_w()"""
        avg_w = self.avg_weight_with_dependencies()

        # compute the gradient descent step
        self.training_task.step(avg_w)

        # broadcast the obtained value to all node's recipients
        self.broadcast_weight_to_recipients()

    def avg_weight_with_dependencies(self):
        """
        Average self.w vector with weights w from dependencies.
        :return: None
        """
        w = self.training_task.get_w()
        dep_w_sum = np.zeros(len(w))
        if len(self.dependencies) > 0:
            for dep in self.dependencies:
                dep_w_sum += self.dequeue_incoming_data(dep.get_id())
            dep_w = dep_w_sum / len(self.dependencies)
            avg_w = self.self_weight * w + dep_w * (1-self.self_weight)
        else:
            avg_w = w

        print_verbose(self.verbose,
            "Node [{}] averages w({}) with dependencies' w({})".format(
                col(self._id, 'cyan'),
                self.iteration,
                self.iteration
            ))

        return avg_w

    def broadcast_weight_to_recipients(self):
        rec_ids = "["
        for recipient in self.recipients:
            recipient.enqueue_outgoing_data(self.get_id(), self.training_task.get_w())
            rec_ids += "{}, ".format(recipient.get_id())

        rec_ids = rec_ids[:-2]
        rec_ids += "]"
        print_verbose(self.verbose, "Node [{}] broadcasts w({}) to recipients = [{}]".format(
            col(self._id, 'cyan'), len(self.training_task.w), rec_ids
        ))

    def avg_z_with_dependencies(self):
        z = self.training_task.get_z()  # self.training_task.get_z()
        if len(self.dependencies) > 0:
            for dep in self.dependencies:
                z += self.dequeue_incoming_data(dep.get_id())

        return z / (len(self.dependencies) + 1)

    def broadcast_z_to_recipients(self):
        for recipient in self.recipients:
            recipient.enqueue_outgoing_data(self.get_id(), self.training_task.get_z())

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
