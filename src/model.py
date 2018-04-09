import copy
import math
import random
import time

from src import mltoolbox


class Cluster:
    def __init__(self, adjacency_matrix, training_setup, setup):
        self.future_event_list = {}

        self.nodes = []
        self.log = []  # TODO: never used
        self.adjacency_matrix = adjacency_matrix
        self.training_setup = training_setup  # setup for training model
        self.settings = setup
        self._bootstrap()

    def _bootstrap(self):
        """
        Bootstrap the cluster in order to get it ready to run.
        :return: None
        """
        # TODO: deepcopy of the training set must be avoided!
        # indeed, should be changed how the following code works

        # deepcopy of the instances of the training set
        X = copy.deepcopy(self.training_setup["X"])

        # deepcopy of all oracle function values of the training set
        y = copy.deepcopy(self.training_setup["y"])

        # if they have different sizes then the training set is bad formatted
        if len(y) != X.shape[0]:
            raise Exception("X has different amount of rows w.r.t. y")
        N = self.adjacency_matrix.shape[0]
        for i in range(N):
            node_setup = self.training_setup

            # size of the subsample of the training set that will be assigned to
            # this node
            batch_size = math.floor(X.shape[0] / (N - i))

            # assign the correct subsample to this node
            node_setup["X"] = copy.deepcopy(X[0:batch_size])  # instances
            node_setup["y"] = copy.deepcopy(y[0:batch_size])  # oracle outputs

            # instantiate new node for the just-selected subsample
            self.nodes.append(Node(i, node_setup))
            self.log.append([])

            # evict the just-already-assigned samples of the training-set
            X = X[batch_size:]
            y = y[batch_size:]

        # set up all nodes' dependencies following the adjacency_matrix
        for i in range(N):
            for j in range(self.adjacency_matrix.shape[1]):
                if i != j and self.adjacency_matrix[i, j] == 1:
                    self.nodes[j].add_dependency(self.nodes[i])
                    self.nodes[i].add_recipient(self.nodes[j])

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
        # enqueue one step event for each node in the cluster
        for _node in self.nodes:
            self.enqueue_event({
                'time': _node.local_clock,
                'type': 'node_step',
                'node': _node
            })

        stop_condition = False
        event = self.dequeue_event()
        while not stop_condition and not event is None:
            if event["type"] == "node_step":
                node = event["node"]

                if node.can_run():
                    self.log[node.id].append(node.gradient_step(self.training_setup["method"]))
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

                # check for the stop condition
                stop_condition = True
                for _node in self.nodes:
                    if _node.iteration < self.settings["iteration_amount"]:
                        stop_condition = False
                        break

                # enqueue the next event for current node
                if not stop_condition:
                    new_event = {
                        'time': node.local_clock,
                        'type': 'node_step',
                        'node': node
                    }
                    self.enqueue_event(new_event)

            elif event["type"] == "":
                pass
            else:
                pass

            event = self.dequeue_event()


class Node:
    """
    Represent a computational node.
    """

    def __init__(self, id, training_setup):
        self.id = id  # id number of the node
        self.dependencies = []  # list of node dependencies
        self.recipients = []
        self.local_clock = 0.0  # local internal clock (float)
        self.iteration = 0  # current iteration
        self.log = [0.0]  # log indexed as "iteration" -> "completion clock"

        # buffer of incoming weights from dependencies
        # it store a queue for each dependency. Such queue can be accessed by
        # addressing the id of the node: "dependency_id" -> dep_queue.
        self.buffer = {}
        self.training_setup = training_setup

        # instantiate training model for the node
        self.training_model = mltoolbox.TrainingModel(
            self.training_setup["X"],
            self.training_setup["y"],
            lambda x: x * x,
            self.training_setup["alpha"]
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
        self.buffer[dependency.id] = []

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
        print("node ({0}) advanced to iteration #{1}".format(self.id, self.iteration))
        return [t0, tf]

    def gradient_step(self, method):
        """
        Perform a single step of the gradient descent method.
        :return: a list containing [clock_before, clock_after] w.r.t. the computation
        """
        # useful vars for estimate the time taken by the computation
        t0 = self.local_clock
        # get the counter before the computation starts
        c0 = time.perf_counter()

        # avg internal self.W vector with W incoming from dependencies
        if self.iteration > 0:
            self.avg_weight_with_dependencies()

        # compute the gradient descent step
        if method == "stochastic":
            self.training_model.stochastic_gradient_descent_step()
        elif method == "batch":
            self.training_model.batch_gradient_descent_step(self.training_setup["batch_size"])
        else:
            self.training_model.gradient_descent_step()

        # broadcast the obtained value to all node's recipients
        self.broadcast_weight_to_recipients()

        # get the counter after the computation has ended
        cf = time.perf_counter()

        # computes the clock when the computation has finished
        tf = t0 + cf - c0
        # update the local_clock
        self.local_clock = tf

        self.iteration += 1
        self.log.append(self.local_clock)

        print("Error in Node {0} = {1}".format(self.id, self.training_model.loss_log[-1]))
        return [t0, tf]

    def avg_weight_with_dependencies(self):
        """
        Average self.W vector with weights W from dependencies.
        :return: None
        """
        if len(self.dependencies) > 0:
            W = self.training_model.W
            for dep in self.dependencies:
                W = W + self.dequeue_weight(dep.id)
            self.training_model.W = W / (len(self.dependencies) + 1)

    def broadcast_weight_to_recipients(self):
        """
        Broadcast the just computed self.W vector to recipients by enqueuing
        it on their buffers.
        :return: None
        """
        for recipient in self.recipients:
            recipient.enqueue_weight(self.id, self.training_model.W)

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
