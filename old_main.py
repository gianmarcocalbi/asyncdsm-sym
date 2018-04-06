import numpy as np
from sklearn.preprocessing import normalize
import time, random, math
from src import mltoolbox
from sklearn.datasets.samples_generator import make_blobs
import copy

np.random.seed(2894)
random.seed(2894)


class GraphGenerator:

    @staticmethod
    def generate_d_regular_graph_by_adjacency(adj_matrix_first_row):
        """
        Generate considering linear adjacency relationship, that is
        if there exists and edge (i,j) then there exists also an
        edge (i+i,j+i). Therefore given the 1st row of the adjacency
        matrix, then , for instance, the 2nd row of the adjacency
        matrix will be the 1st line shifted to the right by one.
        :param adj_matrix_first_row: first row of the adjacency matrix
        :return: whole adjacency NUMPY matrix
        """
        N = len(adj_matrix_first_row)
        adjacency_matrix = np.diag(np.ones(N))
        print(adjacency_matrix)
        for i in range(N):
            for j in range(N):
                adjacency_matrix[i][j] = adj_matrix_first_row[(j - i) % N]
        return adjacency_matrix

    @staticmethod
    def generate_d_regular_graph_by_edges(N, edges):
        """
        Generate a d-regular adjacency graph matrix starting from the
        general form of the edges formatted as a string "i->f(i)" so
        that the right part of the expression (f(i)) is a function of
        i in python language (e.g. f(i)=math.floor(i+math.sqrt(i))). If
        you would like to use the total number of vertices in the graph
        then type "N". So, for instance, "i->(i+math.floor(N/2))%N" is
        a valid expression. NB: always use he right arrow "->" and not "<-"!
        :param N: total amount of vertices in the graph
        :param edges: list of strings formatted as "i->f(i)"
        :return: adjacency numpy matrix
        """
        adjacency_matrix = np.diag(np.ones(N))
        for i in range(N):
            for e in edges:
                u, v = e.replace(" ", "").split("->")
                adjacency_matrix[eval(u) % N][eval(v) % N] = 1
        return adjacency_matrix

    @staticmethod
    def generate_complete_graph(N):
        """
        Generate complete (clique) graph adjacency matrix.
        :param N: amount of vertices
        :return: adjacency numpy matrix
        """
        return np.ones((N, N))

    @staticmethod
    def generate_expander_graph(N, degree):
        # todo
        for i in range(N):
            pass


class Cluster:
    def __init__(self, adjacency_matrix, training_setup, setup):
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
            node_setup = {
                "alpha": self.training_setup["alpha"],
                "activation_function": self.training_setup["activation_function"]
            }

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

    def run(self):
        """
        Run the cluster (distributed computation).
        :return: None
        """
        stop_condition = False
        while not stop_condition:
            # loop on all nodes with smallest local_clock value
            for node in self.get_most_in_late_nodes():
                if node.can_run():
                    self.log[node.id].append(node.gradient_step())
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

    def get_most_in_late_node(self):
        """
        Get the node that has the smallest value for local_clock among all the
        nodes currently within the cluster. If more than one then return the
        latest.
        :return: most in late node
        """
        candidate = None
        for node in self.nodes:
            if candidate is None:
                candidate = node
            elif node.local_clock < candidate.local_clock:
                candidate = node
        return candidate

    def get_most_in_late_nodes(self):
        """
        Get the list of the nodes that have the smallest value for local_clock
        among all the nodes currently within the cluster. If all nodes own the
        same value for local_clock then return the list of all of them.
        :return: list of nodes
        """
        in_late_clock = self.get_most_in_late_node().local_clock
        in_late_nodes = []
        for node in self.nodes:
            if node.local_clock == in_late_clock:
                in_late_nodes.append(node)
            elif node.local_clock < in_late_clock:
                raise Exception("Error: Node clock is less than the most in late clock. Impossible.")
        return in_late_nodes


class Node:
    """
    Represent a computational node.
    """

    def __init__(self, id, training_setup):
        self.id = id  # id number of the node
        self.dependencies = []  # list of node dependencies
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

    def enqueue_weight(self, sender_node, weight):
        """
        Enqueue a weight in the buffer.
        :param sender_node: node that perform the enqueue operation
        :param weight: weight vector to enqueue
        :return: None
        """
        self.buffer[sender_node].append(weight)

    def dequeue_weight(self, dep_node):
        """
        Remove and return the head of the queue for a certain dependency.
        :param dep_node: id of the dependency
        :return: weight vector from such dependency (for the current iteration)
        """
        return self.buffer[dep_node].pop(0)

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

    def gradient_step(self):
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
        self.training_model.gradient_descent_step()

        # broadcast the obtained value to all node's dependencies
        self.broadcast_weight_to_dependencies()

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

    def broadcast_weight_to_dependencies(self):
        """
        Broadcast the just computed self.W vector to dependencies by enqueuing
        it on their buffers.
        :return: None
        """
        for dep in self.dependencies:
            dep.enqueue_weight(self.id, self.training_model.W)

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


if __name__ == "__main__":
    # adjacency_matrix = GraphGenerator.generate_d_regular_graph_by_edges(5, ["i->i+1"])
    __adjacency_matrix = GraphGenerator.generate_complete_graph(1)
    __markov_matrix = normalize(__adjacency_matrix, axis=1, norm='l1')
    (__X, __y) = make_blobs(n_samples=10000, n_features=10, centers=2, cluster_std=2, random_state=20)

    __setup = {
        "iteration_amount": 10000,
    }

    __training_setup = {
        "X": __X,
        "y": __y,
        "alpha": 0.01,
        "activation_function": "sigmoid"
    }

    __cluster = Cluster(__adjacency_matrix, __training_setup, __setup)
    __cluster.run()
