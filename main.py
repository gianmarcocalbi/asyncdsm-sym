import numpy as np
from sklearn.preprocessing import normalize
import time, random, math
import mltoolbox
from sklearn.datasets.samples_generator import make_blobs
import copy

np.random.seed(2894)
random.seed(2894)

class GraphGenerator:
    @staticmethod
    def generate_d_regular_graph_by_adjacency(adj_matrix_first_row):
        N = len(adj_matrix_first_row)
        adjacency_matrix = np.diag(np.ones(N))
        print(adjacency_matrix)
        for i in range(N):
            for j in range(N):
                adjacency_matrix[i][j] = adj_matrix_first_row[(j - i) % N]
        return adjacency_matrix

    @staticmethod
    def generate_d_regular_graph_by_edges(N, edges):
        adjacency_matrix = np.diag(np.ones(N))
        for i in range(N):
            for e in edges:
                u, v = e.replace(" ", "").split("->")
                adjacency_matrix[eval(u) % N][eval(v) % N] = 1
        return adjacency_matrix

    @staticmethod
    def generate_complete_graph(N):
        return np.ones((N, N))

    @staticmethod
    def generate_expander_graph(N, degree):
        for i in range(N):
            pass


class Cluster:
    def __init__(self, adjacency_matrix, training_setup, setup):
        self.nodes = []
        self.log = []
        self.adjacency_matrix = adjacency_matrix
        self.training_setup = training_setup
        self.settings = setup
        self._boot()

    def _boot(self):
        X = copy.deepcopy(self.training_setup["X"])
        y = copy.deepcopy(self.training_setup["y"])
        if len(y) != X.shape[0]:
            raise Exception("X has different amount of rows w.r.t. y")
        N = self.adjacency_matrix.shape[0]
        for i in range(N):
            node_setup = {
                "alpha": self.training_setup["alpha"],
                "activation_function": self.training_setup["activation_function"]
            }
            batch_size = math.floor(X.shape[0] / (N - i))
            node_setup["X"] = copy.deepcopy(X[0:batch_size])
            node_setup["y"] = copy.deepcopy(y[0:batch_size])
            self.nodes.append(Node(i, node_setup))
            self.log.append([])
            X = X[batch_size:]
            y = y[batch_size:]
        for i in range(N):
            for j in range(self.adjacency_matrix.shape[1]):
                if i != j and self.adjacency_matrix[i, j] == 1:
                    self.nodes[j].add_dependency(self.nodes[i])

    def run(self):
        stop_condition = False
        while not stop_condition:
            for node in self.get_most_in_late_nodes():
                if node.can_run():
                    self.log[node.id].append(node.gradient_step())
                else:
                    max_local_clock = node.local_clock
                    for dep in node.dependencies:
                        if dep.get_local_clock_by_iteration(node.iteration) > max_local_clock:
                            max_local_clock = max(max_local_clock, dep.local_clock)
                    node.set_local_clock(max_local_clock)

                stop_condition = True
                for _node in self.nodes:
                    if _node.iteration < self.settings["iteration_amount"]:
                        stop_condition = False
                        break

    def get_most_in_late_node(self):
        candidate = None
        for node in self.nodes:
            if candidate is None:
                candidate = node
            elif node.local_clock < candidate.local_clock:
                candidate = node
        return candidate

    def get_most_in_late_nodes(self):
        in_late_clock = self.get_most_in_late_node().local_clock
        in_late_nodes = []
        for node in self.nodes:
            if node.local_clock == in_late_clock:
                in_late_nodes.append(node)
            elif node.local_clock < in_late_clock:
                raise Exception("Node clock is less than the most in late clock. Impossible.")
        return in_late_nodes


class Node:
    def __init__(self, id, training_setup):
        self.id = id
        self.dependencies = []
        self.local_clock = 0.0
        self.iteration = 0
        self.log = [0.0]
        self.buffer = []
        self.training_setup = training_setup
        self.training_model = mltoolbox.TrainingModel(
            self.training_setup["X"],
            self.training_setup["y"],
            lambda x: x * x,
            self.training_setup["alpha"]
        )

    def set_dependencies(self, dependencies):
        for i in range(len(dependencies)):
            self.dependencies.append(dependencies[i])
            self.buffer.append([])

    def add_dependency(self, dependency):
        self.dependencies.append(dependency)

    def set_local_clock(self, new_local_clock):
        self.local_clock = new_local_clock

    def get_local_clock_by_iteration(self, iteration):
        if len(self.log) > iteration:
            return self.log[iteration]
        return math.inf

    def enqueue_weight(self, sender_node, weight):
        self.buffer[sender_node].append(weight)

    def dequeue_weight(self, dep_node):
        self.buffer[dep_node].pop(0)

    def step(self):
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
        t0 = self.local_clock
        c0 = time.perf_counter()

        # avg with dependencies
        self.avg_weight_with_dependencies()
        self.training_model.gradient_descent_step()
        self.broadcast_weight_to_dependencies()

        cf = time.perf_counter()
        tf = t0 + cf - c0
        self.local_clock = tf
        self.iteration += 1

        print("Error in Node {0} = {1}".format(self.id, self.training_model.loss_log[-1]))

        return [t0, tf]

    def avg_weight_with_dependencies(self):
        if len(self.dependencies) > 0:
            W = self.training_model.W
            for dep in self.dependencies:
                W += self.dequeue_weight(dep.id)
            self.training_model.W = W / (len(self.dependencies) + 1)

    def broadcast_weight_to_dependencies(self):
        for dep in self.dependencies:
            dep.enqueue_weight(self.id, self.training_model.W)

    def can_run(self):
        for dep in self.dependencies:
            if dep.get_local_clock_by_iteration(self.iteration) > self.local_clock:
                return False
        return True


if __name__ == "__main__":
    # adjacency_matrix = GraphGenerator.generate_d_regular_graph_by_edges(5, ["i->i+1"])
    adjacency_matrix = GraphGenerator.generate_complete_graph(1)
    markov_matrix = normalize(adjacency_matrix, axis=1, norm='l1')
    (X, y) = make_blobs(n_samples=10, n_features=10, centers=2, cluster_std=2, random_state=20)

    setup = {
        "iteration_amount": 10,
    }

    training_setup = {
        "X": X,
        "y": y,
        "alpha": 0.005,
        "activation_function": "sigmoid"
    }

    cluster = Cluster(adjacency_matrix, training_setup, setup)
    cluster.run()