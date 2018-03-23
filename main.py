import numpy as np
from sklearn.preprocessing import normalize
import time, random, math


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
    def __init__(self, setup):
        self.nodes = []
        self.log = []
        self.settings = setup
        for i in range(self.settings["adjacency_matrix"].shape[1]):
            self.nodes.append(Node(i))
            self.log.append([])
        for i in range(self.settings["adjacency_matrix"].shape[0]):
            for j in range(self.settings["adjacency_matrix"].shape[1]):
                if i != j and self.settings["adjacency_matrix"][i, j] == 1:
                    self.nodes[j].add_dependency(self.nodes[i])

    def boot(self):
        pass

    def run(self):
        stop_condition = False
        while not stop_condition:
            for node in self.get_most_in_late_nodes():
                if node.can_run():
                    self.log[node.id].append(node.step())
                else:
                    max_local_clock = node.local_clock
                    for dep in node.dependencies:
                        if dep.get_local_clock_by_iteration(node.iteration) > max_local_clock:
                            max_local_clock = max(max_local_clock, dep.local_clock)
                    node.set_local_clock(max_local_clock)

                stop_condition = True
                for node in self.nodes:
                    if node.iteration < self.settings["iteration_amount"]:
                        stop_condition = False
                        break

                if node.id == 0 and node.iteration == 19:
                    print("stuck")
                    # input("press to continue....")
                    pass

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
    def __init__(self, id):
        self.id = id
        self.dependencies = []
        self.local_clock = 0.0
        self.iteration = 0
        self.log = [0.0]

    def set_dependencies(self, dependencies):
        for i in range(len(dependencies)):
            self.dependencies.append(dependencies[i])

    def add_dependency(self, dependency):
        self.dependencies.append(dependency)

    def set_local_clock(self, new_local_clock):
        self.local_clock = new_local_clock

    def get_local_clock_by_iteration(self, iteration):
        if len(self.log) > iteration:
            return self.log[iteration]
        return math.inf

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

    def can_run(self):
        for dep in self.dependencies:
            if dep.get_local_clock_by_iteration(self.iteration) > self.local_clock:
                return False
        return True


if __name__ == "__main__":
    """
    adjacency_matrix = np.matrix([
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 1]
    ])
    adjacency_matrix_diag = np.diag(np.ones(6))
    adjacency_matrix_bsp = np.ones((6, 6))

    markov_matrix = normalize(adjacency_matrix, axis=1, norm='l1')

    random.seed(2894)

    setup = {
        "adjacency_matrix": adjacency_matrix,
        "iteration_amount": 100
    }

    training_set = np.matrix

    cluster = Cluster(setup)
    cluster.run()
    print(cluster.nodes[0].local_clock)
    """

    print(GraphGenerator.generate_complete_graph(10))
