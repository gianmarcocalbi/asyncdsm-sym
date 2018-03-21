import numpy as np
from sklearn.preprocessing import normalize
import time, random, math


class Cluster:
    def __init__(self, adjacency_matrix):
        self.nodes = []
        for i in range(adjacency_matrix.shape[1]):
            self.nodes.append(Node(i))
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if i != j and adjacency_matrix[i, j] == 1:
                    self.nodes[j].add_dependency(self.nodes[i])

    def boot(self):
        pass

    def run(self):
        ITERATION_AMOUNT = 100
        stop_condition = False
        while not stop_condition:
            node = self.get_most_in_late_node()
            if node.can_run():
                node.step()
            else:
                max_local_clock = node.local_clock
                for dep in node.dependencies:
                    # restart from here:
                    # control the following conditions
                    if dep.iteration > node.iteration:
                        if dep.get_local_clock_by_iteration(node.iteration) > max_local_clock:
                            max_local_clock = dep.local_clock
                node.set_local_clock(max_local_clock)

            stop_condition = True
            for node in self.nodes:
                if node.iteration < ITERATION_AMOUNT:
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
        self.local_clock += random.expovariate(0.5)
        self.iteration += 1
        self.log.append(self.local_clock)
        print("node ({0}) advanced to iteration #{1}".format(self.id, self.iteration))

    def can_run(self):
        for dep in self.dependencies:
            if dep.get_local_clock_by_iteration(self.iteration) > self.local_clock:
                return False
        return True


if __name__ == "__main__":
    adjacency_matrix = np.matrix([
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 1]
    ])
    adjacency_matrix_zeros = np.zeros((6, 6))
    adjacency_matrix_diag = np.diag(np.ones(6))
    adjacency_matrix_bsp = np.ones((6, 6))

    markov_matrix = normalize(adjacency_matrix, axis=1, norm='l1')

    nodes_amount = adjacency_matrix.shape[1]

    random.seed(2894)

    cluster = Cluster(adjacency_matrix)
    cluster.run()
    print(cluster.nodes[0].local_clock)
