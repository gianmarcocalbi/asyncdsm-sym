import random, math, time
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize
from sklearn import linear_model
from src.graph_generator import GraphGenerator
from src import model, mltoolbox, console
from curses import wrapper
import matplotlib.pyplot as plt

seed = 28041994
np.random.seed(seed)
random.seed(seed)

def main(stdscr):
    console.stdout.screen = stdscr
    console.stdout.open()

    # __adjacency_matrix = GraphGenerator.generate_d_regular_graph_by_edges(5, ["i->i+1"])
    __adjacency_matrix = GraphGenerator.generate_complete_graph(1)
    # __markov_matrix = normalize(__adjacency_matrix, axis=1, norm='l1')
    # __X, __y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)
    __X, __y = mltoolbox.SampleGenerator.sample_from_function(
        1000, 3, mltoolbox.sphere_function, 1, biased=True)
    # __X, __y = np.loadtxt("./dataset/largescale_challenge/alpha/alpha_train.dat"), np.loadtxt("./dataset/largescale_challenge/alpha/alpha_train.lab")

    #__X = np.matrix([np.zeros(10),np.arange(10)]).T
    #__y = np.apply_along_axis(lambda x : x * x, 0, np.arange(10))

    #__X = np.matrix("0,0;1,1;2,2;3,3;4,4;5,5")
    #__y = np.apply_along_axis(lambda x : x * x, 0, np.arange(6))

    __setup = {
        "iteration_amount": math.inf
    }

    __training_setup = {
        "X": __X,
        "y": __y,
        "learning_rate": 0.01,
        "activation_function": "tanh",  # sigmoid, sign, tanh, identity, whatever other name will lead to identity
        "method": "stochastic",  # classic, stochastic, batch
        "batch_size": 10  # matters only for batch method
    }

    __cluster = model.Cluster(__adjacency_matrix, __training_setup, __setup)
    __cluster.run()

    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    # plt.yscale('log')
    # plt.axis(ymax=0.50)
    # plt.annotate('Error {}'.format(__cluster.nodes[0].training_model.squared_loss_log[-1]),
    #   xy=(len(__cluster.nodes[0].training_model.squared_loss_log)/2, 5))
    plt.plot(list(range(0, len(__cluster.nodes[0].training_model.squared_loss_log))),
             __cluster.nodes[0].training_model.squared_loss_log)
    plt.show()

    console.print("Score: {}".format(__cluster.nodes[0].training_model.score()))

    input("Press an key")

    console.stdout.close()

def main1():
    # __X, __y = make_blobs(n_samples=10000, n_features=100, centers=3, cluster_std=2, random_state=20)
    __X, __y = mltoolbox.SampleGenerator.sample_from_function(1000, 100, mltoolbox.linear_function, 1, biased=False)
    cls = linear_model.SGDClassifier(loss="squared_loss", max_iter=100000)
    cls.fit(__X, __y)
    print(cls.score(__X, __y))

def main2():
    __X, __y = mltoolbox.SampleGenerator.sample_from_function(1000, 3, mltoolbox.linear_function, 10, biased=False)
    cls = linear_model.SGDRegressor(penalty='none', alpha=0.01, max_iter=1000, shuffle=False, learning_rate='constant')
    cls.fit(__X, __y)
    print(cls.score(__X, __y))
    print(cls.predict(np.array([2,4,8]).reshape(1,-1)))

switch = 0

if __name__ == "__main__":
    if switch == 0:
        wrapper(main)
    elif switch == 1:
        main1()
    elif switch == 2:
        main2()
