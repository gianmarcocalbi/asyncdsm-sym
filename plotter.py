import random, math, time
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize
from sklearn import linear_model
from src.model import Cluster
from src.graph_generator import GraphGenerator
from src import mltoolbox
import matplotlib.pyplot as plt

test_folder = "test_A2/"
avg = 10

graphs = (
    "clique",
    "cycle",
    "expand",
    "diag",
)

plots = (
    "iter_time",
    "mse_iter",
    "real-mse_iter",
    "mse_time",
    "real-mse_time",
)

mse_log = {}
real_mse_log = {}
iter_log = {}

for graph in graphs:
    mse_log[graph] = np.loadtxt("test_log/{}{}_global_mean_squared_error_log".format(test_folder, graph))
    real_mse_log[graph] = np.loadtxt("test_log/{}{}_global_real_mean_squared_error_log".format(test_folder, graph))
    iter_log[graph] = np.loadtxt("test_log/{}{}_iterations_time_log".format(test_folder, graph))

n_iter = len(mse_log[graphs[0]])

if not avg is None:
    avg_real_mse_log = {}
    avg_mse_log = {}
    for graph in graphs:
        avg_mse_log[graph] = np.zeros(n_iter)
        avg_real_mse_log[graph] = np.zeros(n_iter)

    for i in range(0, int(n_iter / avg)):
        for graph in graphs:
            beg = i * avg
            end = min((i + 1) * avg, n_iter)
            avg_mse_log[graph] = np.concatenate([
                avg_mse_log[graph][0:beg],
                np.full(end - beg, np.mean(mse_log[graph][beg:end])),
                avg_mse_log[graph][end:]
            ])
            avg_real_mse_log[graph] = np.concatenate([
                avg_real_mse_log[graph][0:beg],
                np.full(end - beg, np.mean(real_mse_log[graph][beg:end])),
                avg_real_mse_log[graph][end:]
            ])

    for graph in graphs:
        mse_log[graph] = avg_mse_log[graph]
        real_mse_log[graph] = avg_real_mse_log[graph]

if "iter_time" in plots:
    plt.title("Global iterations over cluster clock")
    plt.xlabel("Time (s)")
    plt.ylabel("Iteration")
    for graph in graphs:
        plt.scatter(
            iter_log[graph],
            list(range(0, n_iter)),
            label=graph,
            s=0.10
        )
    plt.legend()
    plt.show()

if "mse_iter" in plots:
    plt.title("MSE over global iterations")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.ylim(ymax=50)
    for graph in graphs:
        plt.scatter(
            list(range(0, n_iter)),
            mse_log[graph],
            label=graph,
            s=0.10
        )
    plt.legend()
    plt.show()

if "real-mse_iter" in plots:
    plt.title("Real MSE over global iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Real MSE")
    plt.ylim(ymax=50)
    for graph in graphs:
        plt.scatter(
            list(range(0, n_iter)),
            real_mse_log[graph],
            label=graph,
            s=0.10
        )
    plt.legend()
    plt.show()

if "mse_time" in plots:
    plt.title("MSE over time")
    plt.xlabel("Time (s)")
    plt.ylabel("MSE")
    plt.ylim(ymax=50)
    for graph in graphs:
        plt.scatter(
            iter_log[graph],
            mse_log[graph],
            label=graph,
            s=0.10
        )
    plt.legend()
    plt.show()

if "real-mse_time" in plots:
    plt.title("Real MSE over time")
    plt.xlabel("Time (s)")
    plt.ylim(ymax=50)
    plt.ylabel("Real MSE")
    for graph in graphs:
        plt.scatter(
            iter_log[graph],
            real_mse_log[graph],
            label=graph,
            s=0.10
        )
    plt.legend()
    plt.show()
