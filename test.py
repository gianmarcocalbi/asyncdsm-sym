import random, math, time
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize
from sklearn import linear_model
from src.model import Cluster
from src.graph_generator import GraphGenerator
from src import mltoolbox
import matplotlib.pyplot as plt

clique_mse_log = np.loadtxt("out/clique_global_mean_squared_error_log")
cycle_mse_log = np.loadtxt("out/cycle_global_mean_squared_error_log")
expander_mse_log = np.loadtxt("out/expander_global_mean_squared_error_log")

clique_iter_log = np.loadtxt("out/clique_iterations_time_log")
cycle_iter_log = np.loadtxt("out/cycle_iterations_time_log")
expander_iter_log = np.loadtxt("out/expander_iterations_time_log")

alpha = "0.0005"


n_iter = len(clique_mse_log)
plt.title("MSE over global iterations (α={})".format(alpha))
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.ylim(ymax=50)
plt.plot(
    list(range(0, n_iter)),
    clique_mse_log,
    label="clique MSE"
)
plt.plot(
    list(range(0, n_iter)),
    cycle_mse_log,
    label="cycle MSE"
)
plt.plot(
    list(range(0, n_iter)),
    expander_mse_log,
    label="expand. MSE"
)
plt.legend()
plt.show()


plt.title("Global iterations over cluster clock (α={})".format(alpha))
plt.xlabel("Time (s)")
plt.ylabel("Iteration")
plt.plot(
    clique_iter_log,
    list(range(0, n_iter)),
    label="clique"
)
plt.plot(
    cycle_iter_log,
    list(range(0, n_iter)),
    label="cycle"
)
plt.plot(
    expander_iter_log,
    list(range(0, n_iter)),
    label="expand"
)
plt.legend()
plt.show()


plt.title("MSE over time (α={})".format(alpha))
plt.xlabel("Time (s)")
plt.ylabel("MSE")
plt.ylim(ymax=50)
plt.plot(
    clique_iter_log,
    clique_mse_log,
    label="clique MSE"
)
plt.plot(
    cycle_iter_log,
    cycle_mse_log,
    label="cycle MSE"
)
plt.plot(
    expander_iter_log,
    expander_mse_log,
    label="expand MSE"
)

plt.legend()
plt.show()

