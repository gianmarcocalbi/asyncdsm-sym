import random, math, time
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize
from sklearn import linear_model
from src.model import Cluster
from src.graph_generator import GraphGenerator
from src import mltoolbox
import matplotlib.pyplot as plt

clique_mse_log = np.loadtxt("test_log/clique_global_mean_squared_error_log")
cycle_mse_log = np.loadtxt("test_log/cycle_global_mean_squared_error_log")
expander_mse_log = np.loadtxt("test_log/expander_global_mean_squared_error_log")
diag_mse_log = np.loadtxt("test_log/diag_global_mean_squared_error_log")

clique_iter_log = np.loadtxt("test_log/clique_iterations_time_log")
cycle_iter_log = np.loadtxt("test_log/cycle_iterations_time_log")
expander_iter_log = np.loadtxt("test_log/expander_iterations_time_log")
diag_iter_log = np.loadtxt("test_log/diag_iterations_time_log")

alpha = "0.0005"


clique_mse_avgd = []
cycle_mse_avgd = []
expander_mse_avgd = []
diag_mse_avgd = []

n_iter = len(clique_mse_log)

clique_tmp_avg = 0
cycle_tmp_avg = 0
expander_tmp_avg = 0
diag_tmp_avg = 0
j = -1
for i in range(0, n_iter+1):
    j += 1
    if j == 10:
        for k in range(i-10, i):
            clique_mse_avgd.append(clique_tmp_avg/j)
            cycle_mse_avgd.append(cycle_tmp_avg/j)
            expander_mse_avgd.append(expander_tmp_avg/j)
            diag_mse_avgd.append(diag_tmp_avg / j)

        clique_tmp_avg = 0
        cycle_tmp_avg = 0
        expander_tmp_avg = 0
        diag_tmp_avg = 0
        j = 0
    try:
        clique_tmp_avg += clique_mse_log[i]
        cycle_tmp_avg += cycle_mse_log[i]
        expander_tmp_avg += expander_mse_log[i]
        diag_tmp_avg += diag_mse_log[i]
    except:
        pass



plt.title("MSE over global iterations (α={})".format(alpha))
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.ylim(ymax=50)
plt.plot(
    list(range(0, n_iter)),
    cycle_mse_avgd,
    label="cycle MSE"
)
plt.plot(
    list(range(0, n_iter)),
    expander_mse_avgd,
    label="expand. MSE"
)
plt.plot(
    list(range(0, n_iter)),
    clique_mse_avgd,
    label="clique MSE"
)
plt.plot(
    list(range(0, n_iter)),
    diag_mse_avgd,
    label="diag MSE"
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
plt.plot(
    diag_iter_log,
    list(range(0, n_iter)),
    label="diag"
)
plt.legend()
plt.show()


plt.title("MSE over time (α={})".format(alpha))
plt.xlabel("Time (s)")
plt.ylabel("MSE")
plt.ylim(ymax=50)
plt.plot(
    clique_iter_log,
    clique_mse_avgd,
    label="clique MSE"
)
plt.plot(
    cycle_iter_log,
    cycle_mse_avgd,
    label="cycle MSE"
)
plt.plot(
    expander_iter_log,
    expander_mse_avgd,
    label="expand MSE"
)
plt.plot(
    diag_iter_log,
    diag_mse_avgd,
    label="diag MSE"
)

plt.legend()
plt.show()

