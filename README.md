# asyncdsm
Asynchronous Distributed Subgradient Descend Method

# Directory structure
- `./dataset` : contains the datasets if they are stored in a file and not generated on the fly;
- `./distr` : contains the time distribution if they're loaded from a file and not generated on the fly (one value per line);
- `./docs` : contains the documents (not really);
- `./graphs` : contains those graphs that cannot be generated on the fly by the simulator (usually expander graphs cause they require a lot of time to be computed), those files are then used by the graph module to generate graphs topology objects;
- `./src` : python source code;
- `./test_log` : contains the output of the experiments.

The files in the root `./` are either explained in the following or not important.

# How to use the simulator
## Requirements
- python >=3.6;
- conda full + `termcolor` or equivalently try to run and install the required packages suggestions when the execution fails or try running the following in terminal
```bash
$ python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose termcolor networkx
```

# Entry point
The entry point to run experiments should be the script `main.py`, there you should setup the different experiments: usually define one function for experiment with the paraneters you need to change within and then call it 
def test7_on_sloreg_dataset(seed=None, n=100)


# `simulator.py`
It contains the function `run(...)` that is the only method to call to get a full experiment. It accepts several parameters, many of them optionals, that are used to fully customize your experiment. In the following there their description.

## Parameters
### `seed`
- type: int or `None`
- default value: `None`
Random simulation seed. If None, then the seed will be set equal to the current timestamp (in seconds). A simulation run many times with the same seed will give the same results.

### `n`
- type: int
- default value: 100
Amount of executors in the simulated cluster, also the amount of nodes of the topologies.

### `graphs`
- type: list of strings
List of topologies to run the simulation with.

So each graph topology is "expressed" as a string. Each string is converted in a Graph object by the function `generate_n_nodes_graphs_list(n, graphs)` within `./src/graphs.py`

n_samples : int
    Total number of samples in the generated dataset.
n_features : int
    Number of feature each sample will have.
dataset : str
    Dataset label:
    - "reg": general customizable linear regression dataset;
    - "unireg": unidimensional regression;
    - "svm": multidimensional classification problem;
    - "unisvm": unidimensional dataset that changes with topology spectral gap;
    - "skreg" : regression dataset from sklearn library,
    - "sloreg" and "susysvm" from UCI's repository.
smv_label_flip_prob : float
    Probability that a label is flipped in svm dataset generation.
    Kind of noise added in the dataset.
error_mean : float
    Mean of noise to introduce in regression datasets.
error_std_dev : float
    Standard deviation of noise introduced in regression datasets.
node_error_mean : float
    Mean of the per-node noise introduced in each node's sample.
    Be careful because if used with SVM this can change values of labels.
node_error_std_dev : float
    Standard deviation of the per-node noise introduced in each node's sample.
    Be careful because if used with SVM this can change values of labels.
starting_weights_domain : List[float]
    In the form of [a,b]. Domain of each node's w is uniformly randomly picked within a and b.
max_iter : int
    Maximum iteration after which the simulation is stopped.
max_time : float
    Maximum time value after which the simulation is stopped.
epsilon : float
    Accuracy threshold for objective function below which the simulation is stopped.
method : str
    - "classic" : classic gradient descent, batch is equal to the whole dataset;
    - "stochastic" : stochastic gradient descent;
    - "batch" : batch gradient descent;
    - "subgradient" : subgradient projected gradient descent;
    - "dual_averaging" : dual averaging method.
alpha : float
    Learning rate constant coefficient.
learning_rate : str
    - 'constant' : the learning rate never changes during the simulation (it is euqual to alpha);
    - 'root_decreasing' : learning rate is alpha * 1/math.sqrt(K) where K = #iter.
spectrum_dependent_learning_rate : bool
    If True the learning rate is also multiplied by math.sqrt(spectral_gap), so it is different for each graph.
dual_averaging_radius : int
    Radius of the projection on the feasible set.
time_distr_class : object
    Class of the random time distribution.
time_distr_param : list or list of list
    Parameters list.
    See Also generate_time_distr_param_list.
time_distr_param_rule : str
    Parameters distribution rule.
    See Also generate_time_distr_param_list.
time_const_weight : float
    Weight assigned to constant part of the computation time.
    It is calculated as T_u(t) = E[X_u] * c + (1-c) * X_u(t).
real_y_activation_func : function
    Activation function applied on real_y calculation.
obj_function : str
    Identifier of the objective function (one of those declared in metrics.py).
average_model_toggle : bool
    If True then the average over time of parameter vector is used istead of just x(k).
metrics : list of str
    List of additional metrics to compute (objective function is automatically added to this list).
real_metrics : list of str
    List of real metrics to compute (with regards to the real noiseless model).
real_metrics_toggle : bool
    If False real metrics are not computed (useful to speed up the computation).
metrics_type : int
    - 0 : metrics are computed over the whole dataset using model W equal to the avg of nodes' locla models;
    - 1 : metrics are computed as AVG of local nodes' metrics;
    - 2 : metrics are computed over the whole dataset using the model only from metrics_nodes (see below).
metrics_nodes : int or list of int
    If type is int then it will be put into a list and treated as [int].
    Depends on the value of metrics_type:
    - metrics_type == 0 : no effects;
    - metrics_type == 1 : metrics are computed as avg of local metrics of nodes inside metrics_nodes list;
    - metrics_type == 2 : metrics are computed over the whole dataset using the model obtained as mean of
        nodes inside metrics_nodes.
shuffle : bool
    If True the dataset is shuffled before being split into nodes, otherwise the dataset is untouched.
batch_size : int
    Useful only for batch gradient descent, is the size of the batch.
save_test_to_file : bool
    If True the test is saved to specified folder, otherwise it is stored into tempo folder.
test_folder_name_struct : list
    See generate_test_subfolder_name.
test_parent_folder : str
    Parent test folder: the test will be located in ./test_log/{$PARENT_FOLDER}/{$TEST_NAME_FOLDER}.
    Can be more than one-folder-deep!
instant_plot : bool
    If True plots will be prompted upon finishing simulation. Be careful since it will pause the thread!
plots : list of str
    List of plots' names to create / prompt upon finishing simulation.
    See plotter.py.
save_plot_to_file : bool
    If True plots will be saved into .../{$TEST_FOLDER_NAME}/plots/ folder.
plot_global_w : bool
    If True global W will be prompted after finishing simulation.
    This plot is never automatically saved, save it by yourself if you need to keep it.
plot_node_w : list or False
    List of nodes to plot w which. If False nothing will be prompted.
verbose_main : int
    Verbose policy in simulator.py script.
    - <0 : no print at all except from errors (unsafe).
    -  0 : default messages;
    -  1 : verbose + default messages
    -  2 : verbose + default messages + input required to continue after each message (simulation will be paused
        after each message and will require to press ENTER to go on, useful for debugging).
verbose_cluster : int
    Verbose policy in cluster.py script.
    See verbose_main.
verbose_node : int
    Verbose policy in node.py script.
    See verbose_main.
verbose_task : int
    Verbose policy in tasks.py script.
    See verbose_main.
verbose_plotter : int
    Verbose policy in plotter.py script.
    See verbose_main.
Returns
-------
None
"""