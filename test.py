from src import statistics
print(statistics.single_iteration_velocity_don_exp_bound(
    10,
    statistics.ExponentialDistribution,
    [1]
))