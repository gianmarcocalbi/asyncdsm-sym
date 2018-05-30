from src import statistics

for k in range(100):
    exp = statistics.single_iteration_velocity_upper_bound(
        k,
        statistics.ExponentialDistribution,
        [1]
    )
    uni = statistics.single_iteration_velocity_upper_bound(
        k,
        statistics.UniformDistribution,
        [0,2]
    )
    par = statistics.single_iteration_velocity_upper_bound(
        k,
        statistics.Type2ParetoDistribution,
        [3,2]
    )
    print("k={} -> exp : {} - uni : {} - par : {}".format(k, exp, uni, par))
