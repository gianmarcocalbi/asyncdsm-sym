from src import statistics


for k in range(100):
    print("k={} -> {}".format(k, statistics.MaxOfType2ParetoDistribution.residual_time_mean(3, 2, k)))
