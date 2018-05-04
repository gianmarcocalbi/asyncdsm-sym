import math


def iteration_speed_lower_bound(l, k, time_arr):
    lb = []
    for t in time_arr:
        lb.append(t * l / (1 + sum(1 / i for i in range(1, k + 1))))
    return lb