import math, sys


def iteration_speed_lower_bound(l, k, time_arr):
    """
    Compute almost-lower bound.

    Parameters
    ----------
    l : float
        Lambda (exponential rate).
    k : int
        Graph's degree.
    time_arr : list of float
        #iter -> completion time fo such iteration.

    Returns
    -------
    list of float
    """

    lb = []
    for t in time_arr:
        lb.append(t * l / (1 + sum(1 / i for i in range(1, k + 1))))
    return lb


def progress(current_progress, total_progress, bar_length=50, text_before='', text_after=''):
    filled_len = int(round(bar_length * current_progress / float(total_progress)))
    percents = round(100.0 * current_progress / float(total_progress), 1)
    bar = '#' * filled_len + '-' * (bar_length - filled_len)

    if text_before != '':
        text_before += ' '

    sys.stdout.write('{}[{}] {}% {}\r'.format(text_before, bar, percents, text_after))
    sys.stdout.flush()
