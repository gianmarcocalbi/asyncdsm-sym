from matplotlib import pyplot as plt
from src.utils import *
from src.mltoolbox.metrics import METRICS
import os, math

# root_folder_path = 'C:/Users/grimidev/Dropbox/Share/asynchronous_computing/figures/simulations/'
root_folder_path = './figures/'
colors = {
    'black': [0x00, 0x00, 0x00],
    'purple': [0x88, 0x64, 0xff],
    'magenta': [0xff, 0x64, 0xff],
    'red': [0xff, 0x64, 0x64],
    'light-pink': [0xff, 0x8b, 0xbe],
    'light-orange': [0xff, 0xb2, 0x7b],
    'yellow': [0xdc, 0xdf, 0x00],
    'light-green': [0x99, 0xf4, 0x5c],
    'dark-green': [0x71, 0xac, 0x70],
    'water-green': [0x59, 0xdd, 0xb1],
    'cyan': [0x78, 0xc7, 0xe3],
    'light-blue': [0x72, 0x9c, 0xff],
    'light-purple': [0xac, 0x97, 0xff],
    'blue': [0x46, 0x46, 0xff],
    'dark-blue': [0x36, 0x00, 0xab],
    'purple-blue': [0xbe, 0xbe, 0xff]
}

degree_colors = {
    100: {
        0: 'black',
        1: 'purple',
        2: 'magenta',
        3: 'red',
        4: 'light-orange',
        8: 'yellow',
        10: 'light-green',
        20: 'dark-green',
        50: 'cyan',
        80: 'light-purple',
        99: 'blue'
    },
    1000: {
        2: 'magenta',
        3: 'red',
        4: 'light-pink',
        8: 'light-orange',
        18: 'black',
        20: 'yellow',
        30: 'light-green',
        40: 'dark-green',
        50: 'water-green',
        100: 'cyan',
        200: 'light-blue',
        300: 'light-purple',
        400: 'blue',
        500: 'dark-blue',
        999: 'purple-blue'
    }
}

for Dict in degree_colors.values():
    for deg in Dict:
        Dict[deg] = colors[Dict[deg]]
        for i in [0, 1, 2]:
            Dict[deg][i] /= 0xff
        Dict[deg] = tuple(Dict[deg])
"""
UNISVM
Fig. YYY
n=1000, c=.1, alpha dep from SG, par(3,2), for expanders
"""


def run():
    logs, setup = load_test_logs(
        './test_log/test_us2000_unisvm2_sgC0.05alpha_100n_shuf_lomax[3-2]_mtrT2worst_INFtime_1000iter'
    )
    #unisvm2_1000n_par_error_slope_vs_iter_comparison(logs, setup, save=False)
    save_all()


def save_all():
    logs, setup = load_test_logs(
        './test_log/test_us2000_unisvm2_sgC0.05alpha_100n_shuf_lomax[3-2]_mtrT2worst_INFtime_1000iter')
    unisvm2_1000n_par_iter_time(logs, setup, save=True)
    unisvm2_1000n_par_worst_err_vs_iter(logs, setup, save=True)
    unisvm2_1000n_par_worst_err_vs_time(logs, setup, save=True)
    unisvm2_1000n_par_error_slope_vs_iter_comparison(logs, setup, save=True)


# a) iterations VS time
def unisvm2_1000n_par_iter_time(logs, setup, save=False):
    plt.suptitle('Fig YYY (a)', fontsize=12)
    plt.title('Min iteration VS time', loc='left')
    plt.title("({})".format(setup['time_distr_class'].name), loc='right')
    plt.xlabel('Time')
    plt.ylabel('Iteration')

    for graph, iters in logs['iter_time'].items():
        plt.plot(
            iters,
            list(range(len(iters))),
            label=degree_from_label(graph),
            # markersize=2,
            color=degree_colors[setup['n']][degree_from_label(graph)]
        )

    plt.yscale('linear')
    plt.legend(title="", fontsize='small', fancybox=True)
    if save:
        plt.savefig(root_folder_path + 'unisvm_1000n_par_iter_vs_time.png')
    else:
        plt.show()
    plt.close()


# b) error VS iter
def unisvm2_1000n_par_worst_err_vs_iter(logs, setup, save=False):
    plt.suptitle('Fig YYY (b)', fontsize=12)
    plt.title('Error VS iterations', loc='left')
    plt.title("({})".format(setup['time_distr_class'].name), loc='right')
    plt.xlabel('Iterations')
    plt.ylabel('Hinge loss')

    xlim = -math.inf
    for graph, loss in logs['metrics']['hinge_loss'].items():
        plt.plot(
            list(range(len(loss))),
            loss,
            label=degree_from_label(graph),
            markersize=2,
            color=degree_colors[setup['n']][degree_from_label(graph)],
        )
        xlim = max(xlim, len(loss))
    plt.xlim(-20, xlim)
    plt.yscale('linear')
    plt.legend(title="", fontsize='small', fancybox=True)
    if save:
        plt.savefig(root_folder_path + 'unisvm_1000n_par_worst_err_vs_iter.png')
    else:
        plt.show()
    plt.close()


# c) worst error VS time
def unisvm2_1000n_par_worst_err_vs_time(logs, setup, save=False):
    plt.suptitle('Fig YYY (b)', fontsize=12)
    plt.title('Error VS time', loc='left')
    plt.title("({})".format(setup['time_distr_class'].name), loc='right')
    plt.xlabel('Time')
    plt.ylabel('Hinge loss')

    xlim = math.inf
    for graph in logs['metrics']['hinge_loss']:
        plt.plot(
            logs['iter_time'][graph],
            logs['metrics']['hinge_loss'][graph],
            label=degree_from_label(graph),
            markersize=2,
            color=degree_colors[setup['n']][degree_from_label(graph)],
        )
        xlim = min(xlim, logs['iter_time'][graph][-1])
    plt.xlim(-50, xlim)
    plt.yscale('linear')
    plt.legend(title="", fontsize='small', fancybox=True)
    if save:
        plt.savefig(root_folder_path + 'unisvm_1000n_par_worst_err_vs_time.png')
    else:
        plt.show()
    plt.close()


def unisvm2_1000n_par_error_slope_vs_iter_comparison(logs, setup, save=False):
    target_x0 = 20
    target_x = 80

    objfunc = METRICS[setup['obj_function']]
    degrees = {}
    for graph in setup['graphs']:
        degrees[graph] = degree_from_label(graph)
    graph_filter = [
        '*'
    ]

    pred_ratios = {}
    real_slopes = {}
    real_ratios = {}

    # equal to None so that if it will not be reassigned then it will raise exception
    clique_slope = None
    clique_spectral_gap = None

    for graph, graph_objfunc_log in dict(logs['metrics'][objfunc.id]).items():
        if (graph not in graph_filter and '*' not in graph_filter) or graph == '2-expander':
            del logs['metrics'][objfunc.id][graph]
            continue

        y0 = logs['metrics'][objfunc.id][graph][target_x0]
        y = logs['metrics'][objfunc.id][graph][target_x]
        """if degrees[graph] == 2:
            y0 = logs['metrics'][objfunc.id][graph][3000]
            y = logs['metrics'][objfunc.id][graph][3600]"""

        slope = y - y0
        print(slope)

        real_slopes[graph] = slope
        pred_ratios[graph] = 1 / math.sqrt(mtm_spectral_gap_from_adjacency_matrix(setup['graphs'][graph]))

        if 'clique' in graph:
            clique_slope = slope
            clique_spectral_gap = mtm_spectral_gap_from_adjacency_matrix(setup['graphs'][graph])

    for graph in real_slopes:
        real_ratios[graph] = clique_slope / real_slopes[graph]
        pred_ratios[graph] = pred_ratios[graph] * clique_spectral_gap

    plt.suptitle('Fig ZZZ', fontsize=12)
    plt.title('Error vs iteration curve slope vs their theoretical values', loc='left')
    # plt.title("({})".format(setup['time_distr_class'].name), loc='right')
    plt.xlabel('Predicted value')
    plt.ylabel('Measured value')

    l_keys = pred_ratios.keys()
    lx = [pred_ratios[k] for k in l_keys]
    ly = [real_ratios[k] for k in l_keys]
    plt.plot(
        lx,
        ly,
        color='black',
        linewidth=1.5,
        linestyle='dotted'
    )

    for graph, pred_val in pred_ratios.items():
        plt.plot(
            pred_val,
            real_ratios[graph],
            label=degree_from_label(graph),
            marker='o',
            markersize=6,
            markerfacecolor=degree_colors[setup['n']][degree_from_label(graph)],
            markeredgewidth=0.7,
            markeredgecolor='black',
            color='black',
            linewidth=1,
            linestyle='dotted'
        )

    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend(title="", fontsize='small', fancybox=True)
    if save:
        plt.savefig(root_folder_path + 'unisvm2_1000n_par_error_slope_vs_iter_comparison.png')
    else:
        plt.show()
    plt.close()

    """
    for i in range(len(pred_ratios)):
        plt.text(
            pred_ratios[i],
            real_ratios[i],
            'd={}'.format(degrees[list(logs['metrics'][objfunc.id].keys())[i]]),
            size='xx-small'
        )
    """


if __name__ == '__main__':
    run()
