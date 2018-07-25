from matplotlib import pyplot as plt
from src.utils import *
from src.mltoolbox.metrics import METRICS
from src import statistics
import os, math

# root_folder_path = 'C:/Users/grimidev/Dropbox/Share/asynchronous_computing/figures/simulations/'
root_folder_path = 'C:/Users/grimidev/workspace/asyadcml/test_log/' \
                 'test_rslo100_sloreg_100n_exp[1]_mtrT0all_sgC0.01alpha_52000samp_INFtime_600iter/figures/'
# root_folder_path = './figures/'
colors = {
    'black': [0x00, 0x00, 0x00],
    'purple': [0x88, 0x64, 0xff],
    'magenta': [0xff, 0x64, 0xff],
    'red': [0xff, 0x64, 0x64],
    'light-pink': [0xff, 0xb5, 0xd7],
    'light-orange': [0xff, 0xb2, 0x7b],
    'yellow': [0xdc, 0xdf, 0x00],
    'light-green': [0x99, 0xf4, 0x5c],
    'dark-green': [0x66, 0x9c, 0x65],
    'water-green': [0x59, 0xdd, 0xb1],
    'cyan': [0x78, 0xc7, 0xe3],
    'light-blue': [0x72, 0x9c, 0xff],
    'light-purple': [0x9f, 0x81, 0xff],
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
        40: 'water-green',
        50: 'dark-green',
        100: 'light-blue',
        200: 'light-purple',
        # 300: 'light-purple',
        # 400: 'blue',
        500: 'blue',
        999: 'dark-blue'
    }
}

for col in colors:
    rgb = colors[col][:]
    for i in [0, 1, 2]:
        rgb[i] /= 0xff
    colors[col] = rgb

for n in degree_colors:
    Dict = degree_colors[n]
    for deg in Dict:
        Dict[deg] = colors[Dict[deg]]
"""
UNISVM
Fig. YYY
n=1000, c=.1, alpha dep from SG, par(3,2), for expanders
"""


def run():
    save_all()


def save_all():
    svm_1000n_avg_par_logs, svm_1000n_avg_par_setup = load_test_logs(
        './test_log/bulk/da002/test_da002_svm_sgC0.1alpha_1000n_shuf_par[3-2]_mtrT0all_10000time_2000iter'
    )
    reg_1000n_avg_par_logs, reg_1000n_avg_par_setup = load_test_logs(
        './test_log/bulk/r2002/test_r2002_reg2_1000n_par[3-2]_mtrT0all_sgC0.0001alpha_1000samp_100feat_10000time_2000iter'
    )
    svm_1000n_avg_exp_logs, svm_1000n_avg_exp_setup = load_test_logs(
        './test_log/bulk/da002/test_da002_svm_sgC0.1alpha_1000n_shuf_exp[1]_mtrT0all_10000time_2000iter'
    )
    reg_1000n_avg_exp_logs, reg_1000n_avg_exp_setup = load_test_logs(
        './test_log/bulk/r2002/test_r2002_reg2_1000n_exp[1]_mtrT0all_sgC0.0001alpha_1000samp_100feat_10000time_2000iter'
    )
    svm_1000n_avg_unif_logs, svm_1000n_avg_unif_setup = load_test_logs(
        './test_log/bulk/da002/test_da002_svm_sgC0.1alpha_1000n_shuf_unif[0-2]_mtrT0all_10000time_2000iter'
    )
    reg_1000n_avg_unif_logs, reg_1000n_avg_unif_setup = load_test_logs(
        './test_log/bulk/r2002/test_r2002_reg2_1000n_unif[0-2]_mtrT0all_sgC0.0001alpha_1000samp_100feat_10000time_2000iter'
    )
    #"""
    plot_dataset_nodes_distr_iter_vs_time(
        'svm', 1000, 'par', svm_1000n_avg_par_logs, svm_1000n_avg_par_setup, save=True
    )
    plot_dataset_nodes_distr_err_vs_iter(
        'svm', 1000, 'par', 'avg', svm_1000n_avg_par_logs, svm_1000n_avg_par_setup, save=True
    )
    plot_dataset_nodes_distr_err_vs_time(
        'svm', 1000, 'par', 'avg', svm_1000n_avg_par_logs, svm_1000n_avg_par_setup, save=True
    )
    plot_dataset_nodes_distr_err_slope_vs_iter_comparison(
        'svm', 1000, 'par', 'avg', svm_1000n_avg_par_logs, svm_1000n_avg_par_setup, save=True
    )
    plot_dataset_nodes_distr_iter_vs_time(
        'reg', 1000, 'par', reg_1000n_avg_par_logs, reg_1000n_avg_par_setup, save=True
    )
    plot_dataset_nodes_distr_err_vs_iter(
        'reg', 1000, 'par', 'avg', reg_1000n_avg_par_logs, reg_1000n_avg_par_setup, save=True
    )
    plot_dataset_nodes_distr_err_vs_time(
        'reg', 1000, 'par', 'avg', reg_1000n_avg_par_logs, reg_1000n_avg_par_setup, save=True
    )
    plot_dataset_nodes_distr_err_slope_vs_iter_comparison(
        'reg', 1000, 'par', 'avg', reg_1000n_avg_par_logs, reg_1000n_avg_par_setup, save=True
    )
    #"""
    reg_logs_dict = {
        'par': reg_1000n_avg_par_logs,
        'exp': reg_1000n_avg_exp_logs,
        'unif': reg_1000n_avg_unif_logs
    }
    reg_setup_dict = {
        'par': reg_1000n_avg_par_setup,
        'exp': reg_1000n_avg_exp_setup,
        'unif': reg_1000n_avg_unif_setup
    }
    plot_distr_iter_time_vs_degree('reg', 1000, 'avg', reg_logs_dict, reg_setup_dict, save=True)

    svm_logs_dict = {
        'par': svm_1000n_avg_par_logs,
        'exp': svm_1000n_avg_exp_logs,
        'unif': svm_1000n_avg_unif_logs
    }
    svm_setup_dict = {
        'par': svm_1000n_avg_par_setup,
        'exp': svm_1000n_avg_exp_setup,
        'unif': svm_1000n_avg_unif_setup
    }
    plot_distr_iter_time_vs_degree('svm', 1000, 'avg', svm_logs_dict, svm_setup_dict, save=True)


def plot_dataset_nodes_distr_iter_vs_time(dataset, n, distr, logs, setup, save=False):
    # plt.title('Min iteration VS time', loc='left')
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
        plt.savefig(root_folder_path + '{}_{}n_{}_iter_vs_time.png'.format(
            dataset,
            n,
            distr
        ))
    else:
        plt.show()
    plt.close()


def plot_dataset_nodes_distr_err_vs_iter(dataset, n, distr, error, logs, setup, save=False):
    # plt.title('Error VS iterations', loc='left')
    plt.xlabel('Iterations')
    if dataset == 'svm':
        plt.ylabel('Hinge loss')
    elif dataset == 'reg':
        plt.ylabel('Mean Squared Error')

    xlim = -math.inf
    for graph, loss in logs['metrics'][setup['obj_function']].items():
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
    locs, labels = plt.yticks()

    new_labels = []
    for l in labels:
        l = l.get_text()
        if int(l) > 10000:
            l =  "{:.2e}".format(int(l))
        new_labels.append(l)
    plt.yticks(locs, new_labels)

    plt.legend(title="", fontsize='small', fancybox=True)
    if save:
        plt.savefig(root_folder_path + '{}_{}n_{}_{}_err_vs_iter.png'.format(
            dataset,
            n,
            distr,
            error
        ))
    else:
        plt.show()
    plt.close()


def plot_dataset_nodes_distr_err_vs_time(dataset, n, distr, error, logs, setup, save=False):
    # plt.title('Error VS time', loc='left')
    plt.xlabel('Time')
    if dataset == 'svm':
        plt.ylabel('Hinge loss')
    elif dataset == 'reg':
        plt.ylabel('Mean Squared Error')

    xlim = math.inf
    for graph in logs['metrics'][setup['obj_function']]:
        deg = degree_from_label(graph)
        linestyle = '-'

        if n == 1000 and deg == 20:
            linestyle = ':'

        plt.plot(
            logs['iter_time'][graph],
            logs['metrics'][setup['obj_function']][graph],
            label=degree_from_label(graph),
            markersize=2,
            color=degree_colors[setup['n']][deg],
            linestyle=linestyle
        )
        xlim = min(xlim, logs['iter_time'][graph][-1])
    plt.xlim(-50, xlim)
    plt.yscale('linear')
    plt.legend(title="", fontsize='small', fancybox=True)
    if save:
        plt.savefig(root_folder_path + '{}_{}n_{}_{}_err_vs_time.png'.format(
            dataset,
            n,
            distr,
            error
        ))
    else:
        plt.show()
    plt.close()


def plot_dataset_nodes_distr_err_slope_vs_iter_comparison(dataset, n, distr, error, logs, setup, save=False):
    target_x0 = 50
    target_x = 250

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
        # print(slope)

        real_slopes[graph] = slope
        pred_ratios[graph] = 1 / math.sqrt(mtm_spectral_gap_from_adjacency_matrix(setup['graphs'][graph]))

        if 'clique' in graph:
            clique_slope = slope
            clique_spectral_gap = mtm_spectral_gap_from_adjacency_matrix(setup['graphs'][graph])

    for graph in real_slopes:
        real_ratios[graph] = clique_slope / real_slopes[graph]
        pred_ratios[graph] = pred_ratios[graph] * clique_spectral_gap

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
        plt.savefig(root_folder_path + '{}_{}n_{}_{}_err_slope_vs_iter_comparison.png'.format(
            dataset,
            n,
            distr,
            error
        ))
    else:
        plt.show()
    plt.close()


def plot_distr_iter_time_vs_degree(dataset, n, error, logs_dict, setup_dict, save=False):
    ly = {}
    if dataset == 'reg':
        target_err = 700000
    elif dataset =='svm':
        target_err = 0.92

    for distr in ['par', 'exp', 'unif']:
        ly[distr] = {}
        logs = logs_dict[distr]
        setup = setup_dict[distr]
        for graph in setup['graphs']:
            deg = degree_from_label(graph)
            if deg == 2:
                continue
            """
            speed = len(logs['iter_time'][graph]) / logs['iter_time'][graph][-1]
            ly[distr][degree_from_label(graph)] = speed
            """
            t = 0
            e = 0
            for i in range(len(logs['metrics'][setup['obj_function']][graph])):
                e = logs['metrics'][setup['obj_function']][graph][i]
                t = logs['iter_time'][graph][i]
                if e <= target_err:
                    break
            ly[distr][deg] = t

    plt.xlabel('Degree')
    plt.ylabel('Convergence time'.format(target_err))
    plt.yscale('log')
    plt.xscale('log')

    distr_colors = {
        'par': 'red',
        'exp': 'black',
        'unif': 'blue'
    }

    for distr in ['par', 'exp', 'unif']:
        plt.plot(
            ly[distr].keys(),
            ly[distr].values(),
            label=distr,
            markersize=4,
            marker='o',
            color=distr_colors[distr]
        )

    plt.xticks(
        [x for x in ly['exp'].keys()],
        [x for x in ly['exp'].keys()],
        #rotation='vertical',
        size='x-small'
    )

    plt.legend(title="", fontsize='small', fancybox=True)
    if save:
        plt.savefig(root_folder_path + '{}_{}n_{}_distr_iter_time_vs_degree.png'.format(
            dataset,
            n,
            error
        ))
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    run()
