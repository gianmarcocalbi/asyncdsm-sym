from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from src.utils import *
from src.mltoolbox.metrics import METRICS
from src import statistics
import os, math

root_folder_path = 'C:/Users/grimidev/Dropbox/Share/asynchronous_computing/figures/simulations/'
# root_folder_path = './figures/'
colors = {
    'black': [0x00, 0x00, 0x00, 0xff],
    'purple': [0x88, 0x64, 0xff, 0xff],
    'magenta': [0xff, 0x64, 0xff, 0xff],
    'red': [0xff, 0x64, 0x64, 0xff],
    'light-pink': [0xff, 0xb5, 0xd7, 0xff],
    'light-orange': [0xff, 0xb2, 0x7b, 0xff],
    'yellow': [0xcf, 0xc8, 0x00, 0xc8], #[0xdc, 0xdf, 0x00, 0x80],
    'light-green': [0x99, 0xf4, 0x5c, 0xff],
    'green' : [0x00, 0xff, 0x00, 0xff],
    'dark-green': [0x66, 0x9c, 0x65, 0xff],
    'water-green': [0x59, 0xdd, 0xb1, 0xff],
    'cyan': [0x78, 0xc7, 0xe3, 0xff],
    'light-blue': [0x72, 0x9c, 0xff, 0xff],
    'light-purple': [0x9f, 0x81, 0xff, 0xff],
    'blue': [0x46, 0x46, 0xff, 0xff],
    'dark-blue': [0x36, 0x00, 0xab, 0xff],
    'purple-blue': [0xbe, 0xbe, 0xff, 0xff]
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
        16: 'yellow',
        18: 'black',
        20: 'green',
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
    rgba = colors[col][:]
    for i in [0, 1, 2, 3]:
        rgba[i] /= 0xff
    colors[col] = rgba

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
    logs = {
        'synt_reg': {
            'par': None,
            'unif': None,
            'exp': None
        },
        'synt_svm': {
            'par': None,
            'unif': None,
            'exp': None
        },
        'real_reg': {
            'par': None,
            'unif': None,
            'exp': None
        },
        'real_svm': {
            'par': None,
            'unif': None,
            'exp': None
        },
    }
    setup = {
        'synt_reg': {
            'par': None,
            'unif': None,
            'exp': None
        },
        'synt_svm': {
            'par': None,
            'unif': None,
            'exp': None
        },
        'real_reg': {
            'par': None,
            'unif': None,
            'exp': None
        },
        'real_svm': {
            'par': None,
            'unif': None,
            'exp': None
        },
    }
    avg_logs = {
        'synt_reg': {
            'par': None,
            'unif': None,
            'exp': None
        },
        'synt_svm': {
            'par': None,
            'unif': None,
            'exp': None
        },
        'real_reg': {
            'par': None,
            'unif': None,
            'exp': None
        },
        'real_svm': {
            'par': None,
            'unif': None,
            'exp': None
        },
    }
    avg_setup = {
        'synt_reg': {
            'par': None,
            'unif': None,
            'exp': None
        },
        'synt_svm': {
            'par': None,
            'unif': None,
            'exp': None
        },
        'real_reg': {
            'par': None,
            'unif': None,
            'exp': None
        },
        'real_svm': {
            'par': None,
            'unif': None,
            'exp': None
        },
    }


    logs['synt_reg']['par'], setup['synt_reg']['par'] = load_test_logs(
        './test_log/paper/fixedseed_synt_reg/test_r2100_reg2_1000n_par[3-2]_mtrT0all_sgC0.0001alpha_1000samp_100feat_50000time_3000iter')
    logs['synt_reg']['exp'], setup['synt_reg']['exp'] = load_test_logs(
        './test_log/paper/fixedseed_synt_reg/test_r2100_reg2_1000n_exp[1]_mtrT0all_sgC0.0001alpha_1000samp_100feat_50000time_3000iter')
    logs['synt_reg']['unif'], setup['synt_reg']['unif'] = load_test_logs(
        './test_log/paper/fixedseed_synt_reg/test_r2100_reg2_1000n_unif[0-2]_mtrT0all_sgC0.0001alpha_1000samp_100feat_50000time_3000iter')

    logs['synt_svm']['par'], setup['synt_svm']['par'] = load_test_logs(
        './test_log/paper/fixedseed_synt_svm/test_da100_svm_sgC0.1alpha_1000n_par[3-2]_mtrT0all_50000time_3000iter')
    logs['synt_svm']['exp'], setup['synt_svm']['exp'] = load_test_logs(
        './test_log/paper/fixedseed_synt_svm/test_da100_svm_sgC0.1alpha_1000n_exp[1]_mtrT0all_50000time_3000iter')
    logs['synt_svm']['unif'], setup['synt_svm']['unif'] = load_test_logs(
        './test_log/paper/fixedseed_synt_svm/test_da100_svm_sgC0.1alpha_1000n_unif[0-2]_mtrT0all_50000time_3000iter')

    logs['real_reg']['par'], setup['real_reg']['par'] = load_test_logs(
        './test_log/paper/fixedseed_real_reg/test_rslo001_sloreg_1000n_par[3-2]_mtrT0all_sgC5e-06alpha_52000samp_12000time_2400iter')
    logs['real_reg']['exp'], setup['real_reg']['exp'] = load_test_logs(
        './test_log/paper/fixedseed_real_reg/test_rslo001_sloreg_1000n_exp[1]_mtrT0all_sgC5e-06alpha_52000samp_12000time_2400iter')
    logs['real_reg']['unif'], setup['real_reg']['unif'] = load_test_logs(
        './test_log/paper/fixedseed_real_reg/test_rslo001_sloreg_1000n_unif[0-2]_mtrT0all_sgC5e-06alpha_52000samp_12000time_2400iter')

    #"""
    logs['real_svm']['par'], setup['real_svm']['par'] = load_test_logs(
        './test_log/paper/fixedseed_real_svm/test_ss000_susysvm_1000n_par[3-2]_mtrT0all_sgC0.05alpha_500000samp_12000time_2400iter')
    logs['real_svm']['exp'], setup['real_svm']['exp'] = load_test_logs(
        './test_log/paper/fixedseed_real_svm/test_ss000_susysvm_1000n_exp[1]_mtrT0all_sgC0.05alpha_500000samp_12000time_2400iter')
    logs['real_svm']['unif'], setup['real_svm']['unif'] = load_test_logs(
        './test_log/paper/fixedseed_real_svm/test_ss000_susysvm_1000n_unif[0-2]_mtrT0all_sgC0.05alpha_500000samp_12000time_2400iter')

    avg_logs['synt_svm']['par'], avg_setup['synt_svm']['par'] = load_test_logs(
        './test_log/paper/conv_synt_svm/test_conv01_svm_sgC1.0alpha_1000n_par[3-2]_mtrT0all_INFtime_500iter.AVG')
    avg_logs['synt_svm']['exp'], avg_setup['synt_svm']['exp'] = load_test_logs(
        './test_log/paper/conv_synt_svm/test_conv01_svm_sgC1.0alpha_1000n_exp[1]_mtrT0all_INFtime_500iter.AVG')
    avg_logs['synt_svm']['unif'], avg_setup['synt_svm']['unif'] = load_test_logs(
        './test_log/paper/conv_synt_svm/test_conv01_svm_sgC1.0alpha_1000n_unif[0-2]_mtrT0all_INFtime_500iter.AVG')
    
    avg_logs['synt_reg']['par'], avg_setup['synt_reg']['par'] = load_test_logs(
        './test_log/paper/conv_synt_reg/test_conv01_reg2_1000n_par[3-2]_mtrT0all_sgC0.001alpha_1000samp_100feat_INFtime_500iter.AVG')
    avg_logs['synt_reg']['exp'], avg_setup['synt_reg']['exp'] = load_test_logs(
        './test_log/paper/conv_synt_reg/test_conv01_reg2_1000n_exp[1]_mtrT0all_sgC0.001alpha_1000samp_100feat_INFtime_500iter.AVG')
    avg_logs['synt_reg']['unif'], avg_setup['synt_reg']['unif'] = load_test_logs(
        './test_log/paper/conv_synt_reg/test_conv01_reg2_1000n_unif[0-2]_mtrT0all_sgC0.001alpha_1000samp_100feat_INFtime_500iter.AVG')
    
    avg_logs['real_svm']['par'], avg_setup['real_svm']['par'] = load_test_logs(
        './test_log/paper/conv_real_svm/test_ssC00_susysvm_1000n_par[3-2]_mtrT0all_sgC0.05alpha_500000samp_INFtime_1000iter.AVG')
    avg_logs['real_svm']['exp'], avg_setup['real_svm']['exp'] = load_test_logs(
        './test_log/paper/conv_real_svm/test_ssC00_susysvm_1000n_exp[1]_mtrT0all_sgC0.05alpha_500000samp_INFtime_1000iter.AVG')
    avg_logs['real_svm']['unif'], avg_setup['real_svm']['unif'] = load_test_logs(
        './test_log/paper/conv_real_svm/test_ssC00_susysvm_1000n_unif[0-2]_mtrT0all_sgC0.05alpha_500000samp_INFtime_1000iter.AVG')

    avg_logs['real_reg']['par'], avg_setup['real_reg']['par'] = load_test_logs(
        './test_log/paper/conv_real_reg/test_rsloC01_sloreg_1000n_par[3-2]_mtrT0all_sgC5e-06alpha_52000samp_INFtime_500iter.AVG')
    avg_logs['real_reg']['exp'], avg_setup['real_reg']['exp'] = load_test_logs(
        './test_log/paper/conv_real_reg/test_rsloC01_sloreg_1000n_exp[1]_mtrT0all_sgC5e-06alpha_52000samp_INFtime_500iter.AVG')
    avg_logs['real_reg']['unif'], avg_setup['real_reg']['unif'] = load_test_logs(
        './test_log/paper/conv_real_reg/test_rsloC01_sloreg_1000n_unif[0-2]_mtrT0all_sgC5e-06alpha_52000samp_INFtime_500iter.AVG')
    #"""

    #plot_dataset_nodes_distr_iter_vs_time('real_reg', 1000, 'par', logs['real_reg']['par'], setup['real_reg']['par'], save=False)
    #plot_dataset_nodes_distr_err_vs_iter('real_reg', 1000, 'par', logs['real_reg']['par'], setup['real_reg']['par'], save=False)
    #plot_dataset_nodes_distr_err_vs_time('real_reg', 1000, 'par', logs['real_reg']['par'], setup['real_reg']['par'], save=False)
    #plot_dataset_nodes_distr_err_vs_time('real_reg', 1000, 'par', logs['real_reg']['par'], setup['real_reg']['par'], save=False)


    for sim in logs:
        #continue
        for distr in logs[sim]:
            if not logs[sim][distr] is None:
                plot_dataset_nodes_distr_iter_vs_time(sim, 1000, distr, logs[sim][distr], setup[sim][distr], save=True)
                plot_dataset_nodes_distr_err_vs_iter(sim, 1000, distr, logs[sim][distr], setup[sim][distr], save=True)
                plot_dataset_nodes_distr_err_vs_time(sim, 1000, distr, logs[sim][distr], setup[sim][distr], save=True)
        plot_distr_iter_time_vs_degree(sim, 1000, avg_logs[sim], avg_setup[sim], save=True)


def plot_dataset_nodes_distr_iter_vs_time(dataset, n, distr, logs, setup, save=False):
    # plt.title('Min iteration VS time', loc='left')
    plt.xlabel('Time')
    plt.ylabel('Iteration')

    xlim = 0

    for graph, iters in logs['iter_time'].items():
        deg = degree_from_label(graph)
        if distr == 'par' and deg == 20:
            markersize = 4
            marker = '>'
        elif distr == 'par' and 'clique' in graph:
            markersize = 4
            marker = 'x'
        else:
            markersize = 0
            marker = 'o'

        plt.plot(
            iters,
            list(range(len(iters))),
            label=deg,
            markersize=markersize,
            marker=marker,
            markevery=len(iters) // 33,
            color=degree_colors[setup['n']][deg]
        )
        xlim = max(xlim, iters[-1])

    plt.xlim(xmax=xlim)
    plt.yscale('linear')
    plt.legend(title="Degree (d)", fontsize='small', fancybox=True)
    if save:
        dest = root_folder_path + '{}_{}n_{}_iter_vs_time.png'.format(
            dataset,
            n,
            distr
        )
        plt.savefig(dest,bbox_inches='tight')
        print('Create file {}'.format(dest))
    else:
        plt.show()
    plt.close()


def plot_dataset_nodes_distr_err_vs_iter(dataset, n, distr, logs, setup, error='avg', save=False):
    # plt.title('Error VS iterations', loc='left')

    fig, ax = plt.subplots()

    plt.xlabel('Iterations')
    if 'svm' in dataset:
        plt.ylabel('Hinge loss')
    elif 'reg' in dataset:
        plt.ylabel('Mean Squared Error')

    ylim_up = -math.inf
    ylim_dw = math.inf
    xlim = -math.inf

    for graph, loss in logs['metrics'][setup['obj_function']].items():
        deg = degree_from_label(graph)
        zorder = 1
        if distr == 'par' and deg == 20:
            markersize = 4
            marker = '>'
            zorder = -1
        elif distr == 'par' and 'clique' in graph:
            markersize = 4
            marker = 'x'
        else:
            markersize = 0
            marker = 'o'

        plt.plot(
            list(range(len(loss))),
            loss,
            label=deg,
            markersize=markersize,
            marker = marker,
            markevery = len(loss) // 33,
            color=degree_colors[setup['n']][deg],
            zorder = zorder
        )
        if degree_from_label(graph) >= 4:
            xlim = max(xlim, len(loss))
        ylim_up = max(ylim_up, loss[0])
        ylim_dw = min(ylim_dw, loss[-1])
    plt.xlim(-20, xlim)
    plt.yscale('linear')

    if 'reg' in dataset:
        locs, _ = plt.yticks()
        labels = []
        for y in locs:
            if y >= 100000:
                _n = "$\\mathdefault{{{0}\\times10^{{{1}}}}}$".format(
                    round(y / (10**int(math.log10(y))), 1),
                    int(math.log10(y))
                )
            else:
                _n = y
            labels.append(_n)

        plt.yticks(
            locs,
            labels,
            # rotation='vertical',
            size='x-small'
        )

    ylim_padding = (ylim_up - ylim_dw) / 20
    plt.ylim(max(ylim_dw-ylim_padding, 0), ylim_up+ylim_padding)

    plt.legend(title="Degree (d)", fontsize='small', fancybox=True)
    if save:
        dest = root_folder_path + '{}_{}n_{}_{}_err_vs_iter.png'.format(
            dataset,
            n,
            distr,
            error
        )
        plt.savefig(dest,bbox_inches='tight')
        print('Create file {}'.format(dest))
    else:
        plt.show()
    plt.close()


def plot_dataset_nodes_distr_err_vs_time(dataset, n, distr, logs, setup, error='avg', save=False):
    fig, ax = plt.subplots()
    # plt.title('Error VS time', loc='left')
    plt.xlabel('Time')
    if 'svm' in dataset:
        plt.ylabel('Hinge loss')
    elif 'reg' in dataset:
        plt.ylabel('Mean Squared Error')
    zoom_region = False
    x1, x2, y1, y2 = 0,0,0,0
    zoom_loc = 1
    legend_loc = 0
    markevery = 33
    if dataset == 'synt_reg':
        if distr == 'par':
            x1, x2 = 2500, 4500
            y1, y2 = 630000, 750000
            legend_loc = 5
            zoom_loc = 3
            markevery = 60
            zoom_region = True
        elif distr == 'unif':
            x1, x2 = 1200, 2000
            y1, y2 = 200000, 340000
            legend_loc = 5
            zoom_loc = 9
            zoom_region = True
    elif dataset == 'synt_svm':
        if distr == 'par':
            x1, x2 = 6000, 8500
            y1, y2 = 0.8, 0.9
            legend_loc = 5
            zoom_loc = 3
            zoom_region = True
    elif dataset == 'real_reg':
        if distr == 'exp':
            x1, x2 = 800, 1800
            y1, y2 = 10000, 15000
            legend_loc = 5
            zoom_loc = 9
            zoom_region = True
        elif distr == 'par':
            x1, x2 = 2200, 3400
            y1, y2 = 12800, 16800
            legend_loc = 5
            zoom_loc = 9
            zoom_region = True
        elif distr == 'unif':
            x1, x2 = 360, 1000
            y1, y2 = 7600, 14000
            legend_loc = 5
            zoom_loc = 9
            zoom_region = True
    elif dataset == 'real_svm':
        if distr == 'par':
            x1, x2 = 800, 2200
            y1, y2 = 0.216, 0.225
            legend_loc = 5
            zoom_loc = 3
            zoom_region = True

    axins = None
    if zoom_region:
        axins = zoomed_inset_axes(ax, 3, loc=zoom_loc)
    ylim_up = -math.inf
    ylim_dw = math.inf
    xlim = math.inf
    for graph in logs['metrics'][setup['obj_function']]:
        deg = degree_from_label(graph)
        zorder = 2
        if distr == 'par' and deg == 20:
            markersize = 4
            marker = '>'
            zorder = 10
        elif distr == 'par' and 'clique' in graph:
            markersize = 4
            marker = 'x'
        else:
            markersize = 0
            marker = 'o'

        ax.plot(
            logs['iter_time'][graph],
            logs['metrics'][setup['obj_function']][graph],
            label=degree_from_label(graph),
            color=degree_colors[setup['n']][deg],
            markersize=markersize,
            marker = marker,
            markevery = len(logs['iter_time'][graph]) // markevery,
            zorder = zorder
        )

        if zoom_region:
            axins.plot(
                logs['iter_time'][graph],
                logs['metrics'][setup['obj_function']][graph],
                label=degree_from_label(graph),
                color=degree_colors[setup['n']][deg],
                markersize=markersize,
                marker=marker,
                markevery=len(logs['iter_time'][graph]) // 33,
                zorder=zorder
            )

        xlim = min(xlim, logs['iter_time'][graph][-1])
        ylim_up = max(ylim_up, logs['metrics'][setup['obj_function']][graph][0])
        ylim_dw = min(ylim_dw, logs['metrics'][setup['obj_function']][graph][-1])

    if zoom_region:
        axins.set_xlim(x1, x2)  # apply the x-limits
        axins.set_ylim(y1, y2)  # apply the y-limits
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", zorder=100)

    ax.set_xlim(-50, xlim)
    plt.yscale('linear')


    if 'reg' in dataset:
        locs, _ = plt.yticks()
        labels = []
        for y in locs:
            if y >= 100000:
                _n = "$\\mathdefault{{{0}\\times10^{{{1}}}}}$".format(
                    round(y / (10 ** int(math.log10(y))),1),
                    int(math.log10(y))
                )
            else:
                _n = y
            labels.append(_n)

        plt.yticks(
            locs,
            labels,
            # rotation='vertical',
            size='x-small'
        )

    ylim_padding = (ylim_up - ylim_dw) / 30
    ax.set_ylim(max(ylim_dw - ylim_padding, 0), ylim_up + ylim_padding)

    ax.legend(title="Degree (d)", fontsize='small', fancybox=True, loc=legend_loc)
    if save:
        dest = root_folder_path + '{}_{}n_{}_{}_err_vs_time.png'.format(
            dataset,
            n,
            distr,
            error
        )
        plt.savefig(dest,bbox_inches='tight')
        print('Create file {}'.format(dest))
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
        dest = root_folder_path + '{}_{}n_{}_{}_err_slope_vs_iter_comparison.png'.format(
            dataset,
            n,
            distr,
            error
        )
        plt.savefig(dest,bbox_inches='tight')
        print('Create file {}'.format(dest))
    else:
        plt.show()
    plt.close()


def plot_distr_iter_time_vs_degree(dataset, n, logs_dict, setup_dict, error='avg', save=False):
    ly = {}
    if dataset == 'synt_reg':
        target_err = 150000
    elif dataset == 'synt_svm':
        target_err = 0.60
    elif dataset == 'real_reg':
        target_err = 15000
    elif dataset == 'real_svm':
        target_err = 0.20

    ylim_up = -math.inf
    ylim_dw = math.inf

    for distr in logs_dict:
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
            ylim_up = max(t, ylim_up)
            ylim_dw = min(ylim_dw, t)


    plt.xlabel('Degree (d)')
    plt.ylabel('Convergence time'.format(target_err))
    plt.yscale('log')
    plt.xscale('log')

    distr_colors = {
        'par': 'red',
        'exp': 'black',
        'unif': 'blue'
    }

    for distr in ['par','exp','unif']:
        plt.plot(
            ly[distr].keys(),
            ly[distr].values(),
            label=distr,
            markersize=4,
            marker='o',
            color=distr_colors[distr]
        )

    ylim_padding = (ylim_up - ylim_dw) / 20
    #plt.ylim(ymax=ylim_up + ylim_padding)

    y_tick_values = []
    k = 0
    while 10 ** k <= ylim_up and k < 100:
        if ylim_dw <= 10 ** k:
            y_tick_values.append(10 ** k)
        k += 1

    if len(y_tick_values) > 0:
        if len(y_tick_values) == 1:
            tick = y_tick_values[0]
            y_tick_values.append(math.floor(ylim_up/tick) * tick)
        y_tick_values.sort()
        #'$\\mathdefault{10^{4}}$'
        plt.yticks(
            y_tick_values,
            ["$\\mathdefault{{{0}\\times10^{{{1}}}}}$".format(str(_n)[0], int(math.log10(_n))) for _n in y_tick_values],
            #rotation='vertical',
            size='x-small'
        )

    plt.xticks(
        [x for x in ly['exp'].keys()],
        [x for x in ly['exp'].keys()],
        # rotation='vertical',
        size='x-small'
    )

    plt.legend(title="", fontsize='small', fancybox=True)
    if save:
        plt.savefig(root_folder_path + '{}_{}n_{}_distr_iter_time_vs_degree.png'.format(
            dataset,
            n,
            error
        ),bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    run()
