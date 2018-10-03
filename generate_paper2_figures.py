from matplotlib import rcParams
#rcParams['font.family'] = 'serif'
#rcParams['font.sans-serif'] = ['Times New Roman']
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from src.utils import *

"""
Location Legend:
'best'	        0
'upper right'	1
'upper left'	2
'lower left'	3
'lower right'	4
'right'	        5
'center left'	6
'center right'	7
'lower center'	8
'upper center'	9
'center'	    10
"""

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# root_folder_path = 'C:/Users/grimidev/Dropbox/Mine/Skull/Uni/Erasmus/Report and Thesis/report/figures/simulations/'
root_folder_path = 'C:/Users/grimidev/Dropbox/Share/asynchronous_computing/followup/figures/simulations/'
# root_folder_path = './figures/paper2/'

old_colors = {
    'black': [0x00, 0x00, 0x00, 0xff],
    'purple': [0x88, 0x64, 0xff, 0xff],
    'magenta': [0xff, 0x64, 0xff, 0xff],
    'red': [0xff, 0x64, 0x64, 0xff],
    'light-pink': [0xff, 0xb5, 0xd7, 0xff],
    'light-orange': [0xff, 0xb2, 0x7b, 0xff],
    'yellow': [0xcf, 0xc8, 0x00, 0xc8],  # [0xdc, 0xdf, 0x00, 0x80],
    'light-green': [0x99, 0xf4, 0x5c, 0xff],
    'green': [0x00, 0xff, 0x00, 0xff],
    'dark-green': [0x66, 0x9c, 0x65, 0xff],
    'water-green': [0x59, 0xdd, 0xb1, 0xff],
    'cyan': [0x78, 0xc7, 0xe3, 0xff],
    'light-blue': [0x72, 0x9c, 0xff, 0xff],
    'light-purple': [0x9f, 0x81, 0xff, 0xff],
    'blue': [0x46, 0x46, 0xff, 0xff],
    'dark-blue': [0x36, 0x00, 0xab, 0xff],
    'purple-blue': [0xbe, 0xbe, 0xff, 0xff]
}

"""
Red         #e6194b     [230, 25, 75, 255]       [0, 100, 66, 0]
Green       #3cb44b     [60, 180, 75, 255]       [75, 0, 100, 0]
Yellow      #ffe119     [255, 225, 25, 255]      [0, 25, 95, 0]
Blue        #0082c8     [0, 130, 200, 255]       [100, 35, 0, 0]
Orange      #f58231     [245, 130, 48, 255]      [0, 60, 92, 0]
Purple      #911eb4     [145, 30, 180, 255]      [35, 70, 0, 0]
Cyan        #46f0f0     [70, 240, 240, 255]      [70, 0, 0, 0]
Magenta     #f032e6     [240, 50, 230, 255]      [0, 100, 0, 0]
Lime        #d2f53c     [210, 245, 60, 255]      [35, 0, 100, 0]
Pink        #fabebe     [250, 190, 190, 255]     [0, 30, 15, 0]
Teal        #008080     [0, 128, 128, 255]       [100, 0, 0, 50]
Lavender    #e6beff     [230, 190, 255, 255]     [10, 25, 0, 0]
Brown       #aa6e28     [170, 110, 40, 255]      [0, 35, 75, 33]
Beige       #fffac8     [255, 250, 200, 255]     [5, 10, 30, 0]
Maroon      #800000     [128, 0, 0, 255]         [0, 100, 100, 50]
Mint        #aaffc3     [170, 255, 195, 255]     [33, 0, 23, 0]
Olive       #808000     [128, 128, 0, 255]       [0, 0, 100, 50]
Coral       #ffd8b1     [255, 215, 180, 255]     [0, 15, 30, 0]
Grey        #000080     [0, 0, 128, 255]         [100, 100, 0, 50]
Navy        #808080     [128,128,128, 255]       [0, 0, 0, 50]
White       #FFFFFF     [255,255,255, 255]       [0, 0, 0, 0]
Black       #000000     [0, 0, 0, 255]           [0, 0, 0, 100]
"""

colors = {
    'purple': [0x91, 0x1e, 0xb4, 0xff],
    'magenta': [0xf0, 0x32, 0xe6, 0xff],
    'red': [0xe6, 0x19, 0x4b, 0xff],
    'lavender': [0xe6, 0xbe, 0xff, 0xff],
    'pink': [250, 190, 190, 255],
    'orange': [245, 130, 48, 255],
    'coral': [255, 215, 180, 255],
    'yellow': [255, 225, 25, 255],
    'lime': [210, 245, 60, 255],
    'mint': [170, 255, 195, 255],
    'cyan': [70, 240, 240, 255],
    'blue': [0, 130, 200, 255],
    'navy': [0, 0, 128, 255],
    'teal': [0, 128, 128, 255],
    'green': [60, 180, 75, 255],
    'olive': [128, 128, 0, 255],
    'brown': [170, 110, 40, 255],
    'maroon': [128, 0, 0, 255],
    'beige': [255, 250, 200, 255],
    'grey': [128, 128, 128, 255],
    'black': [0, 0, 0, 255],
}

degree_colors = {
    100: {
        0: 'black',
        1: 'purple',
        2: 'magenta',
        3: 'red',
        4: 'orange',
        8: 'yellow',
        16: 'cyan',
        32: 'blue',
        64: 'green',
        80: 'brown',
        99: 'maroon'
    },
    1000: {
        2: 'purple',  # 'magenta',
        3: 'magenta',  # 'red',
        4: 'red',  # 'light-pink',
        8: 'lavender',  # 'light-orange',
        16: 'pink',  # 'yellow',
        18: 'black',
        20: 'orange',  # 'green',
        30: 'yellow',  # 'light-green',
        40: 'cyan',  # 'water-green',
        50: 'blue',  # 'dark-green',
        100: 'grey',  # 'light-blue',
        200: 'green',  # 'light-purple',
        # 300: 'light-purple',
        # 400: 'blue',
        500: 'brown',  # 'blue',
        999: 'maroon',  # 'dark-blue'
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
    active_tests = [
        'test1',
        'test2',
        'test3',
        'test4',
        'test5',
        'test6_svm',
        'test6_reg'
    ]
    log, setup = load(active_tests)
    # plot_dataset_nodes_distr_err_vs_iter(
    #    'test6_svm', 'susysvm', 100, log['test6_svm']['spark'], setup['test6_svm']['spark'], n_samples=100, save=False)
    # plot_dataset_nodes_distr_err_vs_time(
    #    'test6_svm', 'real_svm', 100, 'spark', log['test6_svm']['spark'], setup['test6_svm']['spark'], n_samples=100,
    #    save=False)
    # plot_dataset_nodes_distr_err_vs_time(
    #    'test6_reg', 'real_reg', 100, 'spark', log['test6_reg']['spark'], setup['test6_reg']['spark'], n_samples=100,
    #    save=False)

    plot_all(log, setup, active_tests)


def plot_all(log, setup, active_tests):
    distributions = ['exp', 'unif', 'par', 'spark', 'custom']

    for test in active_tests:
        if test in ['test1', 'test2', 'test3']:
            plot_dataset_nodes_distr_err_vs_iter(test, 'eigvecsvm', 100, log[test]['exp'], setup[test]['exp'],
                save=True)
        elif test == 'test4':
            plot_dataset_nodes_distr_err_vs_iter(
                test, 'multieigvecsvm', 100, log[test]['par'][2], setup[test]['par'][2], n_samples=2, save=True)
            plot_dataset_nodes_distr_err_vs_iter(
                test, 'multieigvecsvm', 100, log[test]['par'][10], setup[test]['par'][10], n_samples=10, save=True)
            plot_dataset_nodes_distr_err_vs_iter(
                test, 'multieigvecsvm', 100, log[test]['par'][100], setup[test]['par'][100], n_samples=100, save=True)
        elif test == 'test5':
            plot_dataset_nodes_distr_err_vs_iter(
                test, 'multieigvecsvm', 100, log[test]['par'][100], setup[test]['par'][100], n_samples=100, save=True)
        elif test in ['test6_reg', 'test6_svm']:
            dataset = {'test6_reg' : 'real_reg', 'test6_svm' : 'real_svm'}[test]
            plot_dataset_nodes_distr_iter_vs_time(
                test, dataset, 100, 'spark', log[test]['spark'], setup[test]['spark'], save=True)
            plot_dataset_nodes_distr_err_vs_iter(
                test, dataset, 100, log[test]['spark'], setup[test]['spark'], n_samples=100, save=True)
            plot_dataset_nodes_distr_err_vs_time(
                test, dataset, 100, 'spark', log[test]['spark'], setup[test]['spark'], n_samples=100, save=True)


def load(active_tests):
    log = {
        'test1': {'par': None, 'unif': None, 'exp': None},
        'test2': {'par': None, 'unif': None, 'exp': None},
        'test3': {'par': None, 'unif': None, 'exp': None},
        'test4': {'par': {}, 'unif': None, 'exp': None},
        'test5': {'par': {}, 'unif': None, 'exp': None},
        'test6_reg': {'spark': None},
        'test6_svm': {'spark': None}
    }
    setup = {
        'test1': {'par': None, 'unif': None, 'exp': None},
        'test2': {'par': None, 'unif': None, 'exp': None},
        'test3': {'par': None, 'unif': None, 'exp': None},
        'test4': {'par': {}, 'unif': None, 'exp': None},
        'test5': {'par': {}, 'unif': None, 'exp': None},
        'test6_reg': {'spark': None},
        'test6_svm': {'spark': None}
    }

    if 'test1' in active_tests:
        log['test1']['exp'], setup['test1']['exp'] = load_test_logs(
            './test_log/paper2/test1/test_1undir_cycle_eigvecsvm_C0.1alpha_100n_noshuf_Win[1,1]_exp[1]_INFtime_5000iter_mtrT2worst')
        log['test1']['par'], setup['test1']['par'] = load_test_logs(
            './test_log/paper2/test1/test_1undir_cycle_eigvecsvm_C0.1alpha_100n_noshuf_Win[1,1]_par[3-2]_INFtime_5000iter_mtrT2worst')
        log['test1']['unif'], setup['test1']['unif'] = load_test_logs(
            './test_log/paper2/test1/test_1undir_cycle_eigvecsvm_C0.1alpha_100n_noshuf_Win[1,1]_unif[0-2]_INFtime_5000iter_mtrT2worst')

    if 'test2' in active_tests:
        log['test2']['exp'], setup['test2']['exp'] = load_test_logs(
            './test_log/paper2/test2/test_2expander_eigvecsvm_C0.1alpha_100n_noshuf_Win[1,1]_exp[1]_INFtime_5000iter_mtrT2worst')
        log['test2']['par'], setup['test2']['par'] = load_test_logs(
            './test_log/paper2/test2/test_2expander_eigvecsvm_C0.1alpha_100n_noshuf_Win[1,1]_par[3-2]_INFtime_5000iter_mtrT2worst')
        log['test2']['unif'], setup['test2']['unif'] = load_test_logs(
            './test_log/paper2/test2/test_2expander_eigvecsvm_C0.1alpha_100n_noshuf_Win[1,1]_unif[0-2]_INFtime_5000iter_mtrT2worst')

    if 'test3' in active_tests:
        log['test3']['exp'], setup['test3']['exp'] = load_test_logs(
            './test_log/paper2/test3/test_3alt_expander_alteigvecsvm_C0.1alpha_100n_noshuf_Win[1,1]_exp[1]_INFtime_5000iter_mtrT2worst')
        log['test3']['par'], setup['test3']['par'] = load_test_logs(
            './test_log/paper2/test3/test_3alt_expander_alteigvecsvm_C0.1alpha_100n_noshuf_Win[1,1]_par[3-2]_INFtime_5000iter_mtrT2worst')
        log['test3']['unif'], setup['test3']['unif'] = load_test_logs(
            './test_log/paper2/test3/test_3alt_expander_alteigvecsvm_C0.1alpha_100n_noshuf_Win[1,1]_unif[0-2]_INFtime_5000iter_mtrT2worst')

    if 'test4' in active_tests:
        log['test4']['par'][2], setup['test4']['par'][2] = load_test_logs(
            './test_log/paper2/test4/test_test4_multieigvecsvm_2samp_C0.1alpha_100n_Win[1,1]_par[3-2]_5000iter')
        log['test4']['par'][10], setup['test4']['par'][10] = load_test_logs(
            './test_log/paper2/test4/test_test4_multieigvecsvm_10samp_C0.1alpha_100n_Win[1,1]_par[3-2]_5000iter')
        log['test4']['par'][100], setup['test4']['par'][100] = load_test_logs(
            './test_log/paper2/test4/test_test4_multieigvecsvm_100samp_C0.1alpha_100n_Win[1,1]_par[3-2]_5000iter')

    if 'test5' in active_tests:
        log['test5']['par'][2], setup['test5']['par'][2] = load_test_logs(
            './test_log/paper2/test5/test_test5_3-multieigvecsvm_2samp_C0.1alpha_100n_Win[1,1]_par[3-2]_5000iter')
        log['test5']['par'][10], setup['test5']['par'][10] = load_test_logs(
            './test_log/paper2/test5/test_test5_3-multieigvecsvm_10samp_C0.1alpha_100n_Win[1,1]_par[3-2]_5000iter')
        log['test5']['par'][100], setup['test5']['par'][100] = load_test_logs(
            './test_log/paper2/test5/test_test5_3-multieigvecsvm_100samp_C0.1alpha_100n_Win[1,1]_par[3-2]_5000iter')

    if 'test6_svm' in active_tests:
        log['test6_svm']['spark'], setup['test6_svm']['spark'] = load_test_logs(
            './test_log/paper2/test6/test_test6_susysvm_100n_spark_real[None]_mtrT2worst_C0.05alpha_500000samp_INFtime_6000iter')
    if 'test6_reg' in active_tests:
        log['test6_reg']['spark'], setup['test6_reg']['spark'] = load_test_logs(
            './test_log/paper2/test6/test_test6_sloreg_100n_spark_real[None]_mtrT2worst_C5e-06alpha_52000samp_INFtime_5000iter')

    return log, setup


def plot_dataset_nodes_distr_err_vs_iter(test, dataset, n, logs, setup, n_samples=2, save=False):
    fig, ax = plt.subplots()
    # plt.title('Error VS iterations', loc='left')

    plt.xlabel('Iterations')
    # plt.ylabel(r'$F(\bar w_i)$')

    if test in ['test1', 'test2', 'test3', 'test4']:
        plt.ylabel(r'$F(\hat w_j(k))$')
    elif 'svm' in dataset:
        plt.ylabel('Hinge Loss')
    elif 'reg' in dataset:
        plt.ylabel('Mean Squared Error')

    zoom_region = False
    x1, x2, y1, y2 = 0, 0, 0, 0
    loc1, loc2 = 2, 4
    zoom_loc = 1
    zoom_scale = 3
    legend_loc = 0
    markevery = 0.05
    bbox_to_anchor = False

    if test == 'test1':
        x1, x2 = 960, 1050
        y1, y2 = 0.600, 0.665
        legend_loc = 5
        zoom_loc = 3
        zoom_region = True
        zoom_scale = 16
        markevery = 0.05
        loc1 = 1
        loc2 = 2
        bbox_to_anchor = (0.08, 0.06)
    elif test == 'test2':
        x1, x2 = 980, 1020
        y1, y2 = 0.605, 0.715
        legend_loc = 5
        zoom_loc = 9
        zoom_region = True
        zoom_scale = 40
        markevery = 0.05
    elif test == 'test3':
        x1, x2 = 980, 1020
        y1, y2 = 0.606, 0.641
        legend_loc = 1
        zoom_loc = 3
        zoom_region = True
        zoom_scale = 42
        markevery = 0.05
        loc1 = 1
        loc2 = 2
        bbox_to_anchor = (0.1, 0.1)
    elif test == 'test4':
        if n_samples == 2:
            x1, x2 = 980, 1020
            y1, y2 = 0.595, 0.63  # y1, y2 = 0.606, 0.641
            legend_loc = 1
            zoom_loc = 3
            zoom_region = True
            zoom_scale = 40
            markevery = 0.05
            loc1 = 1
            loc2 = 2
            bbox_to_anchor = (0.1, 0.1)
        elif n_samples == 10:
            x1, x2 = 980, 1020
            y1, y2 = 0.595, 0.63  # y1, y2 = 0.60, 0.615
            legend_loc = 1
            zoom_loc = 3
            zoom_region = True
            zoom_scale = 40
            markevery = 0.05
            loc1 = 1
            loc2 = 2
            bbox_to_anchor = (0.1, 0.1)
        elif n_samples == 100:
            x1, x2 = 980, 1020
            y1, y2 = 0.595, 0.63  # y1, y2 = 0.595, 0.61
            legend_loc = 1
            zoom_loc = 3
            zoom_region = True
            zoom_scale = 40
            markevery = 0.05
            loc1 = 1
            loc2 = 2
            bbox_to_anchor = (0.1, 0.1)
    if test == 'test5':
        if n_samples == 100:
            x1, x2 = 950, 1050
            y1, y2 = 0.8765, 0.8855
            legend_loc = 1
            zoom_loc = 9
            zoom_region = True
            zoom_scale = 18
            markevery = 0.05
            loc1 = 2
            loc2 = 4
            # bbox_to_anchor = (0.1, 0.1)
    if test == 'test6_svm':
        if n_samples == 100:
            x1, x2 = 0, 1000
            y1, y2 = 0.2, 0.3
            legend_loc = 1
            zoom_loc = 9
            zoom_region = False
            zoom_scale = 4
            markevery = 0.05
            loc1 = 2
            loc2 = 4
            # bbox_to_anchor = (0.1, 0.1)

    axins = None
    if zoom_region:
        if bbox_to_anchor is False:
            axins = zoomed_inset_axes(
                ax,
                zoom_scale,
                loc=zoom_loc
            )
        else:
            axins = zoomed_inset_axes(
                ax,
                zoom_scale,
                loc=zoom_loc,
                bbox_to_anchor=bbox_to_anchor,
                bbox_transform=ax.transAxes
            )

    for graph, loss in logs['metrics'][setup['obj_function']].items():
        deg = degree_from_label(graph)
        zorder = 1
        if deg == 2:  # and distr == 'par':
            markersize = 6
            marker = 'x'
            zorder = 50
        elif 'clique' in graph:
            markersize = 5
            marker = 'o'
            zorder = 80
        else:
            markersize = 0
            marker = 'o'

        ax.plot(
            list(range(len(loss))),
            loss,
            label=deg,
            markersize=markersize,
            marker=marker,
            markevery=markevery,
            color=degree_colors[setup['n']][deg],
            zorder=zorder
        )

        if zoom_region:
            axins.plot(
                list(range(len(loss))),
                loss,
                label=deg,
                color=degree_colors[setup['n']][deg],
                markersize=markersize,
                marker=marker,
                markevery=0.14,
                zorder=zorder
            )

    if zoom_region:
        axins.set_xlim(x1, x2)  # apply the x-limits
        axins.set_ylim(y1, y2)  # apply the y-limits
        plt.yticks(visible=True, fontsize='x-small')
        plt.xticks(visible=True, fontsize='x-small')
        mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="0.5", zorder=100)

    plt.yscale('linear')

    ax.legend(title="Degree (d)", fontsize='small', fancybox=True, loc=legend_loc)
    if save:
        if dataset == 'multieigvecsvm':
            dest = root_folder_path + '{}_{}_{}n_{}samples_err_vs_iter.png'.format(
                test,
                dataset,
                n,
                n_samples
            )
        else:
            dest = root_folder_path + '{}_{}_{}n_err_vs_iter.png'.format(
                test,
                dataset,
                n
            )
        plt.savefig(dest, bbox_inches='tight')
        print('Create file {}'.format(dest))
    else:
        plt.show()
    plt.close()


def plot_dataset_nodes_distr_err_vs_time(test, dataset, n, distr, logs, setup, n_samples=2, save=False):
    fig, ax = plt.subplots()

    # plt.title('Error VS time', loc='left')
    # plt.title(distr, loc='Right')

    plt.xlabel('Time')

    if test in ['test1', 'test2', 'test3', 'test4']:
        plt.ylabel(r'$F(\hat w_j(k))$')
    elif 'svm' in dataset:
        plt.ylabel('Hinge Loss')
    elif 'reg' in dataset:
        plt.ylabel('Mean Squared Error')

    zoom_region = False
    x1, x2, y1, y2 = 0, 0, 0, 0
    loc1, loc2 = 2, 4
    zoom_loc = 1
    zoom_scale = 3
    legend_loc = 0
    markevery = 0.05
    bbox_to_anchor = False
    xlim1, xlim2, ylim1, ylim2 = None, None, None, None

    if dataset == 'eigvecsvm':
        if True:
            x1, x2 = 6500, 10500
            y1, y2 = -0.3, 0.5
            legend_loc = 5
            zoom_loc = 10
            markevery = 0.025
            zoom_region = True
    if test == 'test6_svm':
        if n_samples == 100:
            x1, x2 = 1e6, 3e6
            y1, y2 = 0.145, 0.2
            legend_loc = 1
            zoom_loc = 10
            zoom_region = False
            zoom_scale = 5
            markevery = 0.05
            loc1 = 2
            loc2 = 4
            bbox_to_anchor = (0.45, 0.55)
            xlim1 = -40000
            xlim2 = 1e6
            ylim1 = 0.17
            ylim2 = 0.716
    if test == 'test6_reg':
        if n_samples == 100:
            x1, x2 = 150000, 270000
            y1, y2 = 11000, 16000
            legend_loc = 1
            zoom_loc = 10
            zoom_region = True
            zoom_scale = 3.8
            markevery = 0.05
            loc1 = 2
            loc2 = 4
            bbox_to_anchor = (0.5, 0.68)
            xlim1 = -40000
            xlim2 = 1e6
            ylim1 = 5500
            ylim2 = 47000

    axins = None
    if zoom_region:
        if bbox_to_anchor is False:
            axins = zoomed_inset_axes(
                ax,
                zoom_scale,
                loc=zoom_loc
            )
        else:
            axins = zoomed_inset_axes(
                ax,
                zoom_scale,
                loc=zoom_loc,
                bbox_to_anchor=bbox_to_anchor,
                bbox_transform=ax.transAxes
            )

    # ylim_up = -math.inf
    # ylim_dw = math.inf
    # xlim = math.inf

    for graph in logs['metrics'][setup['obj_function']]:
        deg = degree_from_label(graph)
        zorder = 2
        if deg == 2:  # and distr == 'par':
            markersize = 6
            marker = 'x'
            zorder = 10
        elif 'clique' in graph:
            markersize = 5
            marker = 'o'
            zorder = 12
        else:
            markersize = 0
            marker = 'o'

        ax.plot(
            logs['iter_time'][graph],
            logs['metrics'][setup['obj_function']][graph],
            label=degree_from_label(graph),
            color=degree_colors[setup['n']][deg],
            markersize=markersize,
            marker=marker,
            markevery=markevery,
            zorder=zorder
        )

        if zoom_region:
            axins.plot(
                logs['iter_time'][graph],
                logs['metrics'][setup['obj_function']][graph],
                label=degree_from_label(graph),
                color=degree_colors[setup['n']][deg],
                markersize=markersize,
                marker=marker,
                markevery=0.14,
                zorder=zorder
            )

        # xlim = min(xlim, logs['iter_time'][graph][-1])
        # ylim_up = max(ylim_up, logs['metrics'][setup['obj_function']][graph][0])
        # ylim_dw = min(ylim_dw, logs['metrics'][setup['obj_function']][graph][-1])

    if zoom_region:
        axins.set_xlim(x1, x2)  # apply the x-limits
        axins.set_ylim(y1, y2)  # apply the y-limits
        plt.yticks(visible=True, fontsize='x-small')
        plt.xticks(visible=True, fontsize='x-small')
        mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="0.5", zorder=100)

    if xlim1 is not None and xlim2 is not None:
        ax.set_xlim(xlim1, xlim2)
    if ylim1 is not None and ylim2 is not None:
        ax.set_ylim(ylim1, ylim2)

    # plt.yscale('linear')

    """locs, _ = plt.yticks()
    labels = []
    for y in locs:
        if y >= 100000:
            int_part = round(y / (10 ** int(math.log10(y))), 1)
            if int_part.is_integer():
                int_part = int(int_part)
            _n = "$\\mathdefault{{{0}\\times10^{{{1}}}}}$".format(
                int_part,
                int(math.log10(y))
            )
        else:
            _n = y
            if float(_n).is_integer():
                _n = int(_n)
            else:
                _n = round(float(_n), 3)
        labels.append(_n)

    plt.yticks(
        locs,
        labels,
        # rotation='vertical',
        # size='small'
    )
    """
    # ylim_padding = (ylim_up - ylim_dw) / 30
    # ax.set_ylim(max(ylim_dw - ylim_padding, 0), ylim_up + ylim_padding)

    ax.legend(title="Degree (d)", fontsize='small', fancybox=True, loc=legend_loc)
    if save:
        dest = root_folder_path + '{}_{}_{}n_{}_err_vs_time.png'.format(
            test,
            dataset,
            n,
            distr
        )
        plt.savefig(dest, bbox_inches='tight')
        print('Create file {}'.format(dest))
    else:
        plt.show()
    plt.close()


def plot_dataset_nodes_distr_iter_vs_time(test, dataset, n, distr, logs, setup, save=False):
    # plt.title('Min iteration VS time', loc='left')
    plt.xlabel('Time')
    plt.ylabel('Iteration')

    xlim = 0

    for graph, iters in logs['iter_time'].items():
        deg = degree_from_label(graph)
        if deg == 2:  # and distr == 'par':
            markersize = 6
            marker = 'x'
        elif 'clique' in graph:
            markersize = 5
            marker = 'o'
        else:
            markersize = 0
            marker = 'o'

        plt.plot(
            iters,
            list(range(len(iters))),
            label=deg,
            markersize=markersize,
            marker=marker,
            markevery=0.05,
            color=degree_colors[setup['n']][deg]
        )
        xlim = max(xlim, iters[-1])

    plt.xlim(xmax=xlim)
    plt.yscale('linear')
    plt.legend(title="Degree (d)", fontsize='small', fancybox=True)
    if save:
        dest = root_folder_path + '{}_{}_{}n_{}_iter_vs_time.png'.format(
            test,
            dataset,
            n,
            distr
        )
        plt.savefig(dest, bbox_inches='tight')
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
        pred_ratios[graph] = 1 / math.sqrt(
            uniform_weighted_Pn_spectral_gap_from_adjacency_matrix(setup['graphs'][graph]))

        if 'clique' in graph:
            clique_slope = slope
            clique_spectral_gap = uniform_weighted_Pn_spectral_gap_from_adjacency_matrix(setup['graphs'][graph])

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
        plt.savefig(dest, bbox_inches='tight')
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

    for distr in ['par', 'exp', 'unif']:
        plt.plot(
            ly[distr].keys(),
            ly[distr].values(),
            label=distr,
            markersize=4,
            marker='o',
            color=distr_colors[distr]
        )

    ylim_padding = (ylim_up - ylim_dw) / 20
    # plt.ylim(ymax=ylim_up + ylim_padding)

    y_tick_values = []
    k = 0
    while 10 ** k <= ylim_up and k < 100:
        if ylim_dw <= 10 ** k:
            y_tick_values.append(10 ** k)
        k += 1

    if len(y_tick_values) > 0:
        if len(y_tick_values) == 1:
            tick = y_tick_values[0]
            y_tick_values.append(math.floor(ylim_up / tick) * tick)
        y_tick_values.sort()
        # '$\\mathdefault{10^{4}}$'
        plt.yticks(
            y_tick_values,
            ["$\\mathdefault{{{0}\\times10^{{{1}}}}}$".format(str(_n)[0], int(math.log10(_n))) for _n in y_tick_values],
            # rotation='vertical',
            # size='x-small'
        )

    l = [x for x in ly['exp'].keys() if x not in [16, 50]]
    plt.xticks(
        l,
        l,
        # rotation='vertical',
        # size='small'
    )

    plt.legend(title="", fontsize='small', fancybox=True)
    if save:
        plt.savefig(root_folder_path + '{}_{}n_{}_distr_iter_time_vs_degree.png'.format(
            dataset,
            n,
            error
        ), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    run()
