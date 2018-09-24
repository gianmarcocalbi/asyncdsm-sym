import networkx as nx
import numpy as np
from src.utils import uniform_weighted_Pn_spectral_gap_from_adjacency_matrix
import os, argparse, glob, datetime, time, math


FILE_NUMBER_LIMIT = 100

def main(n=None, d=None):
    spectral_gap_func = uniform_weighted_Pn_spectral_gap_from_adjacency_matrix
    root_path = './graphs/exp_uniform_weighted_pool/{}n_{}d'.format(n, d)

    title = 'Seek random expander for N={} and d={}'.format(n, d)
    print(''.join(['#' for _ in range(len(title))]))
    print(title)
    print(''.join(['#' for _ in range(len(title))]))
    print('[00:00:00] - Start at ' + str(datetime.datetime.now().strftime('%H:%M:%S.%f')))

    exp_path_list = list(glob.iglob(root_path + '/exp_{}n_{}d*'.format(n, d)))
    exp_dict = {}
    for exp_path in exp_path_list:
        adj = np.loadtxt(exp_path)
        spectrum = spectral_gap_func(adj)
        exp_dict[spectrum] = exp_path

    spectrum_list = list(exp_dict.keys())
    spectrum_list.sort(reverse=True)
    if len(spectrum_list) > FILE_NUMBER_LIMIT:
        for sg in spectrum_list[100::]:
            if os.path.isfile(exp_dict[sg]):
                os.remove(exp_dict[sg])
                del exp_dict[sg]

    try :
        trial = 0
        start_time = time.time()
        while True:
            t0 = time.time()
            G = nx.random_regular_graph(d, n)
            if not nx.is_connected(G):
                trial += 1
                continue
            A = nx.to_numpy_array(G, dtype=int)
            np.fill_diagonal(A, 1)
            spectrum = spectral_gap_func(A)
            nx_time = time.time() - t0

            if spectrum in exp_dict:
                continue

            spectrum_list = list(exp_dict.keys())
            if len(spectrum_list) > 0:
                spectrum_list.sort(reverse=True)
                least_exp_spectrum = spectrum_list[-1]
            else:
                least_exp_spectrum = -math.inf

            if (len(spectrum_list) < FILE_NUMBER_LIMIT or spectrum > least_exp_spectrum) and spectrum not in exp_dict:
                if len(spectrum_list) >= FILE_NUMBER_LIMIT:
                    os.remove(exp_dict[least_exp_spectrum])
                    del exp_dict[least_exp_spectrum]

                path = root_path + '/exp_{}n_{}d_{}spectrum.txt.gz'.format(n, d, spectrum)
                i = 0
                while os.path.isfile(path):
                    path = root_path + '/exp_{}n_{}d_{}spectrum.{}.txt.gz'.format(n, d, spectrum, i)
                    i += 1
                np.savetxt(path, A)
                exp_dict[spectrum] = path
                print('[{}] - trial : {} -> found G with spectrum = {} in {}s'.format(
                    str(datetime.timedelta(seconds=int(time.time() - start_time))),
                    trial,
                    spectrum,
                    round(nx_time, 2)
                ))
            trial += 1

    except:
        print()
        msg = "Interrupt sent. Wait for final checking of matrices!"
        print(''.join(['#' for _ in range(len(msg))]))
        print(msg)
        print(''.join(['#' for _ in range(len(msg))]))
        print('Checking expanders for N={} and d={}'.format(n, d))

        exp_path_list = list(glob.iglob(root_path + '/exp_{}n_{}d*'.format(n, d)))
        exp_dict = {}
        final_check = True
        for exp_path in exp_path_list:
            removed = False
            try:
                adj = np.loadtxt(exp_path)
                spectrum = spectral_gap_func(adj)
                exp_dict[spectrum] = exp_path
            except:
                if os.path.isfile(exp_path):
                    os.remove(exp_path)
                    removed = True
                    final_check = False
            finally:
                removed = 'REMOVED' if removed else 'OK'
                print('Expander {} {}'.format(exp_path, removed))


        if final_check:
            print('All generated expanders were GOOD')
        else:
            print('Some expanders were not good and have been removed')



if __name__ == "__main__":
    # argparse setup
    parser = argparse.ArgumentParser(
        description='Seek random expander'
    )

    parser.add_argument(
        '-d',
        '--degree',
        help='Degree',
        required=True,
        action='store',
        dest='d'
    )

    parser.add_argument(
        '-n',
        '--node-amount',
        help='Nodes amount',
        required=True,
        action='store',
        dest='n'
    )

    args = parser.parse_args()

    main(int(args.n), int(args.d))
