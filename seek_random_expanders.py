import networkx as nx
import numpy as np
from src.utils import Pn_spectral_gap_from_adjacency_matrix, mtm_spectral_gap_from_adjacency_matrix
import os, argparse, glob, datetime, time


def main(matrix_type='mtm', n=None, d=None):
    if matrix_type == 'mtm':
        spectral_gap_func = mtm_spectral_gap_from_adjacency_matrix
        root_path = './graphs/exp_mtm'
    else:
        spectral_gap_func = Pn_spectral_gap_from_adjacency_matrix
        root_path = './graphs/exp_half_weighted'

    title = 'Seek {} random expander for N={} and d={}'.format(matrix_type, n, d)
    print(''.join(['#' for _ in range(len(title))]))
    print(title)
    print(''.join(['#' for _ in range(len(title))]))
    print('[00:00:00] - Start at ' + str(datetime.datetime.now().strftime('%H:%M:%S.%f')))

    exp = None
    max_spectrum = 0
    exp_path_list = list(glob.iglob(root_path + '/exp_{}n_{}d*'.format(n, d)))
    found_new_exp = False
    for exp_path in exp_path_list:
        adj = np.loadtxt(exp_path)
        spectrum = spectral_gap_func(adj)
        if spectrum > max_spectrum:
            max_spectrum = spectrum
            exp = adj

    try:
        exp = None
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
            if spectrum > max_spectrum:
                found_new_exp = True
                max_spectrum = spectrum
                exp = A
                print('[{}] - trial : {} -> found G with spectrum = {} in {}s'.format(
                    str(datetime.timedelta(seconds=int(time.time() - start_time))),
                    trial,
                    max_spectrum,
                    round(nx_time, 2)
                ))
            trial += 1
    except:
        if found_new_exp:
            path = root_path + '/exp_{}n_{}d_{}spectrum.txt.gz'.format(n, d, max_spectrum)
            i = 0
            while os.path.isfile(path):
                path = root_path + '/exp_{}n_{}d_{}spectrum.{}.txt.gz'.format(n, d, max_spectrum, i)
                i += 1
            np.savetxt(path, exp)


if __name__ == "__main__":
    # argparse setup
    parser = argparse.ArgumentParser(
        description='Seek random expander'
    )

    parser.add_argument(
        '-m', '--matrix',
        action='store',
        default='mtm',
        required=True,
        help='Specify matrix type (mtm or else)',
        dest='matrix_type'
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

    main(args.matrix_type, int(args.n), int(args.d))
