import networkx as nx
import numpy as np
from src.functions import Pn_spectral_gap_from_adjacency_matrix
import os, argparse, glob, datetime, time


def main(n=None, d=None):
    print('Seek random expander for N={} and d={}'.format(n, d))
    exp = None
    max_spectrum = 0
    exp_path_list = list(glob.iglob('./graphs/exp_{}n_{}d*'.format(n, d)))
    found_new_exp = False
    for exp_path in exp_path_list:
        adj = np.loadtxt(exp_path)
        spectrum = Pn_spectral_gap_from_adjacency_matrix(adj)
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
            spectrum = Pn_spectral_gap_from_adjacency_matrix(A)
            nx_time = time.time() - t0
            if spectrum > max_spectrum:
                found_new_exp = True
                max_spectrum = spectrum
                exp = A
                print('[{}] - trial : {} -> found G with spectrum = {} in {}s'.format(
                    str(datetime.timedelta(seconds=int(time.time() - start_time))),
                    trial,
                    max_spectrum,
                    round(nx_time,2)
                ))
            trial += 1
    except:
        if found_new_exp:
            path = './graphs/exp_{}n_{}d_{}spectrum.txt.gz'.format(n, d, max_spectrum)
            i = 0
            while os.path.isfile(path):
                path = './graphs/exp_{}n_{}d_{}spectrum.{}.txt.gz'.format(n, d, max_spectrum,i)
                i += 1
            np.savetxt(path, exp)


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
