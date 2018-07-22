import networkx as nx
import numpy as np
from src.utils import Pn_spectral_gap_from_adjacency_matrix, mtm_spectral_gap_from_adjacency_matrix
import os, argparse, glob, datetime, time


def main(matrix_type='mtm'):
    if matrix_type == 'mtm':
        spectral_gap_func = mtm_spectral_gap_from_adjacency_matrix
        root_path = './graphs/exp_mtm'
    else:
        spectral_gap_func = Pn_spectral_gap_from_adjacency_matrix
        root_path = './graphs/exp_half_weighted'

    exp_path_list = glob.iglob(root_path + '/exp_*')

    for exp_path in exp_path_list:
        adj = np.loadtxt(exp_path)
        n = len(adj)
        d = int(sum(adj[0]) - 1)
        spectrum = spectral_gap_func(adj)
        os.remove(exp_path)
        path = root_path + '/exp_{}n_{}d_{}spectrum.txt.gz'.format(n, d, spectrum)
        i = 0
        while os.path.isfile(path):
            path = root_path + '/exp_{}n_{}d_{}spectrum.{}.txt.gz'.format(n, d, spectrum, i)
            i += 1
        np.savetxt(path, adj)
        print('Updated file {}'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Update expanders name'
    )

    parser.add_argument(
        '-m', '--matrix',
        action='store',
        default='mtm',
        required=True,
        help='Specify matrix type (mtm or else)',
        dest='matrix_type'
    )

    args = parser.parse_args()

    main(args.matrix_type)
