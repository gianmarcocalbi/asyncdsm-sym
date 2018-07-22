import glob
import matplotlib.pyplot as plt
from src import statistics
from src.utils import *


def main(compute=True):
    # SETUP BEGIN

    spectral_gap_function = mtm_spectral_gap_from_adjacency_matrix

    exp_path_list_pattern = os.path.normpath("./graphs/exp_*")
    exp_path_list = list(glob.iglob(exp_path_list_pattern))

    # spectrums -> n -> d -> spectrum
    spectrums = {}

    for exp_path in exp_path_list:
        adj = np.loadtxt(exp_path)
        n = int(len(adj))
        d = int(sum(adj[0]))-1
        if compute:
            spectrum = spectral_gap_function(adj)
        else:
            spectrum = float(exp_path.split('_')[-1].split('spectrum')[0])

        if n not in spectrums:
            spectrums[n] = {}

        if d not in spectrums[n]:
            spectrums[n][d] = -1

        spectrums[n][d] = max(spectrums[n][d], spectrum)

    for n in sorted(list(spectrums.keys())):
        degrees = list(spectrums[n].keys())
        degrees.sort()
        for d in degrees:
            print('N = {}, d = {}, spectral_gap = {}'.format(n,d,spectrums[n][d]))
        print('')

if __name__ == "__main__":
    main()