import glob
import math
import warnings

import numpy as np

from src.utils import one_half_diagonal_Pn_spectral_gap_from_adjacency_matrix, \
    uniform_weighted_Pn_spectral_gap_from_adjacency_matrix


def generate_d_regular_graph_by_adjacency(adj_matrix_first_row):
    """
    Generate considering linear adjacency relationship, that is
    if there exists and edge (i,j) then there exists also an
    edge (i+i,j+i). Therefore given the 1st row of the adjacency
    matrix, then , for instance, the 2nd row of the adjacency
    matrix will be the 1st line shifted to the right by one.
    :param adj_matrix_first_row: first row of the adjacency matrix
    :return: whole adjacency NUMPY matrix
    """
    N = len(adj_matrix_first_row)
    adjacency_matrix = np.diag(np.ones(N))
    print(adjacency_matrix)
    for i in range(N):
        for j in range(N):
            adjacency_matrix[i][j] = adj_matrix_first_row[(j - i) % N]
    return adjacency_matrix


def enforce_symmetry_to_matrix(adjacency_matrix, symmetry_source='sup'):
    N = adjacency_matrix.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            if symmetry_source != 'inf':
                adjacency_matrix[j][i] = adjacency_matrix[i][j]
            else:
                adjacency_matrix[i][j] = adjacency_matrix[j][i]

    return adjacency_matrix


def is_regular(adjacency_matrix):
    d = np.sum(adjacency_matrix[0])
    for i in range(1, adjacency_matrix.shape[0]):
        if np.sum(adjacency_matrix[i]) != d:
            return False
    return True


def is_undirected(adjacency_matrix):
    N = adjacency_matrix.shape[0]
    for i in range(N):
        for j in range(i, N):
            if adjacency_matrix[j][i] != adjacency_matrix[i][j]:
                return False
    return True


def is_symmetric(adjacency_matrix):
    return is_undirected(adjacency_matrix)


def generate_graph_by_edges(N, edges, force_symmetry=False):
    """
    Generate a graph given the list of its edges, where each edge (element of
    the list, must be a string "{u} -> {v}" such that u and v follow one of the
    formats below:

    a) "i" : will be applied to each vertex of the graph;
    For example: "i -> 0" means that all vertices are linked to vertex 0.

    b) an int k, 0 <= k <= n : indicates a precise vertex;

    c) a math expression with i;
    For example "i -> math.sqrt(i) + N/2"

    d) "N" : stands for the amount of vertices in the graph.

    Note that all values will be applied the modulo N (% N) operation to, in
    order to avoid exceeding the vertices max index.


    Parameters
    ----------
    N : int
        Amount of vertices in the graph.

    edges : list or tuple of str
        List of edges of the graph.

    force_symmetry : bool
        Whether force the adjacency matrix to be symmetric (undirected graph)

    Returns
    -------
    adjacency_matrix : numpy.ndarray
    """

    adjacency_matrix = np.diag(np.ones(N))
    for i in range(N):
        for e in edges:
            u, v = e.replace(" ", "").split("->")
            adjacency_matrix[eval(u) % N][eval(v) % N] = 1

    if force_symmetry:
        adjacency_matrix = enforce_symmetry_to_matrix(adjacency_matrix)
        if not is_regular(adjacency_matrix):
            warnings.warn("generate_graph_by_edges has generated a non-regular graph")

    return adjacency_matrix


def generate_complete_graph(N):
    """
    Generate complete (clique) graph adjacency matrix.
    :param N: amount of vertices
    :return: adjacency numpy matrix
    """
    return np.ones((N, N))


def generate_uniform_edges_d_regular_graph_by_degree(N, K):
    """
    Generate a regular graph with N vertices and degree K with homogeneous expansion.

    Parameters
    ----------
    N : int
        Amount of nodes in the graph.
    K : int
        Graph's degree.

    Returns
    -------
    numpy multidimensional array of bit
    """
    if N <= 1 or K <= 0:
        edges = []
    else:
        edges = ["i->i+1"]
        for i in range(K - 1):
            edges.append("i->i+{}".format(int((i + 1) * N / K)))

    return generate_graph_by_edges(N, edges)


def generate_n_cycle_d_regular_graph_by_degree(N, K):
    """
    Generate a d regular n cycle with N vertices and degree K.

    Parameters
    ----------
    N : int
        Amount of nodes in the graph.
    K : int
        Graph's degree.

    Returns
    -------
    numpy multidimensional array of bit
    """
    edges = []
    if N > 1 and K > 0:
        for i in range(K):
            edges.append("i->i+{}".format(i + 1))

    return generate_graph_by_edges(N, edges)


def generate_undirected_n_cycle_d_regular_graph_by_degree(N, K):
    """
    Generate an undirected d regular n cycle with N vertices and degree K.

    Parameters
    ----------
    N : int
        Amount of nodes in the graph.
    K : int
        Graph's degree.

    Returns
    -------
    numpy multidimensional array of bit
    """
    edges = []
    if N > 1 and K > 1:
        edges.append("i->i+1")
        edges.append("i+1->i")
        if K % 2 == 0 and K > 2:
            for i in range(1, int(K / 2)):
                edges.append("i->i+{}".format(i + 1))
                edges.append("i+{}->i".format(i + 1))

        elif K % 2 == 1:
            if N % (K + 1) == 0:
                if K == 3:
                    for h in range(N // (2 * (K - 1))):
                        i = h * (K - 1) * 2
                        for j in range(K - 1):
                            u = i + j
                            v = i + j + 2
                            edges.append('{}->{}'.format(u, v))
                            edges.append('{}->{}'.format(v, u))
                else:
                    raise NotImplementedError("Undirected cycle with even degree (d={}) !=3 not implemented yet".format(
                        K
                    ))
            else:
                raise Exception(
                    "Cannot generate undirected cycle with N={} and d={}. N should have been divisible by d+1".format(
                        N, K
                    )
                )

    return generate_graph_by_edges(N, edges)


def generate_expander_graph(N, degree, matrix_type='uniform-weighted'):
    max_spectrum = 0
    max_exp = None
    graphs_root = ''
    spectral_gap_function = None

    if matrix_type == 'one-half-diagonal':
        graphs_root = './graphs/exp_one_half_diagonal'
        spectral_gap_function = one_half_diagonal_Pn_spectral_gap_from_adjacency_matrix
    elif matrix_type == 'uniform-weighted':
        graphs_root = './graphs/exp_uniform_weighted'
        spectral_gap_function = uniform_weighted_Pn_spectral_gap_from_adjacency_matrix
    elif matrix_type == 'uniform-weighted-alt':
        graphs_root = './graphs/exp_uniform_weighted_alt'
        spectral_gap_function = uniform_weighted_Pn_spectral_gap_from_adjacency_matrix

    exp_path_list = list(glob.iglob('{}/exp_{}n_{}d*'.format(graphs_root, N, degree)))
    for exp_path in exp_path_list:
        adj = np.loadtxt(exp_path)
        spectrum = spectral_gap_function(adj)
        if spectrum > max_spectrum:
            max_spectrum = spectrum
            max_exp = adj

    return max_exp


class Graph:
    def __init__(self, N, A, name):
        self.N = N
        self.name = name
        self.A = A


def G(N, gtype, d=0):
    A = None
    if gtype == 'diagonal':
        A = np.diag(np.ones(N))
    elif gtype == 'expander':
        A = generate_expander_graph(N, d)
    elif gtype == 'alt_expander':
        A = generate_expander_graph(N, d, matrix_type='uniform-weighted-alt')
    elif gtype == 'cycle':
        A = generate_n_cycle_d_regular_graph_by_degree(N, d)
    elif gtype == 'undir_cycle':
        A = generate_undirected_n_cycle_d_regular_graph_by_degree(N, d)
    elif gtype == 'uniform_edges':
        A = generate_uniform_edges_d_regular_graph_by_degree(N, d)
    elif gtype == 'root_expander':
        A = generate_graph_by_edges(N, ["i->i+1", "i->i+{}".format(int(math.sqrt(N)))])
        d = 2
    elif 'clique' in gtype:
        A = generate_complete_graph(N)
        d = N - 1
    else:
        raise Exception("Graph gtype {} doesn't exist".format(gtype))

    return Graph(N, A, "{}-{}".format(d, gtype))


def generate_n_nodes_graphs_list(n, graphs_list):
    graphs = {}

    for gstr in graphs_list:
        if gstr[0:3] == 'n-1':
            deg = 'n-1'
            gtype = gstr.split("-", 2)[2]
        else:
            if len(gstr.split("_")[0]) < len(gstr.split("-")[0]):
                sep = '_'
            else:
                sep = '-'
            deg, gtype = gstr.split(sep, 1)

        if 'n' in deg:
            deg = eval(deg)
        g = G(n, gtype, int(deg))
        graphs[g.name] = g.A

    return graphs


def generate_n_nodes_graph(n, graph_name):
    if graph_name[0:3] == 'n-1':
        deg = 'n-1'
        gtype = graph_name.split("-", 2)[2]
    else:
        if len(graph_name.split("_")[0]) < len(graph_name.split("-")[0]):
            sep = '_'
        else:
            sep = '-'
        deg, gtype = graph_name.split(sep, 1)

    if 'n' in deg:
        deg = eval(deg)
    return G(n, gtype, int(deg))
