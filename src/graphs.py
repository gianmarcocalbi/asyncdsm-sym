import numpy as np
import warnings, math


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
    if N > 1 or K > 0:
        for i in range(K):
            edges.append("i->i+{}".format(i + 1))

    return generate_graph_by_edges(N, edges)


def generate_expander_graph(N, degree):
    # todo
    for i in range(N):
        pass

class Graph:
    def __init__(self, N, name):
        self.N = N
        self.name = name

def G(N, type, d=0):
    A = None
    if type == 'diagonal':
        A = np.diag(np.ones(N))
    elif type == 'cycle':
        A = generate_n_cycle_d_regular_graph_by_degree(N, d)
    elif type == 'uniform_edges':
        A = generate_uniform_edges_d_regular_graph_by_degree(N, d)
    else:
        raise Exception("Graph type {} doesn't exist".format(type))

    return Graph(N, "{}-{}".format())

# degree = 2
CYCLE_B = lambda n: generate_graph_by_edges(n, ["i->i+1", "i->i-1"])
DIAM_EXP = lambda n: generate_graph_by_edges(
    n,
    ["i->i+1", "i->i+{}".format(int(n / 2))])
ROOT_EXP = lambda n: generate_graph_by_edges(
    n,
    ["i->i+1", "i->i+{}".format(int(math.sqrt(n)))])

# degree = 3
DIAM_EXP_B = lambda n: generate_graph_by_edges(
    n,
    ["i->i+1", "i->i-1", "i->i+{}".format(int(n / 2))])

# degree = n
STAR = lambda n: generate_graph_by_edges(n, ["i->0", "0->i"])
CLIQUE = lambda n: generate_complete_graph(n)

# regular
UNIFORM_EDGES_REGULAR = lambda n, k: generate_uniform_edges_d_regular_graph_by_degree(n, k)
N_CYCLE_REGULAR = lambda n, k: generate_n_cycle_d_regular_graph_by_degree(n, k)


GRAPHS = {
    "0_diagonal": DIAGONAL(setup['n']),
    "1_cycle": CYCLE(setup['n']),  # degree = 1
    "2_cycle-bi": CYCLE_B(setup['n']), # degree = 2
    "2_diam-expander": DIAM_EXP(setup['n']),  # degree = 2
    "2_root-expander": ROOT_EXP(setup['n']),  # degree = 2
    "3_regular": UNIFORM_EDGES_REGULAR(setup['n'], 3),  # degree = 3
    "4_regular": UNIFORM_EDGES_REGULAR(setup['n'], 4),  # degree = 4
    "5_regular": UNIFORM_EDGES_REGULAR(setup['n'], 5),  # degree = 5
    "6_regular": UNIFORM_EDGES_REGULAR(setup['n'], 6),  # degree = 6
    "7_regular": UNIFORM_EDGES_REGULAR(setup['n'], 7),  # degree = 7
    "8_regular": UNIFORM_EDGES_REGULAR(setup['n'], 8),  # degree = 8
    "9_regular": UNIFORM_EDGES_REGULAR(setup['n'], 9),  # degree = 9
    "10_regular": UNIFORM_EDGES_REGULAR(setup['n'], 10),  # degree = 10
    "20_regular": UNIFORM_EDGES_REGULAR(setup['n'], 20),  # degree = 20
    "50_regular": UNIFORM_EDGES_REGULAR(setup['n'], 50),  # degree = 50
    "n-1_clique": CLIQUE(setup['n']),  # degree = n
    "n-1_star": STAR(setup['n']),

}