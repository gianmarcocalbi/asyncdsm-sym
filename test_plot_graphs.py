# libraries
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from src import graphs


def run1():
    N = 8
    K = 2

    # Build your graph
    G = nx.DiGraph()

    for i in range(1, N + 1):
        G.add_node(i)

    A = graphs.generate_n_cycle_d_regular_graph_by_degree(N, K)

    for i in range(N):
        for j in range(N):
            if A[i][j] == 1:
                G.add_edge(i + 1, j + 1)

    # Plot it
    pos = nx.circular_layout(G)

    plt.figure(figsize=(4.2, 4))
    nx.draw(G, pos,
        with_labels=True,
        node_size=800,
        node_color='white',
        edgecolors='black',
        arrows=True,
        arrowsize=20,
        dge_color='black',
        width=1,
        font_family='Times New Roman',
        font_size=14
    )

    plt.show()


def run2():
    N = 8
    K = 1

    # Build your graph
    G = nx.DiGraph()

    for i in range(1, N + 1):
        G.add_node(i)

    A = graphs.generate_n_cycle_d_regular_graph_by_degree(N,K)

    for i in range(N):
        for j in range(N):
            if A[i][j] == 1:
                G.add_edge(i + 1, j + 1)

    # Plot it
    pos = nx.circular_layout(G)

    plt.figure(figsize=(4.2, 4))
    nx.draw(G, pos,
        with_labels=True,
        node_size=800,
        node_color='white',
        edgecolors='black',
        arrows=True,
        arrowsize=20,
        dge_color='black',
        width=1,
        font_family='Times New Roman',
        font_size=14
    )

    plt.show()


if __name__ == '__main__':
    run2()

