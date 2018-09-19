# libraries
import networkx as nx
import matplotlib.pyplot as plt
from src import graphs

N = 12
K = 4

# Build your graph
G = nx.Graph()

for i in range(N):
    G.add_node(i)

A = graphs.generate_undirected_n_cycle_d_regular_graph_by_degree(N, K)

for i in range(N):
    for j in range(N):
        if A[i][j] == 1:
            G.add_edge(i,j)

# Plot it
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()

