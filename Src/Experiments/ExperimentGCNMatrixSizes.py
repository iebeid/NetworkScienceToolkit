import numpy as np
import copy
import networkx as nx
import scipy as sp
print(sp.__version__)
print(nx.__version__)
G1 = nx.gnm_random_graph(34, 78, 1)
G1A = np.array(nx.adjacency_matrix(G1).todense())
print(G1)
print(G1A)
G2 = copy.deepcopy(G1)
G2A = np.array(nx.adjacency_matrix(G2).todense())
print(G2)
print(G2A)
print(np.array_equal(G1A, G2A))
GD1 = nx.DiGraph()
GD1.add_edges_from(
    [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
     ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])
GD1A = np.array(nx.adjacency_matrix(GD1).todense())
print(GD1)
print(GD1A)