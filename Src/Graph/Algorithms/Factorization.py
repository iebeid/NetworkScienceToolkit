import itertools
import random
import multiprocessing
from collections import defaultdict
from collections import deque
import networkx as nx
from community import community_louvain as louvain
import numpy as np
import scipy.linalg
import scipy.stats
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import identity
from scipy.sparse.csgraph import connected_components
# from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

class Factorization():

    # Constructor
    def __init__(self, graph):
        self.graph = graph

    # SPEK
    def spectral_embedding_method(self, id2refid):
        eigenvals, eigenvecs = scipy.linalg.eig(self.graph.laplacian_weighted.todense())
        print(eigenvals.shape)
        print(eigenvecs.shape)
        eigenvecs = np.array(eigenvecs.real)
        eigenvals = np.array(eigenvals.real)
        eigenvals_sorted = np.sort(eigenvals)
        eigenvecs_transposed = np.transpose(eigenvecs)
        fiedler_vectors = []
        for ev in range(0, len(eigenvals_sorted)):
            corresponding_eigenval_location = eigenvals.tolist().index(eigenvals_sorted[ev])
            corresponding_eigenval = eigenvals[corresponding_eigenval_location]
            scaled_eigenvecs = eigenvecs_transposed[corresponding_eigenval_location] * corresponding_eigenval
            fiedler_vectors.append(scaled_eigenvecs)
        X = np.array(fiedler_vectors).T
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8, affinity="euclidean",
                                             compute_full_tree=True, compute_distances=True,
                                             connectivity=self.graph.adjacency.toarray(), linkage="single").fit(X)
        print(clustering.labels_)
        record_labels = {}
        for (row, label) in enumerate(clustering.labels_):
            print("row %s has label %d" % (id2refid[row], label))
            record_labels[id2refid[row]] = label
        node_label_profile = defaultdict(list)
        for key, val in sorted(record_labels.items()):
            node_label_profile[val].append(key)
        node_label_profile = dict(node_label_profile)
        return node_label_profile


    # spectral embedding using eigen value decomposition
    def spectral_schema_emebdding(self, embedding_dimension):
        eigenvals, eigenvecs = eigsh(self.graph.L_schema, k=embedding_dimension)
        eigenvals_sorted = np.sort(np.array(eigenvals.real))
        eigenvecs_transposed = np.transpose(np.array(eigenvecs.real))
        fiedler_vectors = []
        for ev in range(0, len(eigenvals_sorted)):
            corresponding_eigenval_location = eigenvals.tolist().index(eigenvals_sorted[ev])
            scaled_eigenvecs = eigenvecs_transposed[corresponding_eigenval_location] * eigenvals[
                corresponding_eigenval_location]
            fiedler_vectors.append(scaled_eigenvecs)
        X = list(np.array(fiedler_vectors).T)
        Y = list(self.graph.node_index.keys())
        return X, Y