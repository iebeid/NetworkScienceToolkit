
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

class CommunityDetection():

    # Constructor
    def __init__(self, graph):
        self.graph = graph

    # Louvain method of commnuity detection using modularity optimization based on:
    # Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large
    # networks. Journal of statistical mechanics: theory and experiment, 2008(10), P10008.
    def networkx_louvain_community_detection(self):
        partitions = louvain.best_partition(self.graph.to_networkx, partition=self.graph.node_label)
        partitions_profiles = defaultdict(list)
        for n, com in sorted(partitions.items()):
            partitions_profiles[com].append(n)
        partitions_profiles = dict(partitions_profiles)
        components = []
        for k, v in partitions_profiles.items():
            components.append(v)
        return components

    def akef_modularity_maximizer(self):
        for i in self.graph.nodes.keys():
            # what if i removed i from its community
            i_com = self.graph.tasks.get_node_label(i)
            i_modularity = self.graph.tasks.compute_delta_modularity_node_remove(i_com, i)
            all_possible_moves = []
            all_possible_coms = []
            all_weights = []
            neighbors = self.graph.tasks.get_neighbors(i)
            if len(neighbors) > 0:
                for j in neighbors:
                    if i != j:
                        w = self.graph.tasks.get_edge_weight(i, j)
                        j_com = self.graph.tasks.get_node_label(j)
                        j_modularity = self.graph.tasks.compute_delta_modularity_node_add(j_com, i)
                        delta_modularity = float(i_modularity + j_modularity)
                        all_possible_moves.append(delta_modularity)
                        all_possible_coms.append(j_com)
                        all_weights.append(w)
                max_node_index = np.argmax(np.array(all_possible_moves))
                max_delta_mod_value = all_possible_moves[max_node_index]
                max_com = all_possible_coms[max_node_index]
                max_weight = all_weights[max_node_index]
                if (max_delta_mod_value > 0.0) and max_weight == 1:
                    # if (max_delta_mod_value > 0.0):
                    self.graph.tasks.set_node_label(max_com, i)
        self.graph.update_node_labels(self.graph.node_label)
        components = self.graph.node_labels_to_components()
        return components

    def akef_greedy_modularity_optimizer(self):
        node_label_changes = self.graph.node_label
        final_label_changes = self.graph.node_label
        final_label_profile = self.graph.node_label_profile
        initial_modularity = self.graph.tasks.compute_modularity()
        previous_modularity = initial_modularity
        for i in self.graph.nodes.keys():
            i_com = self.graph.tasks.get_node_label(i)
            # remove i from its community and assign it to a single community
            node_label_changes[i] = int(max(list(final_label_profile.keys())) + 1)
            self.graph.update_node_labels(node_label_changes)
            current_modularity = self.graph.tasks.compute_modularity()
            delta_modularity_1 = current_modularity - previous_modularity
            previous_modularity = current_modularity
            # add i to community of j
            neighborhood = self.graph.tasks.get_neighbors(i)
            candidate_memberships = {}
            for j in neighborhood:
                j_com = self.graph.tasks.get_node_label(j)
                node_label_changes[i] = j_com
                self.graph.update_node_labels(node_label_changes)
                current_modularity = self.graph.tasks.compute_modularity()
                delta_modularity = delta_modularity_1 + (current_modularity - previous_modularity)
                candidate_memberships[j_com] = round(delta_modularity, 3)
            max_com = max(candidate_memberships, key=candidate_memberships.get)
            largest_delta_modularity_value = candidate_memberships[max_com]
            if largest_delta_modularity_value > 0:
                final_label_changes[i] = max_com
            if largest_delta_modularity_value < 0:
                final_label_changes[i] = int(max(list(final_label_profile.keys())) + 1)
            if largest_delta_modularity_value == 0:
                final_label_changes[i] = i_com
            self.graph.update_node_labels(final_label_changes)
            previous_modularity = self.graph.tasks.compute_modularity()
        final_modularity = self.graph.tasks.compute_modularity()
        print(self.graph.node_label_profile)
        print("initial_modularity:" + str(initial_modularity))
        print("final_modularity:" + str(final_modularity))

    # Barber, M. J. (2007). Modularity and community detection in bipartite networks. Physical Review E, 76(6), 066102.
    def akef_modified_brim_bipartite_cluster_detection(self):
        edges = []
        words = []
        documents = []
        for n in self.graph.nodes.keys():
            extracted_words = self.graph.node_attribute[n][0]
            documents.append(n)
            for t in extracted_words:
                edges.append((t, n))
                words.append(t)
        words = list(np.unique(np.array(words)))
        # ---------
        word_membership = []
        document_membership = []
        all_nodes = documents + words
        for mem, n in enumerate(all_nodes):
            if n in words:
                word_membership.append(mem)
            if n in documents:
                document_membership.append(mem)
        # Dimensions of the matrix
        p = len(documents)
        q = len(words)
        # c = doc_com
        c = len(all_nodes)
        # Index dictionaries for the matrix. Note that this set of indices is different of that in the condor object (that one is for the igraph network.)
        rg = {words[i]: i for i in range(0, q)}
        gn = {documents[i]: i for i in range(0, p)}
        # Computes weighted biadjacency matrix.
        A = np.matrix(np.zeros((p, q)))
        for edge in edges:
            A[gn[edge[1]], rg[edge[0]]] = 1.0
        # Computes node degrees for the nodesets.
        ki = A.sum(1)
        dj = A.sum(0)
        combined_degrees = ki @ dj
        # Computes sum of edges and bimodularity matrix.
        m = float(sum(ki))
        B = A - (combined_degrees / m)
        # Computation of initial modularity matrix for tar and reg nodes from the membership dataframe.
        T_ed = zip([gn[j] for j in [i for i in documents]], document_membership)
        T0 = np.zeros((p, c))
        for edge in T_ed:
            T0[edge] = 1
        R_ed = zip([rg[j] for j in [i for i in words]], word_membership)
        R0 = np.zeros((q, c))
        for edge in R_ed:
            R0[edge] = 1
        deltaQmin = min(1 / m, 1e-5)
        Qnow = 0
        deltaQ = 1
        p, q = B.shape
        while (deltaQ > deltaQmin):
            # Right sweep
            Tp = T0.transpose().dot(B)
            R = np.zeros((q, c))
            am = np.array(np.argmax(Tp.transpose(), axis=1))
            for i in range(0, len(am)):
                R[i, am[i][0]] = 1
            # Left sweep
            Rp = B.dot(R)
            T = np.zeros((p, c))
            am = np.array(np.argmax(Rp, axis=1))
            for i in range(0, len(am)):
                T[i, am[i][0]] = 1
            T0 = T
            Qthen = Qnow
            RtBT = T.transpose().dot(B.dot(R))
            Qcoms = (1 / m) * (np.diagonal(RtBT))
            Qnow = sum(Qcoms)
            deltaQ = Qnow - Qthen
        tar_membership = list(zip(list(gn), [T[i, :].argmax() for i in range(0, len(gn))]))
        document_memberships = {}
        for tup in tar_membership:
            document_memberships[str(tup[0])] = int(tup[1])
        grouped_document_memberships = defaultdict(list)
        for index, row in sorted(document_memberships.items()):
            grouped_document_memberships[row].append(index)
        components = []
        for k, v in grouped_document_memberships.items():
            components.append(v)
        self.graph.update_node_labels_with_components(components)
        return components

    def akef_modified_brim_bipartite_cluster_detection_2(self):
        edges = []
        words = []
        documents = []
        document_membership = []
        for n in self.graph.nodes.keys():
            extracted_words = self.graph.node_attribute[n][0]
            document_membership.append(self.graph.node_label[n])
            documents.append(n)
            for t in extracted_words:
                edges.append((t, n))
                words.append(t)
        words = list(np.unique(np.array(words)))
        all_nodes = documents + words

        word_membership = []
        max_doc_com = np.max(np.array(list(self.graph.node_label.values())))
        for mem, n in enumerate(words):
            word_membership.append(mem + max_doc_com + 1)

        # Dimensions of the matrix
        p = len(documents)
        q = len(words)
        # c = doc_com
        c = len(all_nodes)
        # Index dictionaries for the matrix. Note that this set of indices is different of that in the condor object (that one is for the igraph network.)
        rg = {words[i]: i for i in range(0, q)}
        gn = {documents[i]: i for i in range(0, p)}
        # Computes weighted biadjacency matrix.
        A = np.matrix(np.zeros((p, q)))
        for edge in edges:
            A[gn[edge[1]], rg[edge[0]]] = 1.0
        # Computes node degrees for the nodesets.
        ki = A.sum(1)
        dj = A.sum(0)
        combined_degrees = ki @ dj
        # Computes sum of edges and bimodularity matrix.
        m = float(sum(ki))
        B = A - (combined_degrees / m)
        # Computation of initial modularity matrix for tar and reg nodes from the membership dataframe.
        T_ed = zip([gn[j] for j in [i for i in documents]], document_membership)
        T0 = np.zeros((p, c))
        for edge in T_ed:
            T0[edge] = 1
        R_ed = zip([rg[j] for j in [i for i in words]], word_membership)
        R0 = np.zeros((q, c))
        for edge in R_ed:
            R0[edge] = 1
        deltaQmin = min(1 / m, 1e-5)
        Qnow = 0
        deltaQ = 1
        p, q = B.shape
        while (deltaQ > deltaQmin):
            # Right sweep
            Tp = T0.transpose().dot(B)
            R = np.zeros((q, c))
            am = np.array(np.argmax(Tp.transpose(), axis=1))
            for i in range(0, len(am)):
                R[i, am[i][0]] = 1
            # Left sweep
            Rp = B.dot(R)
            T = np.zeros((p, c))
            am = np.array(np.argmax(Rp, axis=1))
            for i in range(0, len(am)):
                T[i, am[i][0]] = 1
            T0 = T
            Qthen = Qnow
            RtBT = T.transpose().dot(B.dot(R))
            Qcoms = (1 / m) * (np.diagonal(RtBT))
            Qnow = sum(Qcoms)
            deltaQ = Qnow - Qthen
        tar_membership = list(zip(list(gn), [T[i, :].argmax() for i in range(0, len(gn))]))
        document_memberships = {}
        for tup in tar_membership:
            document_memberships[str(tup[0])] = int(tup[1])
        grouped_document_memberships = defaultdict(list)
        for index, row in sorted(document_memberships.items()):
            grouped_document_memberships[row].append(index)
        components = []
        for k, v in grouped_document_memberships.items():
            components.append(v)
        self.graph.update_node_labels_with_components(components)
        return components