import random
from collections import defaultdict
import networkx as nx
import numpy as np
import tensorflow as tf
from operator import itemgetter
import math
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import identity
from scipy.sparse.linalg import eigsh, eigs
from Src.Graph.Algorithms.Algorithms import Algorithms

from numpy.random import seed

seed(1)
tf.random.set_seed(2)


class Graph:
    # Constructor
    def __init__(self, edges, nodes=None, label_schema=None, undirected=True, link_single_nodes=False):
        # random.seed(42)
        if len(edges) == 0:
            print("Empty edge list")
        if undirected:
            # EDGES
            self.label_schema = label_schema
            self.original_edges = edges
            self.edges = []
            self.nx_edges = []
            for k, v in edges.items():
                self.edges.append((v["source"], v["target"], v["weight"], v["type"]))
                self.nx_edges.append((v["source"], v["target"], v["weight"]))
            # NODES
            self.single_nodes = []
            unique_nodes = []
            for e in self.edges:
                unique_nodes.append(e[0])
                unique_nodes.append(e[1])
            self.unique_nodes = list(set(unique_nodes))
            if nodes is None:
                i = 0
                l = 0
                self.nodes = {}
                for n in self.unique_nodes:
                    self.nodes[str(n)] = {"index": int(i), "type": "type", "attribute": list([0]), "features": None,
                                          "label": int(l)}
                    i = i + 1
                    l = l + 1
            else:
                self.nodes = dict(sorted(nodes.items()))
                passed_nodes = self.nodes.keys()
                self.single_nodes = list(set(passed_nodes) - set(self.unique_nodes))
                if (len(self.unique_nodes) < len(passed_nodes)) and link_single_nodes:
                    single_nodes = list(set.difference(set(passed_nodes), set(self.unique_nodes)))
                    for n in single_nodes:
                        self.edges.append((str(n), str(n), 1.0))
                if len(self.unique_nodes) > len(passed_nodes):
                    single_nodes = list(set.difference(set(passed_nodes), set(self.unique_nodes)))
                    for n in single_nodes:
                        for e in self.edges[:]:
                            s_n = e[0]
                            t_n = e[1]
                            w = e[2]
                            if n == s_n or n == t_n:
                                self.edges.remove((s_n, t_n, w))
            self.N = len(self.nodes.keys())
            self.node_attributes = {}
            self.node_features = {}
            self.node_label = {}
            self.node_index = {}
            self.node_type = {}
            self.node_inverted_index = {}
            self.node_label_profile = defaultdict(list)
            self.node_type_profile = defaultdict(list)
            self.node_label_count = {}
            self.node_type_count = {}
            self.node_real_id = {}
            node_index = 0
            for i, v in self.nodes.items():
                self.node_attributes[i] = v["attributes"]
                self.node_features[i] = v["features"]
                self.node_label[i] = v["label"]
                self.node_index[i] = node_index
                self.node_inverted_index[node_index] = i
                self.node_type[i] = v["type"]
                if nodes is not None:
                    self.node_real_id[i] = i
                node_index = node_index + 1
            for key, val in sorted(self.node_type.items()):
                self.node_type_profile[val].append(key)
            self.node_type_profile = dict(self.node_type_profile)
            for t, ns in self.node_type_profile.items():
                self.node_type_count[t] = len(ns)
            for key, val in sorted(self.node_label.items()):
                self.node_label_profile[val].append(key)
            self.node_label_profile = dict(self.node_label_profile)
            for t, ns in self.node_label_profile.items():
                self.node_label_count[t] = len(ns)
            # IDENTITY MATRIX
            self.identity = identity(self.N).tocsc()
            self.identity_tensor_flow = tf.constant(
                tf.convert_to_tensor(identity(self.N).todense(), dtype=tf.float32))
            # ADJACENCY MATRIX
            row = []
            col = []
            data_ones = []
            data_weights = []
            for e in self.edges:
                row.append(self.node_index[e[0]])
                col.append(self.node_index[e[1]])
                data_weights.append(e[2])
                data_ones.append(1)
            self.adjacency = scipy.sparse.coo_matrix((data_ones, (row, col)), shape=(self.N, self.N))
            self.adjacency_weighted = scipy.sparse.coo_matrix((data_weights, (row, col)), shape=(self.N, self.N))
            self.adjacency = self.adjacency.tocsc()
            self.adjacency_weighted = self.adjacency_weighted.tocsc()
            self.adjacency_identity = self.adjacency + self.identity

            self.schema = {}
            for edge_id, values in self.original_edges.items():
                source_node_type = str(self.nodes[values["source"]]["type"])
                relation_node_type = str(values["type"])
                target_node_type = str(self.nodes[values["target"]]["type"])
                self.schema[source_node_type + relation_node_type + target_node_type] = {"source": source_node_type,
                                                                                         "target": target_node_type,
                                                                                         "type": relation_node_type,
                                                                                         "weight": float(1)}
            self.relation_adjacency_matrices = {}
            self.relations_adjacency_node_counts = {}
            for k, v in self.schema.items():
                schema_source = v["source"]
                schema_target = v["target"]
                schema_type = v["type"]
                relation_row = []
                relation_col = []
                relation_data_ones = []
                nodes_r = []
                for e in self.edges:
                    source_node = e[0]
                    target_node = e[1]
                    nodes_r.append(source_node)
                    nodes_r.append(target_node)
                    relation_type = e[3]
                    source_node_type = self.nodes[source_node]["type"]
                    target_node_type = self.nodes[target_node]["type"]
                    if (source_node_type == schema_source) and (target_node_type == schema_target) and (
                            relation_type == schema_type):
                        relation_row.append(self.node_index[source_node])
                        relation_col.append(self.node_index[target_node])
                        relation_data_ones.append(1)
                self.relation_adjacency = scipy.sparse.coo_matrix(
                    (relation_data_ones, (relation_row, relation_col)), shape=(self.N, self.N))
                self.relation_adjacency = self.relation_adjacency.tocsc()
                self.relation_adjacency_matrices[k] = self.relation_adjacency
                self.relations_adjacency_node_counts[k] = int(len(list(set(nodes_r))))
                self.A_schema = identity(self.N).tocsc()
                self.L_schema = identity(self.N).tocsc()
            # OPERATIONS
            self.algorithms = Algorithms(self)
            # DEGREE MATRIX
            d_row = []
            d_col = []
            d_data = []
            d_data_weights = []
            for n in self.nodes.keys():
                index = int(self.node_index[n])
                d_row.append(index)
                d_col.append(index)
                d = float(len(self.get_neighbors(n)))
                weighted_degree = float(self.get_sum_of_neighborhood_weights(n))
                d_data_weights.append(weighted_degree)
                degree = float(len(list(self.adjacency[:, index].indices)))
                d_data.append(degree)
            self.degree = scipy.sparse.coo_matrix((d_data, (d_row, d_col)), shape=(self.N, self.N))
            self.degree_weighted = scipy.sparse.coo_matrix((d_data_weights, (d_row, d_col)), shape=(self.N, self.N))
            self.degree = self.degree.tocsc()
            self.degree_weighted = self.degree.tocsc()
            self.relation_degree_matrices = {}
            for k, v in self.schema.items():
                r_d_row = []
                r_d_col = []
                r_d_data = []
                for n in self.nodes.keys():
                    index = int(self.node_index[n])
                    r_d_row.append(index)
                    r_d_col.append(index)
                    r_degree = float(len(list(self.get_relational_neighbors(k, n))))
                    r_d_data.append(r_degree)
                r_degree = scipy.sparse.coo_matrix((r_d_data, (r_d_row, r_d_col)), shape=(self.N, self.N))
                self.relation_degree_matrices[k] = r_degree
            # LAPLACIAN MATRIX
            self.laplacian = self.degree - self.adjacency
            self.laplacian_weighted = self.degree - self.adjacency_weighted
            self.laplacian_weighted_degree_weighted = self.degree_weighted - self.adjacency_weighted
            # NETWORKX
            g = nx.Graph()
            g.add_weighted_edges_from(self.nx_edges)
            self.to_networkx = g

            self.m = self.adjacency_weighted.sum() * 0.5
            cluster_data = []
            cluster_rows = []
            cluster_cols = []
            for n, a in self.nodes.items():
                cluster_rows.append(int(self.node_index[n]))
                cluster_cols.append(int(a["cluster"]))
                cluster_data.append(1)
            max_labels = max(cluster_cols) + 1
            self.node_cluster_matrix = scipy.sparse.csc_matrix(
                (np.array(cluster_data), (np.array(cluster_rows), np.array(cluster_cols))),
                shape=(self.N, max_labels))
        else:
            pass

    def info(self, log_file=None):
        print("GraphObject graph object")
        print("Number of nodes: " + str(len(self.nodes.keys())))
        print("Number of represented nodes: " + str(len(self.unique_nodes)))
        print("Number of single nodes: " + str(len(self.single_nodes)))
        print("Number of edges: " + str(len(self.edges)))
        if log_file:
            print("GraphObject graph object", file=log_file)
            print("Number of nodes: " + str(len(self.nodes.keys())), file=log_file)
            print("Number of represented nodes: " + str(len(self.unique_nodes)), file=log_file)
            print("Number of single nodes: " + str(len(self.single_nodes)), file=log_file)
            print("Number of edges: " + str(len(self.edges)), file=log_file)

    def update_node_labels_with_components(self, components):
        # convert pair list to node labels
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        clusters_dict = {}
        for c in cluster_list:
            clusters_dict[c[1]] = c[0]
        clusters_res = defaultdict(list)
        for key, val in sorted(clusters_dict.items()):
            clusters_res[val].append(key)
        for n in self.nodes.keys():
            self.node_label[n] = None
        i = 0
        for k, v in clusters_res.items():
            for n in v:
                self.node_label[n] = i
            i = i + 1
        k = i
        for n in self.nodes.keys():
            if self.node_label[n] is None:
                self.node_label[n] = k
                k = k + 1
        self.node_label_profile = defaultdict(list)
        for key, val in sorted(self.node_label.items()):
            self.node_label_profile[val].append(key)
        self.node_label_profile = dict(self.node_label_profile)

    def update_node_labels(self, new_node_labels):
        self.node_label = new_node_labels
        self.node_label_profile = defaultdict(list)
        for key, val in sorted(self.node_label.items()):
            self.node_label_profile[val].append(key)
        self.node_label_profile = dict(self.node_label_profile)

    def node_labels_to_components(self):
        components = []
        for cc in self.node_label_profile.values():
            components.append(cc)
        return components

    def filter_edges(self, threshold, link_single_nodes):
        filtered_edges = []
        for e in self.edges:
            w = e[2]
            if w > threshold:
                filtered_edges.append((e[0], e[1], w))
        return Graph(filtered_edges, nodes=self.nodes, undirected=True, link_single_nodes=link_single_nodes)

    def subgraph(self, nodes, link_single_nodes):
        edges = []
        for n in nodes:
            neighbors = self.get_neighbors(n)
            for nn in neighbors:
                if nn not in nodes:
                    continue
                if n < nn:
                    edges.append((n, nn, self.get_edge_weight(n, nn)))
        vertices = {}
        l = 0
        for nn in nodes:
            vertices[str(nn)] = {"real_id": self.node_real_id[nn], "type": self.node_type[nn],
                                 "attribute": self.node_attributes[nn],
                                 "label": int(l)}
            l = l + 1
        return Graph(edges, nodes=vertices, undirected=True, link_single_nodes=link_single_nodes)

    def recast_graph(self, new_edges, link_single_nodes):
        return Graph(new_edges, nodes=self.nodes, undirected=True, link_single_nodes=link_single_nodes)

    def get_relationship_types(self, node_type_1, node_type_2):
        relationship_types = []
        for key, info in self.schema:
            if info["source"] == node_type_1 and info["target"] == node_type_2:
                relationship_types.append(info["type"])
        return relationship_types

    def create_second_degree_graph(self):
        new_edges = {}
        new_nodes = {}
        new_node_id = 0
        sorted(self.edges, key=itemgetter(0))
        for e in self.edges:
            s_n = str(self.node_index[e[0]])
            t_n = str(self.node_index[e[1]])
            w = e[2]
            t = e[3]
            s_n_label = self.label_schema.unique_labels[int(self.node_label[s_n])]
            t_n_label = self.label_schema.unique_labels[int(self.node_label[t_n])]

            common_parent_label = self.label_schema.find_common_parent(s_n_label, t_n_label)
            parent_label_index = self.label_schema.unique_labels.index(common_parent_label)

            new_node_id = new_node_id + 1
            if s_n < t_n:
                new_nodes[s_n + t_n] = {
                    "alt_id": new_node_id,
                    "type": "type",
                    "label": str(parent_label_index),
                    "cluster": 1,
                    "attributes": [s_n, t_n],
                    "features": None
                }
        for n1, info1 in new_nodes.items():
            nodes1 = set(info1["attributes"])
            for n2, info2 in new_nodes.items():
                if n1 == n2:
                    continue
                # elif n1<n2:
                nodes2 = set(info2["attributes"])
                intersect = list(nodes1.intersection(nodes2))
                if len(intersect) > 0:
                    new_edges[n1 + n2] = {
                        "source": n1, "target": n2,
                        "type": "second_degree", "weight": float(1)}
        return Graph(new_edges, nodes=new_nodes, label_schema=self.label_schema, undirected=True,
                     link_single_nodes=True)

    def reduce_graph(self):
        new_nodes = {}
        new_edges = {}
        new_id = 0
        new_label = 0
        for cluster_id, cluster_members in self.node_label_profile.items():
            new_nodes[cluster_id] = {
                "alt_id": new_id,
                "type": "type",
                "label": str(new_label),
                "attribute": None,
                "features": None
            }
            new_label = new_label + 1
            new_id = new_id + 1
            for n_c in cluster_members:
                neighbors = self.get_neighbors(n_c)
                for nbr in neighbors:
                    if nbr not in cluster_members:
                        source_cluster = str(self.node_label[n_c])
                        target_cluster = str(self.node_label[nbr])
                        edge_types = self.get_relationship_types(self.node_type[n_c], self.node_type[nbr])
                        if source_cluster != target_cluster:
                            for et in edge_types:
                                new_edges[str(cluster_id) + et + str(target_cluster)] = {
                                    "source": str(cluster_id), "target": str(target_cluster),
                                    "type": et, "weight": float(1)}
        return Graph(new_edges, nodes=new_nodes, undirected=True, link_single_nodes=True)

    def is_complete_and_full(self):
        number_of_edges = len(self.edges)
        complete_edges = (self.N * (self.N - 1)) / 2
        check = False
        if number_of_edges == complete_edges:
            s = 0
            for e in self.edges:
                s = s + e[2]
            if s == number_of_edges:
                check = True
        return check

    def create_hetergenous_graph(self, length_of_metapath, number_of_metapaths):
        metapaths_commuting_matrices = []
        for m in range(number_of_metapaths):
            metapath = []
            start_edge = self.schema[random.choice(list(self.schema.keys()))]
            for _ in range(length_of_metapath):
                neighbor_edges = []
                for e_k, e_v in self.schema.items():
                    if start_edge["target"] == e_v["source"]:
                        neighbor_edges.append(e_k)
                next_edge = self.schema[random.choice(neighbor_edges)]
                metapath.append((next_edge["source"], next_edge["type"], next_edge["target"]))
                start_edge = next_edge
            A_tot = identity(self.N).tocsc()
            for r in metapath:
                A_tot = A_tot @ self.relation_adjacency_matrices[str(str(r[0]) + str(r[1]) + str(r[2]))]
            print([A_tot])
            metapaths_commuting_matrices.append(A_tot)
            self.A_schema = self.A_schema + A_tot
        self.A_schema = self.A_schema
        self.L_schema = self.degree - self.A_schema

    def create_relational_adjacencies(self):
        self.relation_degree_normalized_adjacency_matrices = {}
        for r, A_r in self.relation_adjacency_matrices.items():
            r_degree = self.relation_degree_matrices[r].todense()
            det_r = scipy.linalg.det(r_degree)
            if det_r != 0:
                r_inv_D = scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(r_degree))
            else:
                r_inv_D = scipy.sparse.csc_matrix(np.identity(self.N))
            self.relation_degree_normalized_adjacency_matrices[r] = tf.constant(
                tf.convert_to_tensor(np.asarray(r_inv_D.dot(A_r).todense(), dtype=np.float32), dtype=tf.float32))

    def balanced_node_label_sampler(self, split_percent):
        k = np.floor((split_percent * self.N) / 100)
        number_of_classes = len(list(self.node_label_profile.keys()))
        sample_label_size = np.floor(k / number_of_classes)
        final_samples = []
        for labels, nodes in self.node_label_profile.items():
            number_of_samples = 0
            if sample_label_size > len(nodes):
                number_of_samples = len(nodes)
            elif sample_label_size <= len(nodes):
                number_of_samples = sample_label_size
            samples = list(np.random.choice(nodes, size=int(number_of_samples), replace=False))
            final_samples = final_samples + samples
        train_samples = list(set(final_samples))
        rest_samples = list(set(list(self.nodes.keys())) - set(train_samples))
        middle_index = np.math.floor(len(rest_samples) / 2)
        valid_samples = rest_samples[0:middle_index]
        test_samples = rest_samples[middle_index:]
        train_samples_indices = []
        for s in train_samples:
            t_s = self.node_index[s]
            train_samples_indices.append(t_s)
        valid_samples_indices = []
        for s in valid_samples:
            t_s = self.node_index[s]
            valid_samples_indices.append(t_s)
        test_samples_indices = []
        for s in test_samples:
            t_s = self.node_index[s]
            test_samples_indices.append(t_s)
        return train_samples_indices, valid_samples_indices, test_samples_indices

    def balanced_node_type_sampler(self, split_percent):
        k = np.floor((split_percent * self.N) / 100)
        number_of_classes = len(list(self.node_type_profile.keys()))
        sample_label_size = np.floor(k / number_of_classes)
        final_samples = []
        for labels, nodes in self.node_type_profile.items():
            number_of_samples = 0
            if sample_label_size > len(nodes):
                number_of_samples = len(nodes)
            elif sample_label_size <= len(nodes):
                number_of_samples = sample_label_size
            samples = list(np.random.choice(nodes, size=int(number_of_samples), replace=False))
            final_samples = final_samples + samples
        train_samples = list(set(final_samples))
        rest_samples = list(set(list(self.nodes.keys())) - set(train_samples))
        middle_index = np.math.floor(len(rest_samples) / 2)
        valid_samples = rest_samples[0:middle_index]
        test_samples = rest_samples[middle_index:]
        train_samples_indices = []
        for s in train_samples:
            t_s = self.node_index[s]
            train_samples_indices.append(t_s)
        valid_samples_indices = []
        for s in valid_samples:
            t_s = self.node_index[s]
            valid_samples_indices.append(t_s)
        test_samples_indices = []
        for s in test_samples:
            t_s = self.node_index[s]
            test_samples_indices.append(t_s)
        return train_samples_indices, valid_samples_indices, test_samples_indices

    def prepare_graph_for_machine_learning_type(self, split):
        # Training data-------------------------------------------------------------------------------------
        y = []
        x = []
        index = []
        for n, info in self.nodes.items():
            x.append(str(n))
            index.append(self.node_index[n])
            y_probabilty_vector = list(np.zeros((len(self.node_type_count.keys()))))
            target_position = list(self.node_type_profile.keys()).index(str(info["type"]))
            y_probabilty_vector[target_position] = 1.0
            y.append(y_probabilty_vector)
        self.y = tf.reshape(tf.convert_to_tensor(y, dtype=tf.float32), shape=(self.N, len(self.node_type_count.keys())))
        train_samples, valid_samples, test_samples = self.balanced_node_type_sampler(split)
        train_mask = np.zeros((self.N), dtype=int)
        train_mask[train_samples] = 1
        self.train_mask = train_mask.astype(bool)
        valid_mask = np.zeros((self.N), dtype=int)
        valid_mask[valid_samples] = 1
        self.valid_mask = valid_mask.astype(bool)
        test_mask = np.zeros((self.N), dtype=int)
        test_mask[test_samples] = 1
        self.test_mask = test_mask.astype(bool)

    def prepare_graph_for_machine_learning_label(self, split):
        # Training data-------------------------------------------------------------------------------------
        y = []
        x = []
        index = []
        for n, info in self.nodes.items():
            x.append(str(n))
            index.append(self.node_index[n])
            y_probabilty_vector = list(np.zeros((len(self.node_label_profile.keys()))))
            target_position = list(self.node_label_profile.keys()).index(str(info["label"]))
            y_probabilty_vector[target_position] = 1.0
            y.append(y_probabilty_vector)
        self.y = tf.reshape(tf.convert_to_tensor(y, dtype=tf.float32),
                            shape=(self.N, len(self.node_label_profile.keys())))
        train_samples, valid_samples, test_samples = self.balanced_node_label_sampler(split)
        train_mask = np.zeros((self.N), dtype=int)
        train_mask[train_samples] = 1
        self.train_mask = train_mask.astype(bool)
        valid_mask = np.zeros((self.N), dtype=int)
        valid_mask[valid_samples] = 1
        self.valid_mask = valid_mask.astype(bool)
        test_mask = np.zeros((self.N), dtype=int)
        test_mask[test_samples] = 1
        self.test_mask = test_mask.astype(bool)

    def get_edge_weight(self, n1, n2):
        return float(self.adjacency_weighted[self.node_index[n1], self.node_index[n2]])

    def get_node_label(self, node):
        return self.node_label[node]

    def set_node_label(self, l, node):
        self.node_label[node] = l

    def get_node_real_id(self, node):
        return self.node_real_id[node]

    def get_node_weighted_degree(self, node):
        return float(self.degree_weighted[self.node_index[node], self.node_index[node]])

    def get_node_degree(self, node):
        return float(self.degree[self.node_index[node], self.node_index[node]])

    def get_neighbors(self, node):
        neighbors = []
        neighbor_indices = self.adjacency[:, self.node_index[node]].indices
        for i in neighbor_indices:
            if i == self.node_index[node]:
                continue
            neighbors.append(self.node_inverted_index[i])
        return neighbors

    def get_relational_neighbors(self, relationship_type, node):
        neighbors = []
        neighbor_indices = self.relation_adjacency_matrices[relationship_type][:,
                           self.node_index[node]].indices
        for i in neighbor_indices:
            if i == self.node_index[node]:
                continue
            neighbors.append(self.node_inverted_index[i])
        return neighbors

    def get_sum_of_neighborhood_weights(self, node):
        return float(self.adjacency_weighted[:, self.node_index[node]].sum(axis=0)[0][0])

    def get_neighborhood_weights(self, node):
        neighbor_indices = self.adjacency_weighted[:, self.node_index[node]].indices
        weights = []
        for i in neighbor_indices:
            w = self.adjacency_weighted[i, self.node_index[node]]
            weights.append(w)
        return weights

    def get_cluster_members_names(self, cluster_id):
        members = []
        members_indices = list(np.where((self.node_cluster_matrix.toarray()[:, cluster_id] == 1))[0])
        for i in members_indices:
            members.append(self.node_inverted_index[i])
        # return list({key: value for key, value in self.graph.node_label.items() if value == cluster_id}.keys())
        return members

    def get_cluster_members_indices(self, cluster_id):
        # return list({key: value for key, value in self.graph.node_label.items() if value == cluster_id}.keys())
        return list(np.where((self.node_cluster_matrix.toarray()[:, cluster_id] == 1))[0])

    def get_sum_of_incident_nodes_to_cluster(self, cluster_id):
        cluster_members = self.get_cluster_members_names(cluster_id)
        sum = 0
        for n in cluster_members:
            neighbors = self.get_neighbors(n)
            for nbr in neighbors:
                if nbr not in cluster_members:
                    sum = sum + self.get_edge_weight(n, nbr)
        return sum

    def get_sum_of_incident_nodes_to_cluster_new_node(self, cluster_id, node):
        cluster_members = self.get_cluster_members_names(cluster_id)
        cluster_members.append(node)
        sum = 0
        for n in cluster_members:
            neighbors = self.get_neighbors(n)
            for nbr in neighbors:
                if nbr not in cluster_members:
                    sum = sum + self.get_edge_weight(n, nbr)
        return sum

    def get_sum_of_incident_nodes_to_cluster_without_node(self, cluster_id, node):
        cluster_members = self.get_cluster_members_names(cluster_id)
        if node in cluster_members:
            cluster_members.remove(node)
        sum = 0
        for n in cluster_members:
            neighbors = self.get_neighbors(n)
            for nbr in neighbors:
                if nbr not in cluster_members:
                    sum = sum + self.get_edge_weight(n, nbr)
        return sum

    def get_sum_of_weights_between_node_and_cluster(self, node, j_membership):
        cluster_indices = self.get_cluster_members_indices(j_membership)
        node_index = self.node_index[node]
        return float(self.adjacency_weighted[cluster_indices, node_index].sum())

    def get_sum_of_weights_inside_cluster(self, cluster_id):
        cluster_indices = self.get_cluster_members_indices(cluster_id)
        return float(self.adjacency_weighted[cluster_indices, cluster_indices].sum() * 0.5)

    def get_sum_of_weights_inside_cluster_with_new_node(self, cluster_id, node):
        cluster_members = self.get_cluster_members_names(cluster_id)
        if node not in cluster_members:
            cluster_members.append(node)
        cluster_indices = []
        for n in cluster_members:
            cluster_indices.append(self.node_index[n])
        return float(self.adjacency_weighted[cluster_indices, cluster_indices].sum() * 0.5)

    def get_sum_of_weights_inside_cluster_without_node(self, cluster_id, node):
        cluster_members = self.get_cluster_members_names(cluster_id)
        if node in cluster_members:
            cluster_members.remove(node)
        cluster_indices = []
        for n in cluster_members:
            cluster_indices.append(self.node_index[n])
        return float(self.adjacency_weighted[cluster_indices, cluster_indices].sum() * 0.5)

    # MODULARITY FUNCTIONS
    def compute_modularity(self):
        res = 0
        for com, nodes in self.node_label_profile.items():
            for i in nodes:
                for j in nodes:
                    a_i_j = self.get_edge_weight(i, j)
                    k_i = self.get_sum_of_neighborhood_weights(i)
                    k_j = self.get_sum_of_neighborhood_weights(j)
                    res = res + (a_i_j - ((k_i * k_j) / (2 * self.m)))
        modularity = res / (2 * self.m)
        return modularity

    def compute_delta_modularity_node_add(self, cluster, node):
        sigma_in = self.get_sum_of_weights_inside_cluster_with_new_node(cluster, node)
        sigma_total = self.get_sum_of_incident_nodes_to_cluster_new_node(cluster, node)
        k_i_in = self.get_sum_of_weights_between_node_and_cluster(node, cluster)
        k_i = self.get_sum_of_neighborhood_weights(node)
        delta_modularity = (((sigma_in + k_i_in) / (2 * self.m)) - ((sigma_total + k_i) / (2 * self.m)) ** 2) - (
                (sigma_in / (2 * self.m)) - ((sigma_total / (2 * self.m)) ** 2) - ((k_i / (2 * self.m)) ** 2))
        return delta_modularity

    def compute_delta_modularity_node_remove(self, cluster, node):
        sigma_in = self.get_sum_of_weights_inside_cluster_without_node(cluster, node)
        sigma_total = self.get_sum_of_incident_nodes_to_cluster_without_node(cluster, node)
        k_i_in = self.get_sum_of_weights_between_node_and_cluster(node, cluster)
        k_i = self.get_sum_of_neighborhood_weights(node)
        delta_modularity = (((sigma_in + k_i_in) / (2 * self.m)) - ((sigma_total + k_i) / (2 * self.m)) ** 2) - (
                (sigma_in / (2 * self.m)) - ((sigma_total / (2 * self.m)) ** 2) - ((k_i / (2 * self.m)) ** 2))
        return delta_modularity

    # ATTRIBUTE TASKS
    def __calculate_cluster_entropy(self, cluster):
        tokenCount = 0
        refCnt = len(cluster)
        if refCnt > 1:
            for j in range(0, refCnt):
                tokenCount = tokenCount + len(cluster[j])
            baseProb = 1 / float(refCnt)
            base = -tokenCount * baseProb * math.log(baseProb, 2)
            entropy = 0.0
            clusterSize = len(cluster)
            for j in range(0, len(cluster) - 1):
                jList = cluster[j]
                for token in jList:
                    cnt = 1
                    for k in range(j + 1, len(cluster)):
                        if token in cluster[k]:
                            cnt += 1
                            cluster[k].remove(token)
                    tokenProb = cnt / clusterSize
                    term = -tokenProb * math.log(tokenProb, 2)
                    entropy += term
                    quality = 1.0 - entropy / base
                    cnt = 0
            for token in cluster[clusterSize - 1]:
                tokenProb = 1.0 / clusterSize
                term = -tokenProb * math.log(tokenProb, 2)
                entropy += term
                quality = 1.0 - entropy / base
            quality = 1.0 - entropy / base
        else:
            quality = 1.0
        return float(quality)

    def compute_graph_entropy(self):
        communities_grouped = defaultdict(list)
        for key, value in sorted(self.node_label_profile.items()):
            communities_grouped[value].append(key)
        communities_grouped = dict(communities_grouped)
        entropies = []
        for cluster, records in communities_grouped.items():
            cluster = []
            for r in records:
                cluster.append(self.node_attributes[r])
            quality = self.__calculate_cluster_entropy(cluster)
            entropies.append(quality)
        return round(np.mean(np.array(entropies)), 4)

    def degree_normalized_adjacency(self):
        return np.asarray(scipy.sparse.linalg.inv(self.degree).dot(self.adjacency + self.identity).todense(),
                          dtype=np.float32)

    def degree_normalized_second_degree_adjacency(self):
        pass

    def degree_normalized_adjacency_tensorflow(self):
        inv_D = scipy.sparse.linalg.inv(self.degree)
        degree_normalized_adjacency = tf.Variable(
            tf.convert_to_tensor(np.asarray(inv_D.dot(self.adjacency + self.identity).todense(), dtype=np.float32),
                                 dtype=tf.float32))
        return degree_normalized_adjacency

    def degree_normalized_adjacency_cluster_gcn_tenorflow(self, lmda):
        inv_D = scipy.sparse.linalg.inv(self.degree + self.identity)
        degree_normalized_adjacency = inv_D.dot(self.adjacency_identity).todense()
        eigenvals, eigenvecs = eigsh(degree_normalized_adjacency, k=self.N)
        eigenvals = np.diag(eigenvals)
        degree_normalized_adjacency = degree_normalized_adjacency + (lmda * eigenvals)
        degree_normalized_adjacency = tf.constant(
            tf.convert_to_tensor(np.asarray(degree_normalized_adjacency, dtype=np.float32), dtype=tf.float32))
        return degree_normalized_adjacency

    def degree_normalized_adjacency_identity(self):
        inv_D = scipy.sparse.linalg.inv(self.degree)
        degree_normalized_adjacency = inv_D.dot(self.adjacency_identity).todense()
        return degree_normalized_adjacency

    def degree_normalized_relational_adjacencies(self):
        self.relation_degree_normalized_adjacency_matrices = {}
        inv_D = scipy.sparse.linalg.inv(self.degree)
        for k, A_r in enumerate(self.relation_degree_normalized_adjacency_matrices):
            self.relation_degree_normalized_adjacency_matrices[k] = inv_D.dot(A_r).todense()

    def degree_normalized_relational_adjacencies_tensorflow(self):
        relation_degree_normalized_adjacency_matrices = {}
        inv_D = scipy.sparse.linalg.inv(self.degree)
        for r, A_r in self.relation_adjacency_matrices.items():
            relation_degree_normalized_adjacency_matrices[r] = tf.constant(
                tf.convert_to_tensor(np.asarray(inv_D.dot(A_r).todense(), dtype=np.float32), dtype=tf.float32))
        return relation_degree_normalized_adjacency_matrices
