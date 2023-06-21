import time
import scipy.spatial
from scipy.stats import norm
import numpy as np


class MathUtils:
    def cosine_distance(self, vector1, vector2):
        return 1 - scipy.spatial.distance.cosine(vector1, vector2)

    def euclidean_distance(self, vector1, vector2):
        return scipy.spatial.distance.euclidean(vector1, vector2)

    def squared_euclidean_distance(self, vector1, vector2):
        return scipy.spatial.distance.sqeuclidean(vector1, vector2)

    def normalize_vector(self, vector):
        return list((vector - np.min(vector)) / (np.max(vector) - np.min(vector)))

    def probability_vector(self, vector):
        p_v = []
        for v in vector:
            p = norm().cdf(v)
            p_v.append(p)
        return p_v


class AlgorithmicUtils:

    @staticmethod
    def invert_dictionary(d):
        values = []
        for k, v in d.items():
            values.append(v)
        unique_values = list(set(values))
        value_profile = {}
        for v1 in unique_values:
            keys_involved = []
            for k, v2 in d.items():
                if v1 == v2:
                    keys_involved.append(k)
            keys_involved = list(set(keys_involved))
            value_profile[v1] = keys_involved
        print(value_profile)

    @staticmethod
    def measure_execution_time(func):
        def wrapper():
            start_time = time.perf_counter()
            func()
            end_time = time.perf_counter()
            print("Total runtime: " + str(round((end_time - start_time), 4)) + " seconds")

        return wrapper

    @staticmethod
    def read_config(config_file):
        config = {}
        with open(config_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                columns = line.split()
                parameter = columns[0]
                value = columns[1]
                config[parameter] = value
        return config


class HierarchicalLabeler:
    def __init__(self, schema):
        self.schema = schema
        self.root = self.find_root()
        unique_labels = []
        for e in self.schema:
            unique_labels.append(e[0])
            unique_labels.append(e[1])
        self.unique_labels = list(np.unique(np.array(unique_labels)))
        pass

    # find the root of the schema
    def find_root(self):
        root = None
        # find unique leaves
        leaves = []
        for edge in self.schema:
            s_n = edge[0]
            t_n = edge[1]
            leaves.append(s_n)
            leaves.append(t_n)
        leaves = list(set(leaves))
        for leaf in leaves:
            check = True
            for edge in self.schema:
                s_n = edge[0]
                t_n = edge[1]
                if t_n == leaf:
                    check = False
            if check:
                root = leaf
        return root

    # find the edge where the leaf is a target node
    def find_parent(self, leaf):
        parent = self.root
        for edge in self.schema:
            s_n = edge[0]
            t_n = edge[1]
            if t_n == leaf:
                parent = s_n
                break
        return parent

    def find_lineage(self, leaf):
        lineage = []
        root = self.root
        parent = leaf
        lineage.append(leaf)
        while parent != root:
            parent = self.find_parent(parent)
            lineage.append(parent)
        lineage.reverse()
        return lineage

    def find_rank_of_leaf(self, leaf):
        l = self.find_lineage(leaf)
        return l.index(leaf) + 1

    def find_common_parent(self, leaf1, leaf2):
        leaf1_lineage = self.find_lineage(leaf1)
        leaf2_lineage = self.find_lineage(leaf2)
        leaf1_lineage_set = set(leaf1_lineage)
        leaf2_lineages_set = set(leaf2_lineage)
        common_labels = list(leaf1_lineage_set.intersection(leaf2_lineages_set))
        ranks = []
        for cl in common_labels:
            rank = self.find_rank_of_leaf(cl)
            ranks.append(rank)
        max_rank_label = common_labels[ranks.index(max(ranks))]
        return max_rank_label
