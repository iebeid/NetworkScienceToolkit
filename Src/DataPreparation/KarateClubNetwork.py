import time
import csv
import networkx as nx
import numpy as np
from Src.Graph.Graph import Graph
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from Src.Graph.Utils import HierarchicalLabeler


class KarateClubNetwork:
    def __init__(self, data_file1, datafile2):
        print("Loading the Karate Club Network")
        self.data_file = data_file1
        self.metadata = datafile2

    def load(self):
        edges = []
        kg_edges = []
        with open(self.data_file, newline="\n") as file:
            data = file.readlines()
            edge_data = data[0:34]
            layer_data = data[34:]
            layers = np.zeros((34, 34))
            for row, line in enumerate(layer_data):
                values = line.strip().split(" ")
                for col, value in enumerate(values):
                    # value = values[col]
                    if value != str(0):
                        layers[row][col] = value
            for row, line in enumerate(edge_data):
                values = line.strip().split(" ")
                for col, value in enumerate(values):
                    # value = values[col]
                    if value != str(0):
                        kg_edges.append((str(row), str(col)))
                        edges.append((str(row), str(int(layers[row][col])), str(col)))
                        # edges.append((str(row), str(int(layers[row][row])), str(row)))
                        # edges.append((str(col), str(int(layers[col][col])), str(col)))
        edges = list(set(edges))
        edges.sort()
        filtered_edges = {}
        edge_index = 0
        for e in edges:
            s_n = e[0]
            r = e[1]
            t_n = e[2]
            filtered_edges[str(edge_index)] = {"source": s_n, "target": t_n, "type": r, "weight": float(1)}
            edge_index = edge_index + 1


        label_schema = [("Science", "CS"), ("Science", "Stats"), ("CS", "Theory"), ("CS", "Algorithms"),
                        ("Stats", "Probabilistic_Methods"), ("Stats", "Case_Based"),
                        ("Probabilistic_Methods", "Rule_Learning"), ("Algorithms", "Genetic_Algorithms"),
                        ("Algorithms", "Machine_Learning"), ("Machine_Learning", "Reinforcement_Learning"),
                        ("Machine_Learning", "Neural_Networks")]
        hl = HierarchicalLabeler(label_schema)

        node_labels_dict = {}
        with open(self.metadata, newline="\n") as file:
            data = file.readlines()
            for line in data:
                values = line.strip().split("\t")
                i = hl.unique_labels.index(str(values[1]))
                node_labels_dict[int(values[0])] = str(i)
        nodes_dict = {}
        l = 0
        KG = nx.Graph(kg_edges)
        cc = nx.algorithms.community.louvain_communities(KG)
        # cc = list(nx.algorithms.community.greedy_modularity_communities(KG))
        type = 0
        for c in cc:
            for n_t in c:
                nodes_dict[str(n_t)] = {
                    "alt_id": "none",
                    "type": str(type),
                    "label": node_labels_dict[int(n_t)],
                    "cluster": 0,
                    "attributes": None,
                    "features": None
                }
                l = l + 1
            type = type + 1


        self.graph = Graph(filtered_edges, nodes=nodes_dict, label_schema=hl,undirected=True, link_single_nodes=False)
