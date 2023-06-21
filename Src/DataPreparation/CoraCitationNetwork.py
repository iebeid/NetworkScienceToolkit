
import pandas as pd
from Src.Graph.Graph import Graph

class CoraCitationNetwork:

    def __init__(self, nodes_data_file, edges_data_file):
        print("Loading the Cora Network")
        self.nodes_data_file = nodes_data_file
        self.edges_data_file = edges_data_file
        self.label_schema = [("Science" ,"CS") ,("Science" ,"Stats") ,("CS" ,"Theory") ,("CS" ,"Algorithms"),
                ("Stats" ,"Probabilistic_Methods") ,("Stats" ,"Case_Based"),
                ("Probabilistic_Methods" ,"Rule_Learning") ,("Algorithms" ,"Genetic_Algorithms"),
                ("Algorithms" ,"Machine_Learning") ,("Machine_Learning" ,"Reinforcement_Learning"),
                ("Machine_Learning" ,"Neural_Networks")]

    def find_root(self, schema):
        root = None
        # find unique leaves
        leaves = []
        for edge in schema:
            s_n = edge[0]
            t_n = edge[1]
            leaves.append(s_n)
            leaves.append(t_n)
        leaves = list(set(leaves))
        for leaf in leaves:
            check = True
            for edge in schema:
                s_n = edge[0]
                t_n = edge[1]
                if t_n == leaf:
                    check = False
            if check:
                root = leaf
        return root

    # find the edge where the leaf is a target node
    def find_parent(self, schema, leaf):
        parent = None
        for edge in schema:
            s_n = edge[0]
            t_n = edge[1]
            if t_n == leaf:
                parent = s_n
                break
        return parent

    def find_lineage(self, schema, leaf):
        lineage = []
        root = self.find_root(schema)
        parent = leaf
        lineage.append(leaf)
        while parent != root:
            parent = self.find_parent(schema, parent)
            lineage.append(parent)
        lineage.reverse()
        return lineage

    def load(self):
        nodes_df = pd.read_csv(self.nodes_data_file)
        edges_df = pd.read_csv(self.edges_data_file)
        print(nodes_df.info())
        unique_labels = []
        unique_types = []
        nodes = {}
        for index, row in nodes_df.iterrows():
            nodes[str(row["nodeId"])]=str(row["subject"])
            unique_labels.append(str(row["labels"]))
            unique_types.append(str(row["subject"]))
        unique_labels = list(set(unique_labels))
        unique_types = list(set(unique_types))
        print(unique_labels)
        print(unique_types)
        print(len(unique_labels))
        print(len(unique_types))
        nodes = {}
        for index, row in nodes_df.iterrows():
            label = str(':'.join(self.find_lineage(self.label_schema,str(row["subject"]))))
            nodes[str(row["nodeId"])]={
                    "alt_id": "none",
                    "type": str(row["subject"]),
                    "label": label,
                    "cluster": 0,
                    "attributes": None,
                    "features": row["features"]
                }
        edges = {}
        c=0
        for index, row in edges_df.iterrows():
            edges[c] = {"source": str(row["sourceNodeId"]), "target": str(row["targetNodeId"]),
                                 "type": str(row["relationshipType"]), "weight": float(1)}
            edges[c+1] = {"source": str(row["targetNodeId"]), "target": str(row["sourceNodeId"]),
                                 "type": str(row["relationshipType"]), "weight": float(1)}
            c=c+2

        self.graph = Graph(edges, nodes=nodes, undirected=True, link_single_nodes=False)
        print(self.graph.info())