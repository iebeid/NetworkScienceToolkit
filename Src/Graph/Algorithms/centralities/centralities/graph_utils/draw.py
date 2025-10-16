import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class DrawGraph:

    def __init__(self):
        pass

    @staticmethod
    def draw_path(graph, nodes, edges):
        
        nodes = list(map(graph.nodes.get, nodes))

        G = nx.path_graph(len(nodes))

        names = {}
        for i, node in enumerate(nodes):
            name = f'{node.node_class}: {node.name}'
            G.add_node(i)
            names[i] = name

        for edge in edges:

            node_src = edge.origin
            node_dest = edge.dest

            node_src_name = f'{node_src.node_class}: {node_src.name}'
            node_dest_name = f'{node_dest.node_class}: {node_dest.name}'

            G.add_edge(*(node_src_name, node_dest_name))

        fig, ax = plt.subplots(figsize=(10,10))
        ax.axis('equal')
        pos = nx.spring_layout(G)
        pos_higher = {}
        for k, v in pos.items():
            pos_higher[k] = (v[0], v[1]+0.08)
       
        G = nx.relabel_nodes(G, names)

        nx.draw_networkx(G, pos=pos, with_labels=True, font_size=8, ax=ax)
        nx.draw_networkx_labels(G, pos=pos_higher, labels=names, ax=ax)
        plt.savefig("path.png", bbox_inches='tight') # save as png
        plt.show() # display