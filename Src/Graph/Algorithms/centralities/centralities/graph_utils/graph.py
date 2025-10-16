import os
import networkx as nx


def construct_graph(filepath, node2class, edge2class):

    G = nx.DiGraph()
    with open(filepath) as file:
        for line in file:
            [node1_link, edge_link, node2_link] = line.strip('\n').split('\t')
            node1_type = node2class.get(node1_link.strip())
            node2_type = node2class.get(node2_link.strip())
            edge_type = edge2class.get(edge_link.strip())

            node1_type = node1_type.strip() if node1_type is not None else None
            node2_type = node2_type.strip() if node2_type is not None else None
            edge_type = edge_type.strip() if edge_type is not None else None

            node1_name = get_name(node1_link)
            node2_name = get_name(node2_link)
            edge_name = get_name(edge_link)

            G.add_node(node1_link, link=node1_link, name=node1_name, type=node1_type)
            G.add_node(node2_link, link=node2_link, name=node2_name, type=node2_type)

            G.add_edge(node1_link, node2_link, name=edge_name, edge_type=edge_type, link=edge_link)

    return G

def get_name(path):
    ''' Get the name of node at end of path '''
    return os.path.basename(path)
