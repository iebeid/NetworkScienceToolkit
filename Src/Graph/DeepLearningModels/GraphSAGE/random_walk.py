import argparse
import random
import networkx as nx

def parse_args():
    parser = argparse.ArgumentParser(description="Random Walk")
    parser.add_argument('--graph', nargs='?', default=None)
    parser.add_argument('--nodes', nargs='?', default='data/c2b2rdf/test/nodes.txt', help='nodes file')
    parser.add_argument('--walk-len', nargs='?', default=5, help='nodes file')
    parser.add_argument('--n-walks', nargs='?', default=10, help='nodes file')

    parser.add_argument('--output', nargs='?', default=None, help="file to save embeddings")
    return parser.parse_args()

def run_random_walk(G, node_idx, walk_length):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs
