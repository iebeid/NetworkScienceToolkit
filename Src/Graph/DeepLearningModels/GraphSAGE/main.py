import numpy as np
import argparse
import networkx as nx
from gensim.models.keyedvectors import KeyedVectors
import gensim
from graph_sage import graphsage_model

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def parse_args():
    parser = argparse.ArgumentParser(description="Skip-gram model")
    parser.add_argument('--vectors', nargs='?', default=None)
    parser.add_argument('--train-pos', nargs='?', default='data/c2b2rdf/test/train_pos.csv', help='train positive relations dataset')
    parser.add_argument('--train-neg', nargs='?', default='data/c2b2rdf/test/train_neg.csv', help='train negative relations dataset')
    parser.add_argument('--walks', nargs='?', default='emb/c2b2rdf/test/n2v_walks.txt', help='Walks generated from node2vec or edge2vec')
    parser.add_argument('--output', nargs='?', default=None, help="file to save embeddings")
    parser.add_argument('--nodes', nargs='?', default='data/c2b2rdf/test/nodes.txt', help='nodes file')
    parser.add_argument('--epochs', nargs='?', default=5, type=int, help='nodes file')
    parser.add_argument('--latent-dim', nargs='?', default=500, type=int, help='nodes file')

    return parser.parse_args()


def create_nodemap(node_file_path):
    d = {}

    with open(node_file_path, 'r') as nodef:

        [d.update({line.strip(): int(idx)}) for idx, line in enumerate(nodef)]

    print("found %d nodes in dict file" % len(d))
    return d

def extract_edges(relations_file):

    with open(relations_file, 'r') as rel_f:
        edges = [(int(edge.split()[0]), int(edge.split()[1])) for edge in rel_f]

    return np.array(edges)


def load_walks(file):
    walks = []
    with open(file) as f:
        for line in f:
            walk = line.split("\t")
            walks.append(walk)
    return np.array(walks)

def process_walk_only_nodes(walk):
    processed_walk = [int(node) for node in walk[::2]]
    return processed_walk

def main(args):
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    G_pos = nx.Graph()
    G_neg = nx.Graph()

    print('--- Creating Nodemap ---')
    nodemap = create_nodemap(args.nodes)

    # print('--- Loading Walks ---')
    # walks = load_walks(args.walks)
    # processed_walks = np.array([process_walk_only_nodes(walk) for walk in walks])
    processed_walks = []
    if args.vectors:
        print('--- Loading Word2Vec Model ---')
        word2vec_model = KeyedVectors.load_word2vec_format(args.vectors , binary=False)
        features = np.array([word2vec_model[str(i)] for i in range(len(nodemap))])
    else:
        features = None
    print('--- Creating Graph ---')
    G_pos.add_nodes_from(list(nodemap.values()))
    G_neg.add_nodes_from(list(nodemap.values()))

    pos_edges = extract_edges(args.train_pos)
    neg_edges = extract_edges(args.train_neg)

    G_pos.add_edges_from(pos_edges)
    G_neg.add_edges_from(neg_edges)

    print('--- Training GraphSAGE ---')
    model = graphsage_model(G_pos, G_neg, nodemap, processed_walks, latent_dim=args.latent_dim, epochs=args.epochs, features=None)
    model.config_model()
    model.train()

    embeddings = {}
    print('--- Computing Embeddings ---')
    for node in nodemap:
        node_idx = nodemap[node]

        node_center = np.zeros(1, 1+model.positive_sample_size+model.negative_sample_size, model.f_dim)
        node_feat, _ = model.aggr(model.G_pos, node_idx)
        node_center[0, 0, :] = node_feat
        embed = model.sess.run(model.dense_center,feed_dict={model.input_x_center:node_center})

        embeddings[node] = embed[0,0,:]
        print(embeddings)
        break

    # print(embeddings)
    # print('--- Saving Embeddings ---')
    # pickle.dump(embeddings, open(args.output, 'wb'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
