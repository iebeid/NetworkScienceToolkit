import argparse
import numpy as np
from scipy import sparse, io
import math
import random

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from models.data_utils import *

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Process walks")
    parser.add_argument('--input', nargs='?', default='emb/wn18/walks.txt',
                        help='Input walks path')
    parser.add_argument('--subwalk-length', type=int, default=5,
                    help='Length of subwalks to generate from each walk. Default is 5. Counts edges AND nodes. ')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10. Counts nodes and paths. ')
    parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')
    parser.add_argument('--output', nargs='?', default='emb/wn18/vectors.txt',
                    help='ALL embeddings output path')
    parser.add_argument('--node-list', nargs='?', default='data/wn18/nodes.txt', help="List of nodes")
    parser.add_argument('--process-method', nargs='?', default='nodes',
                        help='What method to use to process relation walks. Provided are "relational", "paths" and "nodes". "nodes" is default. ')
    parser.add_argument('--dimensions', type=int, default=500,
                        help='Number of dimensions. Default is 500.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    return parser.parse_args()

def process_walk_only_nodes(walk):
    return [walk[::2]]

def process_walk_relational(walk):
    walks = []
    for idx in range(0, len(walk) - 2, 2):
        head = walk[idx]
        edge = walk[idx + 1]
        tail = walk[idx + 2]
        nwalk = [head, edgepath2str([edge], tail), tail]
        # We want the head to predict the tail + relation, the tail alone, and the relation alone, in that order of importance.
        # The relation alone is so common that the skip-gram overcompensates for it, so I left it out.
        walks.append(nwalk)
    return walks

"""
Predict_all returns a list of walks, processed to merge edges and tail nodes.
"""
def process_walk_predict_all(walk, subwalk_length):
    walks = []
    for idx in range(0, len(walk) - subwalk_length + 1, 2):
        start_node = walk[idx]
        newwalk = [start_node]
        edgelist = []
        for j in range(idx + 1, subwalk_length + idx, 2):
            newedge = walk[j]
            newnode = walk[j+1]
            edgelist.append(newedge)
            newwalk.append(newnode)
            newwalk.append(edgepath2str(edgelist, newnode))
        walks.append(newwalk) #Append is thread-safe.
    return walks


def main(args):
    print("------loading walks------")
    walks = load_walks(args.input)
    print("------processing walks--------")
    node_dict = load_line_list(args.node_list)
    num_nodes = len(node_dict)
    processed_walks = []
    processor = None
    train_on_pairs = False
    if args.process_method == "nodes":
        train_on_pairs = False
        processor = (lambda walk: process_walk_only_nodes(walk))
    elif args.process_method == "relational":
        train_on_pairs = True
        processor = (lambda walk: process_walk_relational(walk))
    elif args.process_method == "paths":
        train_on_pairs = True
        processor = (lambda walk: process_walk_predict_all(walk, args.subwalk_length))
    else:
        print("Unspecified process method %s! Using 'nodes' instead. " % args.process_method)
        processor = (lambda walk: process_walk_only_nodes(walk))
    print("Processing %d walks..." % len(walks))
    for walk in (walks):
        processed_walks.extend(processor(walk))

    # print walks
    print("------Made %d walks-------" % len(processed_walks))
    print("------Running Word2Vec on processed walks--------")
    model = Word2Vec(processed_walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
    model.init_sims(replace=True)
    print("--------Generating numpy featarray-----------")
    feats = np.array([model.wv[str(i)] for i in range(num_nodes)])
    filename = args.output.replace('.txt', '.npm')
    fp = np.memmap(filename, dtype='float32', mode='w+', shape=feats.shape)
    fp[:] = feats[:]
    del fp
    comp_np = args.output.replace('.txt', '')
    np.savez_compressed(comp_np, features=feats)
    print("------Saving embeddings to file %s--------" % args.output)
    model.wv.save_word2vec_format(args.output)
    # Save numpy vectors.
    save_features(args.output + "np", feats)
if __name__ == "__main__":
    args = parse_args()

    main(args)
