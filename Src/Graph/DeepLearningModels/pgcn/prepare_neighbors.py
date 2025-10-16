from models.data_utils import *
import pickle
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Create inductive embeddings.") 
    parser.add_argument('--input-walks', nargs='?', default='emb/aifb/walks.txt',
                        help='Input walks path')
    parser.add_argument('--node-features', nargs='?', default="./emb/aifb/vectors.txtnp", help='File of node features. If not set, will default to one-hot vectors per node.')
    parser.add_argument('--edge-list', nargs='?', default='data/aifb/edges.txt')
    parser.add_argument('--node-list', nargs='?', default='data/aifb/nodes.txt')
    parser.add_argument('--min-subwalk-length', type=int, default=3, help='Minimum subwalk length')
    parser.add_argument('--max-subwalk-length', type=int, default=5, help='Maximum subwalk length')
    parser.add_argument('--output', nargs='?', default="emb/aifb/stackmeta", help='Output file for neighbors and pathdata. ')
    return parser.parse_args()

def main(args):
    print("Loading walks and node list...")
    walks = load_walks(args.input_walks)
    node_dict = load_dict(args.node_list)
    if args.node_features is None or args.node_features == "None":
        #Make zero vectors. 
        print("---WARNING!!!! You have specified no features for nodes!-------")
        node_features = np.zeros((len(node_dict), 1), dtype=np.float64)
    else:
        node_features = load_features(args.node_features)
    pd = get_path_data(walks, node_dict, min_subwalk_length=args.min_subwalk_length, max_subwalk_length=args.max_subwalk_length, node_features = node_features)
    min_samples = 10
    pd = dict(filter(lambda data: data[1]["count"] >= (min_samples ** ((data[1]["length"] - 1) / 2)), pd.items()))
    compute_ids(pd)
    print("Making triples from walks...")
    path_triples = make_triples_from_walks(path_data = pd, walks = walks, node_dict = node_dict, min_subwalk_length=args.min_subwalk_length, max_subwalk_length=args.max_subwalk_length)
    print("Making neighbor feats...")
    neighbor_feats = make_neighbors_from_triples(triples=path_triples, path_data=pd, node_dict=node_dict, node_features=node_features)
    # Pickle the neighbor feats to a file. 
    final = {}
    final['path_data'] = pd
    final['neighbor_feats'] = neighbor_feats
    final['edge_dict'] = load_dict(args.edge_list)
    save_cpickle(args.output, final)

if __name__ == "__main__":
    args = parse_args()

    main(args)

