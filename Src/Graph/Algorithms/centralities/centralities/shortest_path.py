import argparse
from graph_utils.graph import construct_graph
from graph_utils.utils import Utils
from graph_utils.metrics import Metrics

def parse_args():
    
    ''' Parse command line arguments '''
    
    parser = argparse.ArgumentParser(description='Compute shortest path')
    parser.add_argument('--file', '-f', help='file containing data', default='../data/c2b2rdf/chem2bio2rdf.txt')
    parser.add_argument('--node2class', help='file containing node to type information', default='../data/c2b2rdf/node2class.txt')
    parser.add_argument('--edge2class', help='file containing edge to type information', default='../data/c2b2rdf/edge2class.txt')
    parser.add_argument('--source', help='source link')
    parser.add_argument('--target', help='target link')
    args = parser.parse_args()
    return args

def main(args):

    utils = Utils()
    node2class = utils.text_to_dict(args.node2class)
    edge2class = utils.text_to_dict(args.edge2class)

    graph = construct_graph(args.file, node2class, edge2class)

    metrics = Metrics()
    path = metrics.shortest_path(graph, src=args.source, target=args.target)
    print(f'Path: {path}')
    print(f'Distance = {len(path)}')

if __name__ == '__main__':
    args = parse_args()
    main(args)