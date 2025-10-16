import argparse
import networkx as nx
import random
import numpy as np   
import math
from walks import *
from multiprocessing import Pool, cpu_count, current_process
import functools, itertools


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run edge transition matrix.") 

    parser.add_argument('--input', nargs='?', default='data/bgs/train_pos.csv',
                        help='Input graph path')
    parser.add_argument('--input-edge-list', nargs='?', default='data/bgs/edges.txt',
                        help='Input edge list') 
    parser.add_argument('--input-node-list', nargs='?', default='data/bgs/nodes.txt',
                        help='Input edge list') 
    parser.add_argument('--output', nargs='?', default='emb/bgs/walks.txt',
                        help='Walks file save path') 
    parser.add_argument('--matrix', nargs='?', default='default', 
                        help='Transition matrix path') 
    parser.add_argument('--walk-length', type=int, default=50,
                        help='Length of walk per source. Default is 50. Counts edges AND nodes. ')

    parser.add_argument('--num-walks', type=int, default=4,
                        help='Number of walks per source. Default is 4.')
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--walk-type', nargs='?', default='edge2vec', help='What type of walk method to use. Provides edge2vec, node2vec. ')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='weighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is directed. Default is directed.')
    parser.add_argument('--undirected', dest='directed', action='store_false')
    parser.set_defaults(directed=True) # Note to future Jack: Inverse relationships actually decrease performance. Weird, right? I think it makes the data sparser/too many/double labels to deal with. 
    return parser.parse_args()

def start_process():
    print('Starting %s' % current_process().name)

def simulate_walks_worker(tup, G_loc, walk_length, is_directed, matrix, p, q, num_edge_types, bsize, walk_type):
    # print "chosen node id: ",nodes
    nodes = tup[0]
    count = tup[1]
    G = read_graph(G_loc,num_edge_types, False, is_directed) 
    print(("doing iter %d" % (count)))
    walks = []
    for node in nodes:
        if walk_type == "edge2vec":
            walk = edge2vec_walk(G, walk_length, [node], is_directed, matrix, p, q)
        elif walk_type == "node2vec":
            walk = node2vec_walk(G, walk_length, [node], is_directed, p, q)
        else:
            walk = []
            print("Unknown walk type %s!" % walk_type)
        if(len(walk) < walk_length): #Happens if there's a dead end. 
            print("Dead end found at node %s" % str(node))
            walk = []
        walks.append(walk)
    return walks
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)
def simulate_walks(G_loc, num_walks, walk_length,is_directed, matrix, p,q, num_edge_types, num_nodes, walk_type):
    '''
    generate random walk paths constrainted by transition matrix
    '''
    walks = []
    nodes = [str(i) for i in range(num_nodes)] 
    num_cores = int(cpu_count() - 1) # Leave a core open so my ssh session doesn't get booted lol
    print("Making pool with %d cpus..." % num_cores)
    pool = Pool(processes=num_cores, initializer=start_process)
    print("Nodes: %d, num_walks %d for a total number of %d walks" % (len(nodes), num_walks, len(nodes) * num_walks))
    print("Graph location: %s" % G_loc)
    bsize = math.ceil(num_nodes / num_cores)
    partial_func = functools.partial(simulate_walks_worker, G_loc=G_loc, walk_length=walk_length, is_directed=is_directed, matrix=matrix, p=p, q=q,num_edge_types=num_edge_types,bsize=bsize,walk_type=walk_type)
    print("NUmber of iterations %d of batch size %d" % (int(len(nodes) * num_walks / bsize), bsize))
    counted_nodes = itertools.islice(itertools.cycle(zip(grouper(nodes, bsize, fillvalue="0") , range(len(nodes)))), int(len(nodes) * num_walks / bsize))
    print("Made node iterator! Calling map...")

    walks_nested = pool.map(partial_func, counted_nodes)
    walks = [item for sublist in walks_nested for item in sublist]
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks
    return walks

def main(args):
    print("------begin to read transition matrix--------")
    edges = load_dict(args.input_edge_list)
    matrix_size = len(edges)
    walk_type = args.walk_type
    if args.directed:
        matrix_size = 2 * matrix_size
    if matrix_size > 10e4:
        # THat matrix can't fix in memory. 
        print("Too many edge types! Falling back to node2vec...")
        walk_type = "node2vec"
    if walk_type == "edge2vec":
        if args.matrix == "default":
            trans_matrix = np.ones((matrix_size, matrix_size)) * 1/(len(edges) * len(edges))
        else:
            trans_matrix = read_edge_type_matrix(args.matrix)
        print(trans_matrix)
        print(matrix_size)
    else:
        trans_matrix = None
    num_edges = len(edges)
    print(num_edges)
    num_nodes = len(load_dict(args.input_node_list))
    weighted = args.weighted
    directed = args.directed
    print("------begin to simulate walk---------") 
    print("Walk type: %s" % walk_type)
    
    walks = simulate_walks(args.input,args.num_walks, args.walk_length,args.directed, trans_matrix, args.p,args.q, num_edges, num_nodes, walk_type)
    print("------saving %d walks--------" % len(walks))
    save_walks(walks, args.output)

if __name__ == "__main__":
    args = parse_args()

    main(args)   
