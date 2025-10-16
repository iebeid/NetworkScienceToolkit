from rdflib import Graph, URIRef, Literal
import rdflib as rdf
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import networkx as nx
import csv
import argparse

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Process an ntriples rdf file. ") 

    parser.add_argument('--data', nargs='?', default='./tmp/bgs_stripped.nt',
                        help='Location of rdf nt file')
    parser.add_argument('--output', nargs='?', default='./data/bgs/relations.txt',
                        help='Output file')
    return parser.parse_args()

def main(args):
    g = Graph()
    print("Reading graph...")
    g.parse(args.data, format="nt")
    print("Transforming to networkx...")
    nxg = rdflib_to_networkx_multidigraph(g, edge_attrs=(lambda s, p, o: {"type": p}))
    print("Saving to %s" % args.output)
    nx.write_edgelist(nxg, args.output, delimiter='\t', data=["type"])

if __name__ == "__main__":
    args = parse_args()
    main(args)
