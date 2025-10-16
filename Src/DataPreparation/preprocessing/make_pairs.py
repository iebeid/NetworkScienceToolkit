import os
import csv
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="generates compound and target pairs")
    parser.add_argument('--entity1', nargs='?', help='file containing ids of entity1')
    parser.add_argument('--entity1_url', nargs='?', help='entity1 chem2bio2rdf url', default='http://chem2bio2rdf.org/pubchem/resource/pubchem_compound/')
    parser.add_argument('--entity2', nargs='?', help='file containing ids of entity2')
    parser.add_argument('--entity2_url', nargs='?', help='entity2 chem2bio2rdf url', default='http://chem2bio2rdf.org/uniprot/resource/gene/')
    parser.add_argument('--nodes', nargs='?', default='data/c2b2rdf/nodes.txt', help='file containing all nodes in graph')
    parser.add_argument('--output', nargs='?', default='/home/ubuntu/jack/KnowledgeGraphAlgos/data/ctp/compoundgenerelations.csv', help='output path')
    return parser.parse_args()


def read_file(filepath):

    result = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            result.append(line)

    return result

def main(args):


    entity1 = read_file(args.entity1)
    entity2 = read_file(args.entity2)
    nodes = read_file(args.nodes)


    pairs = []
    not_found_entity1 = []
    not_found_entity2 = []
    for node1 in entity1:
        node1_with_url = os.path.join(args.entity1_url, node1)
        if node1_with_url not in nodes:
            not_found_entity1.append(node1)
        else:
            for node2 in entity2:
                node2_with_url = os.path.join(args.entity2_url, node2)
                if node2_with_url not in nodes:
                    not_found_entity2.append(node2)
                else:
                    pair = (node1_with_url, node2_with_url, '1')
                    pairs.append(pair)



    print(f'entity1 {not_found_entity1} not in Graph')
    print(f'entity2 {not_found_entity2} not in Graph')

    df = pd.DataFrame(pairs)
    df.to_csv(args.output, index=False, sep='\t', header=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
