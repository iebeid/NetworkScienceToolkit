import os
import csv
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="generates compound and target pairs")
    parser.add_argument('--compounds', nargs='?', help='file containing compound ids')
    parser.add_argument('--targets', nargs='?', help='file containing targets')
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
    COMPOUND_URL = 'http://chem2bio2rdf.org/pubchem/resource/pubchem_compound/'
    GENE_URL = 'http://chem2bio2rdf.org/uniprot/resource/gene/'

    compounds = read_file(args.compounds)
    genes = read_file(args.targets)
    nodes = read_file(args.nodes)


    pairs = []
    for compound in compounds:
        compound = os.path.join(COMPOUND_URL, compound)
        for gene in genes:
            gene = os.path.join(GENE_URL, gene)
            pair = (compound, gene, '1')
            pairs.append(pair)

    df = pd.DataFrame(pairs)
    df.to_csv(args.output, index=False, sep='\t', header=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
