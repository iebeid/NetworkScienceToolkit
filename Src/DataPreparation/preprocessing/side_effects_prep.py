import os
import csv
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="generates compound and target pairs")
    parser.add_argument('--file', nargs='?', help='filepath to side effects file')
    parser.add_argument('--nodes', nargs='?', default='data/c2b2rdf/nodes.txt', help='file containing all nodes in graph')
    parser.add_argument('--output', nargs='?', default='/home/ubuntu/jack/KnowledgeGraphAlgos/data/ctp/compoundgenerelations.csv', help='output path')
    return parser.parse_args()

def read_nodes(filepath):

    result = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            result.append(line)

    return result

def read_side_effects(filepath):

    result = pd.read_csv(filepath, delimiter='\t', header=None)

    return np.array(result)

def main(args):

    data = read_side_effects(args.file)
    nodes = read_nodes(args.nodes)

    frequncy_needed = ['common', 'very common', 'frequent', 'postmarketing']
    URL = 'http://chem2bio2rdf.org/sider/resource/sider/'
    # print(data)
    processed_data = []
    ids = []
    for row in data:

        id = row[2].strip()
        frequency = row[4]
        check_pt = row[7]
        name = row[9].strip()
        if frequency in frequncy_needed:
            if check_pt == 'PT' and os.path.join(URL, id) in nodes:
                if id not in ids:
                    ids.append([id])
                    processed_data.append([id, name, frequency])

    df = pd.DataFrame(processed_data)
    id_df = pd.DataFrame(ids)

    df.to_csv(args.output, index=False, sep='\t', header=False)

    filename = os.path.join(os.path.dirname(args.output), 'side_effect_ids.csv')
    id_df.to_csv(filename, index=False, header=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)
