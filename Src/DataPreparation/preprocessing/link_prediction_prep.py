import os
import csv
import sys

import numpy as np
import math
import random
import argparse

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="script to run tests")

    parser.add_argument('--file1', nargs='?', help='first file')
    parser.add_argument('--file2',  nargs='?', help='second file')
    parser.add_argument('--c2b2rdf-relations',  nargs='?', help='chem2bio2rdf file')
    parser.add_argument('--links-output', nargs='?', help='links output file path')
    parser.add_argument('--nodes', nargs='?', help='nodes file path')
    parser.add_argument('--edges', nargs='?', help='nodes file path')

    return parser.parse_args()


def read_map(file):
    d = {}
    with open(file, 'r') as f:
        [d.update({line.strip(): str(idx)}) for idx, line in enumerate(f)]

    return d

def check_if_only_one_edge(node, node_counts):
    if node_counts[node] == 1:
        return True
    else:
        node_counts[node] -= 1
        return False

def main(args):

    file1_links = []
    file2_links = []
    with open(args.file1, newline='') as file1:
        reader1 = csv.reader(file1, delimiter='\t', quotechar='|')

        with open(args.file2, newline='') as file2:
            reader2 = csv.reader(file2, delimiter='\t', quotechar='|')

            for row1 in reader1:
                if row1[2] == '1':
                    file1_links.append(row1)

            for row2 in reader2:
                if row2[2] == '1':
                    file2_links.append(row2)

            similar_links = [link1 for link1 in file1_links for link2 in file2_links if link1 == link2]

        # remove duplicates
        similar_links = list(map(list, (set(map(tuple, similar_links)))))
        print(f'similar links count: {len(similar_links)}')

        # combine the two lists to get all links to remove from c2b2rdf
        links_to_remove = np.concatenate((file1_links, file2_links), axis=0)
        # remove duplicates
        links_to_remove = list(map(list, set(map(tuple, links_to_remove))))
        print(f'Links to remove from c2b2rdf length: {len(links_to_remove)}')


    with open(args.file1, newline='') as file1:
        reader1 = csv.reader(file1, delimiter='\t', quotechar='|')

        filename = os.path.basename(args.file1.replace('.csv', '_train.csv'))
        filename = os.path.join(os.path.dirname(args.links_output), filename)
        with open(filename, 'w', newline='') as train_output_file:
            train_link_writer = csv.writer(train_output_file, delimiter='\t', quotechar='|')
            count = 0
            for row in reader1:
                if row not in similar_links:
                    train_link_writer.writerow(row)


    node_map = read_map(args.nodes)
    edge_map = read_map(args.edges)
    relations = []
    node_counts = {}
    with open(args.c2b2rdf_relations, newline='') as c2b2rdf_file:
        reader = csv.reader(c2b2rdf_file, delimiter='\t', quotechar='|')
        for idx, row in enumerate(reader):
            node1 = row[0].strip()
            node2 = row[2].strip()
            edge = row[1].strip()

            if node1 not in node_counts:
                node_counts[node1] = 1
            else:
                node_counts[node1] += 1
            if node2 not in node_counts:
                node_counts[node2] = 1
            else:
                node_counts[node2] += 1

            relations.append([node1, edge, node2, idx])

    relations = np.array(relations)
    shuffle = np.random.choice(len(relations), len(relations), replace=False)
    relations = relations[shuffle]

    with open(args.links_output, 'w', newline='') as links_output_file:
        link_writer = csv.writer(links_output_file, delimiter='\t', quotechar='|')
        for row in relations:
            node1 = row[0]
            node2 = row[2]
            edge = row[1]
            idx = row[3]
            pair = [node2, node1, '1']
            check = check_if_only_one_edge(node1, node_counts) or check_if_only_one_edge(node2, node_counts)

            if (pair in links_to_remove and check) or (pair not in links_to_remove):
                if node2 == 'http://chem2bio2rdf.org/uniprot/resource/gene/AAT2':
                    print(node2.split(' '))
                node2 = node2.split(' ')
                if len(node2) > 1:
                    node2 = '_'.join(node2)
                else:
                    node2 = node2[0]
                rel_id = edge_map[edge]
                node1_id = node_map[node1]
                node2_id = node_map[node2]

                relation = [node1_id, node2_id, rel_id, idx]
                link_writer.writerow(relation)



if __name__ == '__main__':
    args = parse_args()
    main(args)
