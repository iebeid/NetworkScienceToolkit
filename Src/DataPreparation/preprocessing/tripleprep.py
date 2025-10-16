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
    parser = argparse.ArgumentParser(description="Run edge transition matrix.")

    parser.add_argument('--data', nargs='?', default='c2b2rdf',
                        help='Name of data')
    parser.add_argument('--train_test_split', nargs='?', default=0.9, type=float)
    parser.add_argument('--savepath', nargs='?', default='data/c2b2rdf/')
    parser.add_argument('--reversed', dest='reversed', action='store_true',
                        help='Boolean specifying if the file is "head, tail, edge" instead of "head, edge, tail". Defatuls to false. ')
    parser.set_defaults(reversed=False)
    return parser.parse_args()

def main(args):
    rel_to_index = {}
    rel_count = 0
    node_to_index = {}
    node_count = 0
    node_to_relations = dict()
    train_test_split = args.train_test_split
    savepath = args.savepath
    do_inverses = False
    total_idx = 0

    csv.field_size_limit(sys.maxsize)

    DATA_PATH = "./data/" + args.data + "/"
    relation_counts = dict()
    bad_rows = 0
    with open( DATA_PATH + 'relations.txt', newline='') as infile:
        reader = csv.reader(infile, delimiter='\t', quotechar='|')
        for row in reader:
            if len(row) != 3:
                bad_rows += 1
                continue
            n1 = row[0].replace(" ", "_")
            if args.reversed:
                rel = row[2]
                n2 = row[1].replace(" ", "_")
            else:
                rel = row[1]
                n2 = row[2].replace(" ", "_")
            rel_inv = rel + "_INV"
            if n1 not in node_to_index:
                node_to_index[n1] = node_count
                node_count += 1
            if n2 not in node_to_index:
                node_to_index[n2] = node_count
                node_count += 1
            if rel not in rel_to_index:
                rel_to_index[rel] = rel_count
                rel_count += 1
            rel_inv_id = -1
            if do_inverses:
                if rel_inv not in rel_to_index:
                    rel_to_index[rel_inv] = rel_count
                    rel_count += 1
                rel_inv_id = rel_to_index[rel_inv]
            n1_id = node_to_index[n1]
            n2_id = node_to_index[n2]
            rel_id = rel_to_index[rel]
            relation = (n1_id, n2_id, rel_id, rel_inv_id, total_idx, total_idx + 1)
            if n1_id in node_to_relations:
                node_to_relations[n1_id].append(relation)
            else:
                node_to_relations[n1_id] = [relation]
            if n2_id in node_to_relations:
                node_to_relations[n2_id].append(relation)
            else:
                node_to_relations[n2_id] = [relation]
            relation_counts[relation] = 2
            total_idx += 2
    print(bad_rows)
    total_relation_set = set(relation_counts.keys())
    num_train_relations = math.floor(train_test_split * len(total_relation_set))
    print("Going for %d train out of %d total and %d relation mappings" % (num_train_relations, total_idx, len(total_relation_set)))
    train_relations = []
    # Put a relation with one of each node into the training set.
    for node, nrelations in node_to_relations.items():
        next_rel = random.choice(nrelations)
        if relation_counts[next_rel] is 2:
            # We haven't yet added this relation to the training set.
            train_relations.append(next_rel)
        if next_rel in total_relation_set:
            total_relation_set.remove(next_rel)
        relation_counts[next_rel] -= 1
        if relation_counts[next_rel] is 0:
            relation_counts.pop(next_rel)
    relations = list(total_relation_set)
    random.shuffle(relations)
    num_remaining_train = num_train_relations - len(train_relations)
    train_relations = train_relations + relations[:num_remaining_train]
    test_relations = relations[num_remaining_train:]
    print("Num test and train: %d and %d" % (len(test_relations), len(train_relations)))
    node_ids = list(node_to_index.values())
    with open(savepath + 'train_pos.csv', 'w', newline='') as outfile:
        with open(savepath + 'train_neg.csv', 'w', newline='') as noutfile:
            outwriter = csv.writer(outfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            noutwriter = csv.writer(noutfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # Make sure no nodes with only a single connection are left out of the training file, or when we test we'll be like "what is this node" b/c it wasn't in any walks.
            for  n1, n2, rel, rel_inv, idx1, idx2 in train_relations:
                outwriter.writerow([n1, n2, rel, idx1])
                if do_inverses:
                    outwriter.writerow([n2, n1, rel_inv, idx2])
                # Corrupt the head, and then the tail, to create negtive examples.
                corrupted_n1 = random.choice(node_ids)
                if (corrupted_n1, n2, rel, rel_inv, idx1, idx2) not in total_relation_set:
                    noutwriter.writerow([corrupted_n1, n2, rel, idx1])
                    if do_inverses:
                        noutwriter.writerow([n2, corrupted_n1, rel_inv, idx2])
                corrupted_n2 = random.choice(node_ids)
                if (n1, corrupted_n2, rel, rel_inv, idx1, idx2) not in total_relation_set:
                    noutwriter.writerow([n1, corrupted_n2, rel, idx1])
                    if do_inverses:
                        noutwriter.writerow([corrupted_n2, n1, rel_inv, idx2])
    with open(savepath + "test_pos.csv", 'w', newline='') as outfile:
        with open(savepath + 'test_neg.csv', 'w', newline='') as noutfile:
            negwriter = csv.writer(noutfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            outwriter = csv.writer(outfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for n1, n2, rel, rel_inv, idx1, idx2 in test_relations:
                outwriter.writerow([n1, n2, rel, idx1])
                if do_inverses:
                    outwriter.writerow([n2, n1, rel_inv, idx2])
                # Corrupt the head, and then the tail, to create negtive examples.
                corrupted_n1 = random.choice(node_ids)
                if (corrupted_n1, n2, rel, rel_inv, idx1, idx2) not in total_relation_set:
                    negwriter.writerow([corrupted_n1, n2, rel, idx1])
                    if do_inverses:
                        negwriter.writerow([n2, corrupted_n1, rel_inv, idx2])
                corrupted_n2 = random.choice(node_ids)
                if (n1, corrupted_n2, rel, rel_inv, idx1, idx2) not in total_relation_set:
                    negwriter.writerow([n1, corrupted_n2, rel, idx1])
                    if do_inverses:
                        negwriter.writerow([corrupted_n2, n1, rel_inv, idx2])


    with open(savepath + "edges.txt", "w") as edgef:
        for rel in sorted(rel_to_index, key=rel_to_index.get):
            if not rel.endswith("_INV") or do_inverses:
                edgef.write(rel + "\n")

    with open(savepath + "nodes.txt", "w") as nodef:
        for node in sorted(node_to_index, key=node_to_index.get):
            nodef.write(node + "\n")
if __name__ == "__main__":
    args = parse_args()
    main(args)
