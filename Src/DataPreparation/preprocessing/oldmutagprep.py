import csv
import numpy as np
from random import shuffle
from models.data_utils import *

def main():
    DATA_PATH = "./data/mutag/"
    # Here's how this is going to look: We're trying to find mutanogenic compounds. Each compound is a graph of molecules. 
    # So, we give each compound a node. It connects to a node for every atom in it. Atoms connect to each other via. edges
    # Atom-Atom links are indexed 0, 1, 2, or 3. The atom-graph edges will be 4. 
    compound_nodes = set()
    atom_nodes = set()
    atom_graph_edge_type = 4
    triples = set()
    with open(DATA_PATH + "Mutag.graph_idx") as node2graphf:
        for idx, line in enumerate(node2graphf):
            atom_nodes.add(idx)
            compound_nodes.add(int(line))
            triples.add((idx, atom_graph_edge_type, int(line)))
    # Because they're in the same graph, update the compound indicies by adding the number of atoms to them. 
    triples = set(map(lambda trip: (trip[0], trip[1], trip[2] + len(atom_nodes)), triples))
    with open(DATA_PATH + "Mutag.edges") as edgef:
        with open(DATA_PATH + "Mutag.link_labels") as edge_labelf:
            for idx, lines in enumerate(zip(edgef, edge_labelf)):
                edgestr, labelstr = lines
                label = int(labelstr)
                head = edgestr.split(",")[0]
                tail = edgestr.split(",")[1]
                triples.add((int(head), label, int(tail)))

    # Serialize the triples to the node/edges file format I use.
    with open(DATA_PATH + "relations.txt", 'w', newline='') as relf:
        outwriter = csv.writer(relf, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for triple in triples:
            outwriter.writerow(triple)
    with open(DATA_PATH + "edges.txt", 'w', newline='') as relf:
        outwriter = csv.writer(relf, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(5): # 5 types of edges
            outwriter.writerow([str(i)])
    with open(DATA_PATH + "nodes.txt", 'w', newline='') as relf:
        outwriter = csv.writer(relf, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        num_nodes = len(compound_nodes) + len(atom_nodes)
        for i in range(num_nodes):
            outwriter.writerow([str(i)])
    #Finally, make a node attributes file. 
    attributes = []
    with open(DATA_PATH + "Mutag.node_labels") as nlf:
        for idx, attr in enumerate(nlf):
            attributes.append(np.empty([1]))
            attributes[idx][0] = int(attr.split(",")[1])
    save_features(DATA_PATH + "nfeatures.txt", attributes)    
    pairs = set()
    with open(DATA_PATH + "Mutag.graph_labels", newline='') as graphlabelf:
        with open(DATA_PATH + "labels.txt", 'w', newline='') as labelf:
            outwriter = csv.writer(labelf, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for idx, isMutStr in enumerate(graphlabelf):
                pairs.add((idx + len(atom_nodes), int(isMutStr)))
                outwriter.writerow([idx + len(atom_nodes), int(isMutStr)]) 
    #Finally, split it into train and test sets
    plist = list(pairs)
    shuffle(plist)
    splitidx = int(len(plist) * 0.9)
    train = plist[:splitidx]
    test = plist[splitidx:]
    with open(DATA_PATH + "train_class.txt", 'w') as trainf:
        outwriter = csv.writer(trainf, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for t in train:
            outwriter.writerow(t)
    with open(DATA_PATH + "test_class.txt", 'w') as testf:
        outwriter = csv.writer(testf, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for t in test:
            outwriter.writerow(t)
    

main()