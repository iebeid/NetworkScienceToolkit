import csv
from random import shuffle
from evaluation.eval_utils import *

def main():
    DATA_PATH = "./data/c2b2rdf/"
    GENE_URI = "http://chem2bio2rdf.org/uniprot/resource/gene/"
    COMPOUND_URI = "http://chem2bio2rdf.org/pubchem/resource/pubchem_compound/"
    prawtriples = []
    nrawtriples = []
    train_test_split = 0.9
    with open(DATA_PATH + "compound-gene/compoundgenerelations.txt", 'w', newline='') as cgrfile:
        outwriter = csv.writer(cgrfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        with open(DATA_PATH + 'compound-gene/internal_testset_label/positive.txt', newline='') as posfile:
            reader = csv.reader(posfile, delimiter='\t', quotechar='|')
            for row in reader:
                compound_id = COMPOUND_URI + row[0]
                gene = GENE_URI + row[1]
                rel = "1"
                prawtriples.append((gene, rel, compound_id))
                outwriter.writerow([gene, compound_id, rel])
        with open(DATA_PATH + 'compound-gene/internal_testset_label/negative.txt', newline='') as negfile:
            reader = csv.reader(negfile, delimiter='\t', quotechar='|')
            for row in reader:
                compound_id = COMPOUND_URI + row[0]
                gene = GENE_URI + row[1]
                rel = "0"
                prawtriples.append((gene, rel, compound_id))
                outwriter.writerow([gene, compound_id, rel])
    # node_dict = load_dict(DATA_PATH + "nodes.txt")
    # edge_dict = load_dict(DATA_PATH + "edges.txt")
    # ptriples = [(node_dict[t[0]], t[1], node_dict[t[2]]) for t in prawtriples]
    # ntriples = [(node_dict[t[0]], t[1], node_dict[t[2]]) for t in prawtriples]
    # num_trainp = int(len(ptriples) * train_test_split)
    # num_trainn = int(len(ntriples) * train_test_split)
    # shuffle(ptriples)
    # shuffle(ntriples)
    # train_p = ptriples[0:num_trainp]
    # train_n = ntriples[0:num_trainn]
    # test_p = ptriples[num_trainp:len(ptriples)]
    # test_n = ptriples[num_trainn:len(ntriples)]
    # with open(DATA_PATH + "cg_train_negatives.csv", 'w', newline='') as nfile:
    #     outwriter = csv.writer(nfile, delimiter='\t',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for ntriple in train_n:
    #         outwriter.writerow([ntriple[0], ntriple[2], 1])
    # with open(DATA_PATH + "cg_train_positives.csv", 'w', newline='') as nfile:
    #     outwriter = csv.writer(nfile, delimiter='\t',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for ntriple in train_p:
    #         outwriter.writerow([ntriple[0], ntriple[2], 1])
    # with open(DATA_PATH + "cg_test_positives.csv", 'w', newline='') as nfile:
    #     outwriter = csv.writer(nfile, delimiter='\t',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for ntriple in test_p:
    #         outwriter.writerow([ntriple[0], ntriple[2], 1])
    # with open(DATA_PATH + "cg_test_negatives.csv", 'w', newline='') as nfile:
    #     outwriter = csv.writer(nfile, delimiter='\t',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for ntriple in test_n:
    #         outwriter.writerow([ntriple[0], ntriple[2], 1])

main()
