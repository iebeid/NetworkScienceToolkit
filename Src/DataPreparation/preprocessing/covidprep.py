import csv
from random import shuffle
from eval_utils import * 

def main():
    DATA_PATH = "./data/c2b2rdf/"
    GENE_URI = "http://chem2bio2rdf.org/uniprot/resource/gene/"
    COMPOUND_URI = "http://chem2bio2rdf.org/pubchem/resource/pubchem_compound/"
    prawtriples = []
    nrawtriples = []
    with open(DATA_PATH + "covid-compoundgenerelations.txt", 'w', newline='') as cgrfile:
        outwriter = csv.writer(cgrfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        with open(DATA_PATH + 'internal_testset_label/COVID_DATASET.csv', newline='') as posfile:
            reader = csv.reader(posfile, delimiter='\t', quotechar='|')
            for row in reader:    
                print(row)
                compound_id = COMPOUND_URI + row[1]
                gene = GENE_URI + row[0]
                rel = "1"
                prawtriples.append((gene, rel, compound_id))
                outwriter.writerow([gene, compound_id, rel])
    node_dict = load_dict(DATA_PATH + "nodes.txt")
    edge_dict = load_dict(DATA_PATH + "edges.txt")
    trips = []
    for t in prawtriples:
        if t[0] in node_dict and t[2] in node_dict:
            trips.append((node_dict[t[0]], t[1], node_dict[t[2]]))
    """
    with open(DATA_PATH + "cg_train_negatives.csv", 'w', newline='') as nfile:
        outwriter = csv.writer(nfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for ntriple in train_n:
            outwriter.writerow([ntriple[0], ntriple[2], 1])
    with open(DATA_PATH + "cg_train_positives.csv", 'w', newline='') as nfile:
        outwriter = csv.writer(nfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for ntriple in train_p:
            outwriter.writerow([ntriple[0], ntriple[2], 1])
    """
    with open(DATA_PATH + "covid_test_positives.csv", 'w', newline='') as nfile:
        outwriter = csv.writer(nfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for ntriple in trips:
            outwriter.writerow([ntriple[0], ntriple[2], 1])

main()