import csv
from random import shuffle
def load_dict(file):
    d = dict()
    
    with open(file) as nodef:
        for idx, line in enumerate(nodef):
            d[line.strip()] = str(idx)
    print("found %d values in dict file" % len(d))
    return d
def main():
    DATA_PATH = "./data/c2b2rdf/"
    GENE_URI = "http://chem2bio2rdf.org/uniprot/resource/gene/"
    prawtriples = []
    train_test_split = 0.9

    with open(DATA_PATH + 'gene-gene/full.txt', newline='') as posfile:
        reader = csv.reader(posfile, delimiter='\t', quotechar='|')
        for row in reader:    
            headg = GENE_URI + row[0]
            tailgs = row[1].split(" ")
            rel = "1"
            for tailg in tailgs:
                prawtriples.append((headg, rel, GENE_URI + tailg))
    node_dict = load_dict(DATA_PATH + "nodes.txt")
    edge_dict = load_dict(DATA_PATH + "edges.txt")
    ptriples = []
    for t in prawtriples:
        print(t)
        if t[0] in node_dict and t[2] in node_dict:
            ptriples.append((node_dict[t[0]], t[1], node_dict[t[2]]))
    num_trainp = int(len(ptriples) * train_test_split)
    shuffle(ptriples)
    train_p = ptriples[0:num_trainp]
    test_p = ptriples[num_trainp:len(ptriples)]
    with open(DATA_PATH + "gene-gene/gg_train_positives.csv", 'w', newline='') as nfile:
        outwriter = csv.writer(nfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for ntriple in train_p:
            outwriter.writerow([ntriple[0], ntriple[2], 1])
    with open(DATA_PATH + "gene-gene/gg_test_positives.csv", 'w', newline='') as nfile:
        outwriter = csv.writer(nfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for ntriple in test_p:
            outwriter.writerow([ntriple[0], ntriple[2], 1])

main()