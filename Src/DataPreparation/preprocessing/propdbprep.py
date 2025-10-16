import csv
from random import shuffle
def load_line_dict(file):
    d = dict()
    
    with open(file) as nodef:
        for idx, line in enumerate(nodef):
            d[line.strip()] = str(idx)
    print("found %d values in dict file" % len(d))
    return d
def load_dict(file):
    d = dict()
    
    with open(file) as nodef:
        for idx, line in enumerate(nodef):
            vals = line.split("\t")
            d[vals[0].strip()] = vals[1].strip()
    print("found %d values in dict file" % len(d))
    return d
def main():
    DATA_PATH = "./data/c2b2rdf/"
    SUB_PATH = "drugbank/"
    GENE_URI = "http://chem2bio2rdf.org/uniprot/resource/gene/"
    PUBCHEM_URI = "http://chem2bio2rdf.org/pubchem/resource/pubchem_compound/"
    DB_PREFIX = "https://www.drugbank.ca/drugs/"
    triples = [[], [], [], []] #ptrain, ntrain, ptest, ntest
    node_dict = load_line_dict(DATA_PATH + "nodes.txt")
    edge_dict = load_line_dict(DATA_PATH + "edges.txt")
    db_pubchem_map = load_dict(DATA_PATH + SUB_PATH + "db-to-pubchem.csv")
    with open(DATA_PATH + SUB_PATH + "prot-db.txt", newline='') as posfile:
        reader = csv.reader(posfile, delimiter='\t', quotechar='|')
        for row in reader:    
            heads = map(lambda h: GENE_URI + h, row[2].split(" "))
            in_test = row[7] == "0"
            neg = row[8] == "0"
            idx = int(neg) + 2 * int(in_test)
            tails = filter(lambda t: t != '', row[12:])
            tails = map(lambda t: t[len(DB_PREFIX):].strip(), tails)
            tails = filter(lambda t: t in db_pubchem_map, tails)
            tails = map(lambda t: PUBCHEM_URI + db_pubchem_map[t], tails)
            for h in heads:
                for t in tails:
                    if t in node_dict and h in node_dict:
                        triples[idx].append((h, "1", t))
    filenames = ["train_pos", "train_neg", "test_pos", "test_neg"]
    for fn, trips in zip(filenames, triples):
        with open(DATA_PATH + SUB_PATH + fn + ".csv", 'w', newline='') as nfile:
            outwriter = csv.writer(nfile, delimiter='\t',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for trip in trips:
                outwriter.writerow([node_dict[trip[0]], node_dict[trip[2]], trip[1]])
    with open(DATA_PATH + SUB_PATH + "compoundgenerelations.csv", 'w', newline='') as nfile:
        outwriter = csv.writer(nfile, delimiter='\t',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        trips = triples[0] + triples[1] + triples[2] + triples[3]
        for trip in trips:
            outwriter.writerow([trip[0], trip[2], trip[1]])


main()