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
    genes = set()
    drugs = set()
    pairs = set()
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
                    if t in node_dict:
                        drugs.add(t)
                    if h in node_dict:
                        genes.add(h)
                    pairs.add((h, t))
    # Now make pairs for each drug-gene relation that isn't in the original list. 
    with open(DATA_PATH + SUB_PATH + "new-compoundgenerelations.csv", 'w', newline='') as nfile:
        outwriter = csv.writer(nfile, delimiter='\t',
            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for drug in drugs:
            for gene in genes:
                pair = (gene, drug)
                if pair not in pairs:
                    outwriter.writerow([gene, drug, "1"])

main()