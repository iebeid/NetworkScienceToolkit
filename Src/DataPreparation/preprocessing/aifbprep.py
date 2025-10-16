import csv
from random import shuffle

def main():
    DATA_PATH = "./data/aifb/"
    employesrel = "http://swrc.ontoware.org/ontology#employs"
    affiliatesrel = "http://swrc.ontoware.org/ontology#affiliation"
    pairs = set()
    print("Prepping...")
    with open(DATA_PATH + "relations.txt", newline='') as relf:
        reader = csv.reader(relf, delimiter='\t', quotechar='|')
        for row in reader:    
            head = row[0]
            rel = row[1]
            tail = row[2]
            if rel == employesrel: 
                pairs.add((tail, head))
            if rel == affiliatesrel:
                pairs.add((head, tail))
    plist = list(pairs)
    shuffle(plist)
    print(len(plist))
    with open(DATA_PATH + "labels.txt", 'w', newline='') as labelf:
        outwriter = csv.writer(labelf, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for pair in pairs:
            outwriter.writerow(pair) 

    split_idx = int((len(plist) * 0.8))
    train = plist[:split_idx]
    test = plist[split_idx:]
    print(len(train))
    print(len(test))
    with open(DATA_PATH + "train_class.csv", 'w', newline = '') as trainf:
        outwriter = csv.writer(trainf, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for p in train:
            outwriter.writerow(p)
    with open(DATA_PATH + "test_class.csv", 'w', newline = '') as testf:
        outwriter = csv.writer(testf, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for p in test:
            outwriter.writerow(p)
main()