from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score,cross_val_predict
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
import numpy as np


def edgepath2str(edges, node):
    ret = ""
    for edge in edges:
        ret += (str(edge) + "->")
    return ret + str(node)

def edge2str(edge):
    return "EDGE_" + str(edge)

def nodefromrelstr(instr):
    split = instr.split("->")
    return split[len(split) - 1]

def edgesfromrelstr(instr):
    split = instr.split("->")
    return list(map(str2edge, split[:-1]))
def str2edge(instr):
    return instr[5:]

def load_dict(file):
    d = dict()
    
    with open(file) as nodef:
        for idx, line in enumerate(nodef):
            d[line.strip()] = str(idx)
    print("found %d values in dict file" % len(d))
    return d
def add_edges_from_relation_vectors(nodes, edges, relation_vectors, combine_type):
    print("Scraping all edge-node pairs")
    all_relations = dict()
    for node in range(len(nodes)):
        for edge in range(len(edges)):
            rel = edgepath2str([edge2str(edge)], node)
            if rel not in relation_vectors:
                continue
            vec = relation_vectors[rel]
            if edge in all_relations:
                all_relations[edge].append(vec)
            else:
                all_relations[edge] = [vec]
    print("NUMBER OF KEYS: %d" % len(relation_vectors.vocab.keys()))
    print("Combining each vector list for each node...")
    edges = []
    final_vecs = []
    for edge, vectors in all_relations.items():
        edges.append(edge)
        final_vecs.append(combine_relation_vectors(vectors, combine_type))
    if len(edges) == 0:
        print("!!!!WARNING!!!!!: No relation vectors were found. ")
        return
    print("Adding in %d vectors" % len(edges))
    relation_vectors.add(edges, final_vecs)


def combine_relation_vectors(vectors, combine_type):
    if combine_type is 0:
        # Sum combine. 
        agg = vectors[0]
        for v in vectors[1:]:
            agg = agg + v
        return agg
    elif combine_type is 1:
        # Mean combine. 
        agg = vectors[0]
        for v in vectors[1:]:
            agg = agg + v
        return agg / len(vectors)
        

def nodes_from_relation_vectors(relation_vectors, combine_type):
    print("Getting all node/edge pairs correlated with each node...")
    all_relations = dict()
    for rel in list(relation_vectors.vocab.keys()): 
        vec = relation_vectors[rel]
        node = nodefromrelstr(rel)
        if node in all_relations:
            all_relations[node].append(vec)
        else:
            all_relations[node] = [vec]
    # now combine each list of vectors. 
    print("Combining each vector list for each node...")
    final_vecs = []
    for vecs in list(all_relations.values()):
        final_vecs.append(combine_relation_vectors(vecs, combine_type))
    node_vectors = KeyedVectors(args.dimensions)
    print("Adding combined node vectors to KeyedVectors")
    node_vectors.add(nodes, final_vecs)
    return node_vectors

def get_classifier(method):
    if method is "svc":
        c = svm.SVC(C=1, kernel='linear')
    elif method is "logit":
        c = LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
    elif method is "mlp":
        c = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200,), random_state=1)
    else:
        raise ValueError("Bad type of classification method %s!" % method)
    return c
def evaluate(data_Y, predicted):
    '''
    calculate evaluation metrics
    '''
    
    print("accuracy",metrics.accuracy_score(data_Y, predicted))
    print("f1 score macro",metrics.f1_score(data_Y, predicted, average='macro')) 
    print("f1 score micro",metrics.f1_score(data_Y, predicted, average='micro')) 
    print("precision score",metrics.precision_score(data_Y, predicted, average='macro')) 
    print("recall score",metrics.recall_score(data_Y, predicted, average='macro')) 
    print("hamming_loss",metrics.hamming_loss(data_Y, predicted))
    print("classification_report", metrics.classification_report(data_Y, predicted))
    # print "log_loss", metrics.log_loss(data_Y, predicted)
    print("zero_one_loss", metrics.zero_one_loss(data_Y, predicted))
    # print "AUC&ROC",metrics.roc_auc_score(data_Y, predicted)
    # print "matthews_corrcoef", metrics.matthews_corrcoef(data_Y, predicted)

def save_features(file, data):
    return np.savetxt(file, data)

def make_edge_embeddings(model):
    edge_embeddings = dict()
    edge_counts = dict()
    for word in model.index2word:
        if "EDGE" in word:
            # What edge is here? 
            edge = edgesfromrelstr(word)[0]
            embedding = model.wv[edge]
            if edge not in edge_embeddings:
                edge_embeddings[edge] = embedding
                edge_counts[edge] = 1
            else:
                edge_embeddings[edge] = np.add(embedding, edge_embeddings[edge])
                edge_counts[edge] += 1
    for edge, embedding in edge_embeddings.items():
        edge_embeddings[edge] = np.divide(embedding, float(edge_counts[edge]))
    return edge_embeddings        


def load_triples(test_file):
    triples = []
    count = 0
    with open(test_file) as f:
        for line in f:
            count = count+1
            result = line.rstrip().split("\t")
            node1 = result[0]
            node2 = result[1]
            rel = edge2str(result[2])
            edgepathstr = edgepath2str([rel], node2)
            # true_label.append(int(relation))
            # vector1 = model.wv[node1]
            # vector2 = model.wv[node2]
            # vector = np.subtract(vector1, vector2) #np.concatenate((vector1,vector2), axis=0)
            triples.append((node1, rel, node2))
            # instance.append(vector)
            if count % 10000 == 0:
            # print vector,relation
                print("load data",str(count))
    return triples