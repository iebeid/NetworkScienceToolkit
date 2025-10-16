import stellargraph as sg
from gensim.models import Word2Vec
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from gensim import utils
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import gensim.models
import pickle
import random as rn
from tqdm import tqdm
import pandas as pd
from urllib.parse import urlparse
import sys
import argparse
import scipy
from Word2VecSGNS import run
from Word2VecSGNSPlusPlus import runpp



def map_to_letter(number):
    map_value = 0
    if number == '0':
        map_value = 'a'
    if number == '1':
        map_value = 'b'
    if number == '2':
        map_value = 'c'
    if number == '3':
        map_value = 'd'
    if number == '4':
        map_value = 'e'
    if number == '5':
        map_value = 'f'
    if number == '6':
        map_value = 'g'
    if number == '7':
        map_value = 'h'
    if number == '8':
        map_value = 'i'
    if number == '9':
        map_value = 'j'
    return map_value


def map_to_number(number):
    map_value = 0
    if number == 'a':
        map_value = '0'
    if number == 'b':
        map_value = '1'
    if number == 'c':
        map_value = '2'
    if number == 'd':
        map_value = '3'
    if number == 'e':
        map_value = '4'
    if number == 'f':
        map_value = '5'
    if number == 'g':
        map_value = '6'
    if number == 'h':
        map_value = '7'
    if number == 'i':
        map_value = '8'
    if number == 'j':
        map_value = '9'
    return map_value


def map_code_to_letter(code):
    map_value = ''
    for c in code:
        single_map_value = map_to_letter(c)
        map_value = map_value + str(single_map_value)
    return map_value


def map_code_to_number(code):
    map_value = ''
    for c in code:
        single_map_value = map_to_number(c)
        map_value = map_value + str(single_map_value)
    return int(map_value)


def encode_node_type(node_type):
    node_type_code = ""
    if node_type == "Gene ontology":
        node_type_code = "[go]"
    if node_type == "Chemical ontology":
        node_type_code = "[co]"
    if node_type == "Substructure":
        node_type_code = "[ss]"
    if node_type == "Target":
        node_type_code = "[ta]"
    if node_type == "Tissue":
        node_type_code = "[ti]"
    if node_type == "Pathway":
        node_type_code = "[pa]"
    if node_type == "Disease":
        node_type_code = "[di]"
    if node_type == "Chemical Compound/Drug":
        node_type_code = "[dr]"
    if node_type == "Side effect":
        node_type_code = "[se]"
    if node_type == "N/A":
        node_type_code = "[na]"
    return str(node_type_code)


def encode_node_type_to_number(node_type):
    node_type_code = ''
    if node_type == "Gene ontology":
        node_type_code = '1'
    if node_type == "Chemical ontology":
        node_type_code = '2'
    if node_type == "Substructure":
        node_type_code = '3'
    if node_type == "Target":
        node_type_code = '4'
    if node_type == "Tissue":
        node_type_code = '5'
    if node_type == "Pathway":
        node_type_code = '6'
    if node_type == "Disease":
        node_type_code = '7'
    if node_type == "Chemical Compound/Drug":
        node_type_code = '8'
    if node_type == "Side effect":
        node_type_code = '9'
    return str(node_type_code)


def convert_entity_to_code(entity,nodes_dataframe):
    entity_index = nodes_dataframe.index[nodes_dataframe["filtered_node_name"] == str(entity)]
    entity_code = None
    if not entity_index.empty:
        entity_id = str(nodes_dataframe.at[entity_index[0], "node_id"])
        entity_code = map_code_to_letter(str(entity_id))
    return entity_code


def load_graph(node_file, edge_file):
    nodes_df = pd.read_csv(node_file, sep=",", header=None, encoding='utf-8')
    edges_df = pd.read_csv(edge_file, sep=",", header=None, encoding='utf-8')

    node_go_list = []
    node_co_list = []
    node_ss_list = []
    node_ta_list = []
    node_ti_list = []
    node_pa_list = []
    node_di_list = []
    node_dr_list = []
    node_se_list = []
    node_na_list = []
    edges_source_list = []
    edges_target_list = []

    all_nodes = []
    all_node_types = []

    for index, line in nodes_df.iterrows():
        node_id = str(line[0]).rstrip()
        node_type = str(line[1]).rstrip()
        node_type = encode_node_type(node_type)
        node_id_coded = map_code_to_letter(node_id)
        node_id_type_coded = str(node_id_coded)
        if node_type == "[go]":
            node_go_list.append(node_type[1:3]+node_id_type_coded)
        if node_type == "[co]":
            node_co_list.append(node_type[1:3]+node_id_type_coded)
        if node_type == "[ss]":
            node_ss_list.append(node_type[1:3]+node_id_type_coded)
        if node_type == "[ta]":
            node_ta_list.append(node_type[1:3]+node_id_type_coded)
        if node_type == "[ti]":
            node_ti_list.append(node_type[1:3]+node_id_type_coded)
        if node_type == "[pa]":
            node_pa_list.append(node_type[1:3]+node_id_type_coded)
        if node_type == "[di]":
            node_di_list.append(node_type[1:3]+node_id_type_coded)
        if node_type == "[dr]":
            node_dr_list.append(node_type[1:3]+node_id_type_coded)
        if node_type == "[se]":
            node_se_list.append(node_type[1:3]+node_id_type_coded)
        if node_type == "[na]":
            node_na_list.append(node_type[1:3]+node_id_type_coded)
        all_nodes.append(node_id_type_coded)
        all_node_types.append(node_type[1:3])

    for index, line in edges_df.iterrows():
        source = str(line[0]).rstrip()
        target = str(line[1]).rstrip()
        source_node_id_coded = map_code_to_letter(source)
        target_node_id_coded = map_code_to_letter(target)
        source_type = str(all_node_types[int(all_nodes.index(source_node_id_coded))])
        target_type = str(all_node_types[int(all_nodes.index(target_node_id_coded))])
        edges_source_list.append(source_type+source_node_id_coded)
        edges_target_list.append(target_type+target_node_id_coded)

    node_go_df = pd.DataFrame([], index=node_go_list)
    node_co_df = pd.DataFrame([], index=node_co_list)
    node_ss_df = pd.DataFrame([], index=node_ss_list)
    node_ta_df = pd.DataFrame([], index=node_ta_list)
    node_ti_df = pd.DataFrame([], index=node_ti_list)
    node_pa_df = pd.DataFrame([], index=node_pa_list)
    node_di_df = pd.DataFrame([], index=node_di_list)
    node_dr_df = pd.DataFrame([], index=node_dr_list)
    node_se_df = pd.DataFrame([], index=node_se_list)
    node_na_df = pd.DataFrame([], index=node_na_list)
    edges_df = pd.DataFrame({"source": edges_source_list, "target": edges_target_list})

    return sg.StellarGraph({"go": node_go_df,
                            "co": node_co_df,
                            "ss": node_ss_df,
                            "ta": node_ta_df,
                            "ti": node_ti_df,
                            "pa": node_pa_df,
                            "di": node_di_df,
                            "dr": node_dr_df,
                            "se": node_se_df,
                            "na": node_na_df}, edges_df)


def load_meta_paths(metapaths_file):
    metapaths_list = []
    metapaths_df = pd.read_csv(metapaths_file, sep=",", header=None, encoding='utf-8', names=["id","a","b","c","d","e"])
    for index, row in metapaths_df.iterrows():
        metapath_list = []
        for node in row[1:]:
            if node != "0":
                metapath_list.append(str(node))
        metapaths_list.append(metapath_list)
    for m in metapaths_list:
        print(m)
    return metapaths_list


def generate_random_walks(graph, walk_length, walks_per_node, metapaths):
    nodes = list(graph.nodes())
    walks = []
    for metapath in tqdm(metapaths):
        minimum_lengths_of_metapath = len(metapath)
        metapath = metapath * ((walk_length // (len(metapath) - 1)) + 1)
        for _ in tqdm(range(walks_per_node)):
            for node in nodes:
                current_node = node
                walk = ([])
                if current_node[0:2] == metapath[0]:
                    for d in range(len(metapath)-1):
                        walk.append(current_node)
                        neighbours = graph.neighbors(current_node, use_ilocs=False)
                        filtered_neighbors = []
                        for n_node in neighbours:
                            if n_node[0:2] == metapath[d+1]:
                                filtered_neighbors.append(n_node)
                        if len(filtered_neighbors) == 0:
                            break
                        current_node = rn.choice(filtered_neighbors)
                if len(walk) >= minimum_lengths_of_metapath:
                    walks.append(walk)
    return walks


def save_walks_to_file(walks_list, filename):
    with open(filename, 'w') as f:
        for item in walks_list:
            print(item)
            for i in item:
                f.write("%s " % str(i))
            f.write("\n")


def metapath2vec(random_walks_file, metapath2vec_model_file, embedding_size, window_size, number_of_negative_samples, epochs, gensim):
    class MyCorpus(object):
        def __iter__(self):
            for line in open(random_walks_file):
                # assume there's one document per line, tokens separated by whitespace
                yield utils.simple_preprocess(line)
    walks = MyCorpus()
    if gensim:
        model = Word2Vec(sentences=walks,size=embedding_size, alpha=0.05, window=window_size, workers=32, iter=epochs,
                         min_alpha=0.0001, sg=1, hs=0, negative=number_of_negative_samples, sample=0, compute_loss=True, sorted_vocab=1, min_count=0)
        model.save(metapath2vec_model_file)
        convert_gensim_model_to_local_model(model,metapath2vec_model_file)
    else:
        run()


def metapath2vecplusplus():
    print("Metapath2vec++")
    runpp()


def classify_nodes(model):
    model_df = pd.read_csv(model, sep=",", encoding='utf-8')
    vectors = []
    types = []
    for index,row in model_df.iterrows():
        vectors.append(list(row[2]))
        types.append(row[2][0:2])
    X = np.array(vectors)
    T = np.array(types)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, T, test_size=0.25, random_state=0)
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)
    predictions = logisticRegr.predict(x_test)
    score = logisticRegr.score(x_test, y_test)
    print(score)


def visualize_embedding(model,figure_filename,upper_limit):
    model_df = pd.read_csv(model, sep=",", encoding='utf-8')
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    vectors = []
    types = []
    i = 0
    for index,row in model_df.iterrows():
        vectors.append(list(row[2]))
        types.append(row[2][0:2])
        i = i + 1
        if i == upper_limit:
            break
    X = np.array(vectors)
    T = np.array(types)
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=T,  legend='full', palette='bright')
    plt.savefig(figure_filename)


def get_word_index(model, word):
    target_word_index = model.index[model["word"] == word]
    return int(target_word_index.tolist()[0])


def cosine_similarity(vector1, vector2):
    similarity = float(float(1.0) - (
        scipy.spatial.distance.cosine(np.array(vector1, dtype=np.float), np.array(vector2, dtype=np.float))))
    return similarity

def most_similar(word, top_n, model, model_dataframe):
    word_vector = model[get_word_index(model_dataframe, word)]
    scores_dictionary = {}
    scores = []
    for index, row in enumerate(model):
        w_vector = list(row)
        score = cosine_similarity(word_vector, w_vector)
        current_word = str(model_dataframe.at[index, "word"])
        current_dictionary = {float(score): current_word}
        scores_dictionary.update(current_dictionary)
        scores.append(score)
    scores.sort(reverse=True)
    similar_words = []
    for i in range(top_n):
        current_score = scores[i]
        similar_word = scores_dictionary.get(current_score)
        if word != similar_word:
            similar_words.append((similar_word, current_score))
    return similar_words


def most_similar_nodes_gensim(model,node,hit,left_node_types,right_node_types):
    result = model.wv.most_similar(positive=[str(node)], topn=int(len(model.wv.vocab)-1))
    node_type = node[0:2]
    similar_drugs = []
    similar_targets = []
    filtered_similar_drugs = []
    filtered_similar_targets = []
    for similar_node in result:
        similar_node = similar_node[0]
        similar_node_type = similar_node[0:2]
        if similar_node_type in left_node_types:
            similar_drugs.append(similar_node)
        if similar_node_type in right_node_types:
            similar_targets.append(similar_node)
    for i in range(hit):
        if node_type in left_node_types:
            filtered_similar_targets.append(similar_targets[i])
        if node_type in right_node_types:
            filtered_similar_drugs.append(similar_drugs[i])
    if node_type in left_node_types:
        return filtered_similar_targets
    if node_type in right_node_types:
        return filtered_similar_drugs


def evaluate_hits_chem2bio2rdf(hits_at,evaluation_set_file,model_file,evaluation_results_file):
    similarity_dataframe = pd.read_csv(evaluation_set_file, sep="\t", header=None, encoding='utf-8')
    true_positive = []
    false_negative = []
    true_negative = []
    false_positive  = []
    model = gensim.models.Word2Vec.load(model_file, mmap='r')
    print(model)
    results_list = []
    left_node_types = ["dr","go","di","co","ti","pa"]
    right_node_types = ["ta"]
    for hit in hits_at:
        for index, row in tqdm(similarity_dataframe.iterrows()):
            a = row[0]
            b = row[1]
            expected = row[2]
            if a[0:2] in left_node_types and b[0:2] in right_node_types:
                if (a in model.wv.vocab) and (b in model.wv.vocab):
                    result_a = most_similar_nodes_gensim(model,a,hit,left_node_types,right_node_types)
                    result_b = most_similar_nodes_gensim(model,b,hit,left_node_types,right_node_types)
                    if expected == 1:
                        if (b in result_a) or (a in result_b):
                            true_positive.append(1)
                        else:
                            false_negative.append(1)
                    if expected == 0:
                        if (b in result_a) or (a in result_b):
                            false_positive.append(1)
                        else:
                            true_negative.append(1)
        true_positive_count = len(true_positive)
        false_negative_count = len(false_negative)
        true_negative_count = len(true_negative)
        false_positive_count = len(false_positive)
        f1 = float(true_positive_count/(true_positive_count + 0.5*(false_positive_count + false_negative_count)))
        print("F1 Accuracy Score @ " + str(hit) + " is " + str(f1))
        results_list.append("F1 Accuracy Score @ " + str(hit) + " is " + str(f1))
    with open(evaluation_results_file, 'w') as f:
        for item in results_list:
            f.write("%s " % str(item))
            f.write("\n")


def convert_gensim_model_to_local_model(model,model_name,nodes_file):
    nodes_dataframe = pd.read_csv(nodes_file, sep=",", encoding='utf-8')
    node_names = []
    node_vectors = []
    node_ids = []
    for entity in model.wv.vocab:
        node_vector = model.wv[entity]
        coded_node_name = entity[2:]
        coded_node_to_id = map_code_to_number(coded_node_name)
        node_name = nodes_dataframe.at[coded_node_to_id, "node_name"]
        node_names.append(node_name)
        node_vectors.append(node_vector)
        node_ids.append(entity)
    model_df = pd.DataFrame(list(zip(node_ids, node_names, node_vectors)), columns=['node_id', 'node_name', 'node_vector'])
    model_df.to_csv(model_name)


def generate_evaluation_set(positive_file,negative_file,nodes_file,result_evaluation_file):
    positive_dataframe = pd.read_csv(positive_file, sep="\t", header=None, encoding='utf-8')
    negative_dataframe = pd.read_csv(negative_file, sep="\t", header=None, encoding='utf-8')
    nodes_dataframe = pd.read_csv(nodes_file, sep=",", encoding='utf-8')
    list_filtered_entites = []
    for index, row in tqdm(nodes_dataframe.iterrows()):
        parsed = urlparse(row[1])
        parts = parsed.path.split("/")
        entity = parts[-1].strip("0")
        filtered_entity = entity.split(":")[-1].strip("0")
        list_filtered_entites.append(filtered_entity)
    nodes_dataframe["filtered_node_name"] = list_filtered_entites
    all_nodes = []
    all_node_types = []
    for index, line in tqdm(nodes_dataframe.iterrows()):
        node_id = str(line[0]).rstrip()
        node_type = str(line[2]).rstrip()
        node_type = encode_node_type(node_type)
        node_id_coded = map_code_to_letter(node_id)
        all_nodes.append(str(node_id_coded))
        all_node_types.append(node_type[1:3])
    sample_tuples = []
    for index, row in tqdm(positive_dataframe.iterrows()):
        positive_a = convert_entity_to_code(row[0],nodes_dataframe)
        positive_b = convert_entity_to_code(row[1],nodes_dataframe)
        if positive_a in all_nodes and positive_b in all_nodes:
            positive_a_type = str(all_node_types[int(all_nodes.index(positive_a))])
            positive_b_type = str(all_node_types[int(all_nodes.index(positive_b))])
            positive_sample_tuple = (positive_a_type+positive_a,positive_b_type+positive_b,1)
            sample_tuples.append(positive_sample_tuple)
    for index2, row2 in tqdm(negative_dataframe.iterrows()):
        negative_a = convert_entity_to_code(row2[0],nodes_dataframe)
        negative_b = convert_entity_to_code(row2[1],nodes_dataframe)
        if negative_a in all_nodes and negative_b in all_nodes:
            negative_a_type = str(all_node_types[int(all_nodes.index(negative_a))])
            negative_b_type = str(all_node_types[int(all_nodes.index(negative_b))])
            negative_sample_tuple = (negative_a_type+negative_a,negative_b_type+negative_b,0)
            sample_tuples.append(negative_sample_tuple)
    evaluation_samples = pd.DataFrame(sample_tuples)
    evaluation_samples = evaluation_samples.dropna()
    evaluation_samples.to_csv(result_evaluation_file,index=False,sep="\t")


def main(walk_length = 6,walks_per_node = 2,embedding_size = 128,window_size = 7,number_of_negative_samples = 5,evaluation_hits = [5,10,25,50,100],pp = False,gensim = True):
    # Files
    node_file = "fixed_nodes_ids.csv"
    edge_file = "fixed_edges_ids.csv"
    metapaths_file = "metapaths.txt"
    random_walks_file = "metapaths_random_walks_"+str(walk_length)+"_"+str(walks_per_node)+".txt"
    graph_pickle_file = "Chem2Bio2StellarGraph"
    metapath2vec_model_file = "gensim-full-c2b2r_"+str(walk_length)+"_"+str(walks_per_node)+".model"
    evaluation_set_file = "similarity-evaluation.txt"
    evaluation_results_file = "evaluation_results.txt"
    positive_file = "positive.txt"
    negative_file = "negative.txt"
    nodes_file = "nodes.csv"
    # Load metapaths
    metapaths = load_meta_paths(metapaths_file)
    # # Preprocess graph - conver graph to binary file for fast loading - this can be done one time
    # graph_binary = load_graph(node_file, edge_file)
    # pickle.dump(graph_binary, open(graph_pickle_file, "wb"))
    # Load graph
    graph = pickle.load(open(graph_pickle_file, "rb"))
    print("Number of nodes {} and number of edges {} in graph.".format(graph.number_of_nodes(), graph.number_of_edges()))
    # Generate random walks
    walks = generate_random_walks(graph, walk_length, walks_per_node, metapaths)
    print("Number of random walks: {}".format(len(list(walks))))
    # Save generated walks to file
    save_walks_to_file(walks, random_walks_file)
    # Run Metapath2vec
    if pp:
         metapath2vecplusplus()
    else:
         metapath2vec(random_walks_file, metapath2vec_model_file, embedding_size, window_size, number_of_negative_samples)
    # Evaluation
    generate_evaluation_set(positive_file,negative_file,nodes_file)
    evaluate_hits_chem2bio2rdf(evaluation_hits,evaluation_set_file,metapath2vec_model_file,evaluation_results_file)
    # Other experiments
    # 1- Visualize embedding
    visualize_embedding(model, node_file, edge_file, figure_filename, visualization_upper_limit, embedding_size)
    # 2- Node classification
    classify_nodes(model,node_file,edge_file,embedding_size)


def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.",add_help=False)
    parser.add_argument("-l", "--walk_length", type=int, help="Your input file.")
    parser.add_argument("-n", "--walks_per_node", type=int, help="Your destination output file.")
    parser.add_argument("-d", "--embedding_size", type=int, help="embedding vector size")
    parser.add_argument("-w", "--window_size", type=int, help="skip gram context window size")
    parser.add_argument("-k", "--number_of_negative_samples", type=int, help="number of negative samples to be taken into consideration in the skip gram")
    parser.add_argument("-h", "--evaluation_hits", type=int, nargs="+", help="a list of evaluation hits")
    parser.add_argument("-pp", "--pp", type=bool, help="choose to run metpath2vec++")
    parser.add_argument("-g", "--gensim", type=bool, help="in metapath2vec we can use the gensim implementation of word2vec")
    options = parser.parse_args(args)
    return options

if __name__ == "__main__":
    if sys.argv[1:] != None:
        options = getOptions(sys.argv[1:])
        walk_length = options.walk_length
        walks_per_node = options.walks_per_node
        embedding_size = options.embedding_size
        window_size = options.window_size
        number_of_negative_samples = options.number_of_negative_samples
        evaluation_hits = options.evaluation_hits
        pp = options.pp
        gensim = options.gensim
        main(walk_length,walks_per_node,embedding_size,window_size,number_of_negative_samples,evaluation_hits,pp,gensim)
    else:
        main()
