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


# def map_to_number(number):
#     map_value = 0
#     if number == 'a':
#         map_value = '0'
#     if number == 'b':
#         map_value = '1'
#     if number == 'c':
#         map_value = '2'
#     if number == 'd':
#         map_value = '3'
#     if number == 'e':
#         map_value = '4'
#     if number == 'f':
#         map_value = '5'
#     if number == 'g':
#         map_value = '6'
#     if number == 'h':
#         map_value = '7'
#     if number == 'i':
#         map_value = '8'
#     if number == 'j':
#         map_value = '9'
#     return map_value
#
# def map_code_to_number(code):
#     map_value = ''
#     for c in code:
#         single_map_value = map_to_number(c)
#         map_value = map_value + str(single_map_value)
#     return int(map_value)
#
#
#
# nodes_file = "nodes.csv"
# model_file = "gensim-full-c2b2r_6_2.model"
#
# model = gensim.models.Word2Vec.load(model_file, mmap='r')
# print(model)
#
# nodes_dataframe = pd.read_csv(nodes_file, sep=",", encoding='utf-8')
#
# model_vectors = []
#
# for entity in tqdm(model.wv.vocab):
#     node_vector = model.wv[entity]
#     coded_node_name = entity[2:]
#     coded_node_to_id = map_code_to_number(coded_node_name)
#     node_name = nodes_dataframe.at[coded_node_to_id, "node_name"]
#     node = dict(node_name=str(node_name), node_vector=np.array(node_vector))
#     model_vectors.append(node)
#
# pickle.dump(model_vectors, open("c2b2r_6_2_model_embeddings", "wb"))

embeddings = pickle.load(open("c2b2r_6_2_model_embeddings", "rb"))

for n in embeddings:
    print(n)
