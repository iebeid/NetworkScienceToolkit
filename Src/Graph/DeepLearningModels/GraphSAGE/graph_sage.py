import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import gensim
from itertools import groupby

from gensim.models.keyedvectors import KeyedVectors

#import utils
tf.disable_v2_behavior()

tf.config.experimental.list_physical_devices('GPU')


class graphsage_model():
    """
    create heterogenous model for chem2bio2rdf
    """
    def __init__(self, G_pos, G_neg, nodemap, walks, latent_dim=500, epochs=5, features=None):

        self.G_pos = G_pos
        self.G_neg = G_neg
        self.nodemap = nodemap
        self.n_nodes = len(nodemap.values())
        if features is None:
            data = np.array(list(map(str, nodemap.values())))
            data = data.reshape(data.shape[0], 1)
            model = gensim.models.Word2Vec(data, size=500, min_count=0)
            self.features = np.array([model[str(idx)] for idx in range(self.n_nodes)])
        else:
            self.features = features

        self.walks = walks
        self.walk_length = 5
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.f_dim = self.features.shape[1]
        # self.total_size = len(G_pos.nodes)
        # self.train_nodes_size = len(self.train_nodes)
        self.batch_size = 64
        # self.pos_compound_size = 2
        # self.pos_gene_size = 1
        # self.neg_compound_size = 15
        # self.neg_gene_size = 10
        self.negative_sample_size = self.walk_length
        self.positive_sample_size = self.walk_length

        """
        initialize input variables
        """
        self.input_x = tf.placeholder(tf.float32,[None, 1+self.positive_sample_size+self.negative_sample_size, self.f_dim])
        self.input_x_center = tf.placeholder(tf.float32,[None, 1+self.positive_sample_size+self.negative_sample_size, self.f_dim])
        # self.compound = tf.placeholder(tf.float32,[None, self.pos_compound_size+self.neg_compound_size, self.compound_size])
        # self.gene = tf.placeholder(tf.float32,[None, self.pos_gene_size+self.neg_gene_size, self.gene_size])
        """
        initial relation type binds
        """
        self.init_binds = tf.keras.initializers.he_normal(seed=None)
        self.shape_relation = (self.latent_dim,)
        self.relation_binds = tf.Variable(self.init_binds(shape=self.shape_relation))
        """
        initial relation type similar
        """
        self.init_similar = tf.keras.initializers.he_normal(seed=None)
        self.shape_relation = (self.latent_dim,)
        self.relation_similar = tf.Variable(self.init_similar(shape=self.shape_relation))


    def config_model(self):
        self.build_model()
        self.get_latent_rep()
        self.SGNN_loss()
        self.train_step_neg = tf.train.AdamOptimizer(1e-3).minimize(self.negative_sum)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def build_model(self):
        """
        build local gcn layer
        """
        self.Dense_gcn = tf.layers.dense(inputs=self.input_x,
                                         units=self.latent_dim,
                                         kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                         activation=tf.nn.relu)

        self.dense_center = tf.layers.dense(inputs=self.input_x_center,
                                            units=self.latent_dim,
                                            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                            activation=tf.nn.relu)

        self.concat_x = tf.concat((self.Dense_gcn,self.dense_center),axis=2)

        self.Dense_final = tf.layers.dense(inputs=self.concat_x,
                                           units=self.latent_dim,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.relu)

    def get_latent_rep(self):
        """
        prepare latent representation for skip-gram model
        """


        """
        get center node representation, case where center node is compound
        """
        idx_origin = tf.constant([0])
        self.x_origin =tf.gather(self.Dense_final,idx_origin,axis=1)
        """
        total data case
        """
        idx_skip = tf.constant([i+1 for i in range(self.positive_sample_size)])
        idx_negative = tf.constant([i+self.positive_sample_size+1 for i in range(self.negative_sample_size)])
        self.x_skip = tf.gather(self.Dense_final,idx_skip,axis=1)
        self.x_negative = tf.gather(self.Dense_final,idx_negative,axis=1)

    def SGNN_loss(self):
        """
        implement sgnn loss
        """
        negative_training_norm = tf.math.l2_normalize(self.x_negative, axis=2)

        skip_training = tf.broadcast_to(self.x_origin,
                                        [self.batch_size, self.negative_sample_size, self.latent_dim])

        skip_training_norm = tf.math.l2_normalize(skip_training, axis=2)

        dot_prod = tf.multiply(skip_training_norm, negative_training_norm)

        dot_prod_sum = tf.reduce_sum(dot_prod, 2)

        sum_log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(tf.reduce_mean(dot_prod_sum, 1))))

        positive_training = tf.broadcast_to(self.x_origin, [self.batch_size, self.walk_length, self.latent_dim])

        positive_skip_norm = tf.math.l2_normalize(self.x_skip, axis=2)

        positive_training_norm = tf.math.l2_normalize(positive_training, axis=2)

        dot_prod_positive = tf.multiply(positive_skip_norm, positive_training_norm)

        dot_prod_sum_positive = tf.reduce_sum(dot_prod_positive, 2)

        sum_log_dot_prod_positive = tf.math.log(tf.math.sigmoid(tf.reduce_mean(dot_prod_sum_positive, 1)))

        self.negative_sum = tf.math.negative(
            tf.reduce_sum(tf.math.add(sum_log_dot_prod, sum_log_dot_prod_positive)))



    def get_features(self, node_idx):

        return self.features[node_idx]

    def aggr(self, G, center_node_idx):

        ''' Aggregate features of surrounding nodes '''

        features = self.get_features(center_node_idx)
        features = features.reshape(1, features.shape[0])
        center_node_vector = features
        neighbors = [n for n in G.neighbors(center_node_idx)]
        center_node_degree = len(neighbors)

        aggr_vector = np.zeros(center_node_vector.shape)
        if neighbors:
            for neighbor_node_idx in neighbors:
                neighbor_degree = len([n for n in G.neighbors(neighbor_node_idx)])
                neighbor_vector = self.get_features(neighbor_node_idx)
                normalization = math.sqrt(center_node_degree*neighbor_degree)
                normalization = 1.0/normalization
                neighbor_vector_normed = normalization*neighbor_vector
                aggr_vector += neighbor_vector_normed

        return center_node_vector, aggr_vector


    def sample_pos_walk(self, node_idx):
        ''' Retrieve the nodes from the walk associated with the node '''

        return self.run_random_walk(self.G_pos, node_idx)


    def create_all_batches(self):
        ''' Create the batches of indices that are used to sample from the data '''
        batch_indices = np.array(list(range(len(self.G_pos.nodes))))
        shuffle = np.random.choice(len(batch_indices), len(batch_indices), replace=0)
        batch_indices = batch_indices[shuffle]
        batches = []
        for i in range(0, len(batch_indices), self.batch_size):
            batches.append(batch_indices[i:i+self.batch_size])

        return np.array(batches)

    def create_batch(self, batch_indices):

        center_nodes_batch = np.zeros((len(batch_indices), 1+self.positive_sample_size+self.negative_sample_size, self.f_dim))
        aggr_nodes_batch = np.zeros((len(batch_indices), 1+self.positive_sample_size+self.negative_sample_size, self.f_dim))

        for i, node_idx in enumerate(batch_indices):

            node_center, node_aggr = self.aggr(self.G_pos, node_idx)
            center_nodes_batch[i, 0, :] = node_center
            aggr_nodes_batch[i, 0, :] = node_aggr

            j = 1
            for n_idx in self.sample_pos_walk(node_idx):
                neighbor_center, neighbor_aggr = self.aggr(self.G_pos, n_idx)
                center_nodes_batch[i, j, :] = neighbor_center
                aggr_nodes_batch[i, j, :] = neighbor_aggr
                j += 1

            k = j
            for n_idx in self.sample_neg_walk(node_idx):
                neighbor_center, neighbor_aggr = self.aggr(self.G_neg, n_idx)
                center_nodes_batch[i, k, :] = neighbor_center
                aggr_nodes_batch[i, k, :] = neighbor_aggr
                k += 1

        return center_nodes_batch, aggr_nodes_batch

    def sample_neg_walk(self, node_idx):
        walk = self.run_random_walk(self.G_neg, node_idx)
        return walk

    def run_random_walk(self, G, node_idx):
        walk = []

        curr_node = node_idx
        for j in range(self.walk_length):
            neighbors = [n for n in G.neighbors(curr_node)]
            if neighbors:
                next_node = random.choice(neighbors)
            else:
                return walk
            # self co-occurrences are useless
            if curr_node != node_idx:
                walk.append(curr_node)
            curr_node = next_node

        return walk

    def train(self):

        batches = self.create_all_batches()
        for i in range(self.epochs):
            for k, batch in enumerate(batches):
                center_batch, aggr_batch = self.create_batch(batch)
                err_ = self.sess.run([self.negative_sum,self.train_step_neg],feed_dict={self.input_x:aggr_batch,
                                                                                        self.input_x_center:center_batch})

                print(f'epoch: {i}\tbatch: {k}/{len(batches)}\terror: {err_[0]}')
                break
    # """
    # GCN aggregator for compound
    # """
    # def gcn_agg_compound(self,compoundid):
    #     #one_sample = np.zeros(self.total_size)
    #     #neighbor_compound = self.kg.dic_compound[compoundid]['neighbor_compound']
    #     neighbor_gene = self.kg.dic_compound[compoundid]['neighbor_gene']
    #     agg_vec = self.assign_value_compound(compoundid)
    #     self.compount_origin = agg_vec
    #     center_neighbor_size = len(neighbor_gene)
    #     ave_factor = 1.0 / np.sqrt(center_neighbor_size*center_neighbor_size)
    #     one_sample = agg_vec * ave_factor
    #     """
    #     for i in neighbor_compound:
    #         neighbor_compound_len = len(self.kg.dic_compound[i]['neighbor_compound'])
    #         neighbor_gene_len = len(self.kg.dic_compound[i]['neighbor_gene'])
    #         neighbor_size = neighbor_compound_len + neighbor_gene_len
    #         ave_factor = 1.0 / np.sqrt(neighbor_size * center_neighbor_size)
    #         agg_cur = self.assign_value_compound(i) * ave_factor
    #         one_sample = one_sample + agg_cur
    #     """
    #     for i in neighbor_gene:
    #         neighbor_compound_len = len(self.kg.dic_gene[i]['neighbor_compound'])
    #         neighbor_size = neighbor_compound_len
    #         ave_factor = 1.0 / np.sqrt(neighbor_size * center_neighbor_size)
    #         agg_cur = self.assign_value_gene(i) * ave_factor
    #         one_sample = one_sample + agg_cur
    #
    #     one_sample_final = np.concatenate((agg_vec, one_sample),0)
    #
    #     return one_sample,agg_vec
    #
    # """
    # GCN aggregator for gene
    # """
    # def gcn_agg_gene(self,geneid):
    #     #one_sample = np.zeros(self.total_size)
    #     neighbor_compound = self.kg.dic_gene[geneid]['neighbor_compound']
    #     agg_vec = self.assign_value_gene(geneid)
    #     self.gene_origin = agg_vec
    #     center_neighbor_size = len(neighbor_compound)
    #     ave_factor = 1.0 / np.sqrt(center_neighbor_size*center_neighbor_size)
    #     one_sample = agg_vec * ave_factor
    #     for i in neighbor_compound:
    #         self.check_compound = i
    #         neighbor_compound_len = 0
    #         neighbor_gene_len = 0
    #         if self.kg.dic_compound[i].has_key('neighbor_compound'):
    #             neighbor_compound_len = len(self.kg.dic_compound[i]['neighbor_compound'])
    #         neighbor_gene_len = len(self.kg.dic_compound[i]['neighbor_gene'])
    #         neighbor_size = neighbor_compound_len + neighbor_gene_len
    #         ave_factor = 1.0 / np.sqrt(neighbor_size * center_neighbor_size)
    #         agg_cur = self.assign_value_compound(i) * ave_factor
    #         one_sample = one_sample + agg_cur
    #     one_sample_final = np.concatenate((agg_vec,one_sample),0)
    #
    #     return one_sample,agg_vec
    #
    #
    # """
    # assign value to one compound sample
    # """
    # def assign_value_compound(self,compoundid):
    #     one_sample = np.zeros(self.total_size)
    #     index = self.kg.dic_compound[compoundid]['compound_index']
    #     one_sample[index] = 1
    #
    #     return one_sample
    #
    #
    # """
    # assign value to one gene sample
    # """
    # def assign_value_gene(self,geneid):
    #     one_sample = np.zeros(self.total_size)
    #     index = self.kg.dic_gene[geneid]['gene_index']
    #     one_sample[self.compound_size+index] = 1
    #
    #     return one_sample
    #
    #
    # """
    # preparing data for one metapath
    # """
    # def get_positive_sample_metapath(self,meta_path):
    #     self.compound_nodes = []
    #     self.gene_nodes = []
    #     self.compound_center = []
    #     self.gene_center = []
    #     for i in meta_path:
    #         if i[0] == 'c':
    #             compound_id = i[1]
    #             compound_sample,compound_sample_center = self.gcn_agg_compound(compound_id)
    #             self.compound_nodes.append(compound_sample)
    #             self.compound_center.append(compound_sample_center)
    #         if i[0] == 'g':
    #             gene_id = i[1]
    #             gene_sample,gene_sample_center = self.gcn_agg_gene(gene_id)
    #             self.gene_nodes.append(gene_sample)
    #             self.gene_center.append(gene_sample_center)
    #
    # """
    # prepare data for one metapath negative sample
    # """
    # def get_negative_sample_metapath(self):
    #
    #     self.gene_neg_sample = np.zeros((self.neg_gene_size,self.total_size))
    #     self.compound_neg_sample = np.zeros((self.neg_compound_size,self.total_size))
    #     self.gene_neg_center = np.zeros((self.neg_gene_size,self.total_size))
    #     self.compound_neg_center = np.zeros((self.neg_compound_size,self.total_size))
    #     index = 0
    #     for i in self.neg_nodes_gene:
    #         one_sample_neg_gene,one_sample_neg_gene_center= self.gcn_agg_gene(i)
    #         self.gene_neg_sample[index,:] = one_sample_neg_gene
    #         self.gene_neg_center[index,:] = one_sample_neg_gene_center
    #         index += 1
    #     index = 0
    #     for i in self.neg_nodes_compound:
    #         one_sample_neg_compound, one_sample_neg_compound_center = self.gcn_agg_compound(i)
    #         self.compound_neg_sample[index,:] = one_sample_neg_compound
    #         self.compound_neg_center[index,:] = one_sample_neg_compound_center
    #         index += 1
    #
    # """
    # prepare data for negative hererogenous sampling
    # """
    # def get_negative_samples(self,center_node_type,center_node_index):
    #     self.neg_nodes_compound = []
    #     self.neg_nodes_gene = []
    #     """
    #     get neg set for gene
    #     """
    #     if center_node_type == 'c':
    #         gene_neighbor_nodes = self.kg.dic_compound[center_node_index]['neighbor_gene']
    #         whole_gene_nodes = self.kg.dic_gene.keys()
    #         gene_neighbor_nodes = gene_neighbor_nodes + self.walk_gene
    #         neg_set_gene = [i for i in whole_gene_nodes if i not in gene_neighbor_nodes]
    #         for j in range(self.neg_gene_size):
    #             index_sample = np.int(np.floor(np.random.uniform(0,len(neg_set_gene),1)))
    #             self.neg_nodes_gene.append(neg_set_gene[index_sample])
    #         compound_neighbor_nodes = self.kg.dic_compound[center_node_index]['neighbor_compound']
    #         whole_compound_nodes = self.kg.dic_compound.keys()
    #         compound_neighbor_nodes = compound_neighbor_nodes + self.walk_compound
    #         neg_set_compound = [i for i in whole_compound_nodes if i not in compound_neighbor_nodes]
    #         for j in range(self.neg_compound_size):
    #             index_sample = np.int(np.floor(np.random.uniform(0,len(neg_set_compound),1)))
    #             self.neg_nodes_compound.append(neg_set_compound[index_sample])






    # """
    # prepare one batch data
    # """
    # def get_one_batch(self,meta_path_type,center_node_type,start_index):
    #     #compound_sample = np.zeros((self.batch_size,self.pos_compound_size+self.neg_compound_size,self.compound_size))
    #     #gene_sample = np.zeros((self.batch_size,self.pos_gene_size+self.neg_gene_size,self.gene_size))
    #     one_batch_train = np.zeros((self.batch_size,1+self.positive_sample_size+self.negative_sample_size,self.total_size))
    #     one_batch_train_center = np.zeros((self.batch_size,1+self.positive_sample_size+self.negative_sample_size,self.total_size))
    #     num_sample = 0
    #     increament_step = 0
    #     while num_sample < self.batch_size:
    #         center_node_index = self.train_nodes[increament_step+start_index]
    #         if not 'neighbor_compound' in self.kg.dic_compound[center_node_index]:
    #             increament_step += 1
    #             continue
    #         single_meta_path = self.extract_meta_path(center_node_type,center_node_index,meta_path_type)
    #         self.get_positive_sample_metapath(single_meta_path)
    #         self.get_negative_samples(center_node_type,center_node_index)
    #         self.get_negative_sample_metapath()
    #         #single_compound_sample = np.concatenate((self.compound_nodes,self.compound_neg_sample))
    #         #single_gene_sample = np.concatenate((self.gene_nodes,self.gene_neg_sample))
    #         #compound_sample[num_sample,:,:] = single_compound_sample
    #         #gene_sample[num_sample,:,:] = single_gene_sample
    #         positive_sample = np.concatenate((self.compound_nodes,self.gene_nodes))
    #         positive_sample_center = np.concatenate((self.compound_center,self.gene_center))
    #         negative_sample = np.concatenate((self.compound_neg_sample,self.gene_neg_sample))
    #         negative_sample_center = np.concatenate((self.compound_neg_center,self.gene_neg_center))
    #         total_sample = np.concatenate((positive_sample,negative_sample))
    #         total_sample_center = np.concatenate((positive_sample_center,negative_sample_center))
    #         one_batch_train[num_sample,:,:] = total_sample
    #         one_batch_train_center[num_sample,:,:] = total_sample_center
    #
    #         num_sample += 1
    #
    #     return one_batch_train, one_batch_train_center

    # """
    # train model
    # """
    # def train(self):
    #     iteration = np.int(np.floor(np.float(self.train_nodes_size)/self.batch_size))
    #     epoch=6
    #     self.iter_num = 0
    #     for j in range(epoch):
    #         for i in range(iteration):
    #             self.iter_num = i
    #             #if i > 300:
    #                # break
    #             batch_total,batch_total_center = self.get_one_batch(self.meta_path1,'c',i*self.batch_size)
    #             err_ = self.sess.run([self.negative_sum,self.train_step_neg],feed_dict={self.input_x:batch_total,
    #                                                                                     self.input_x_center:batch_total_center})
    #             print(err_[0])


    def test_whole(self):
        test_compound = np.zeors((self.compound_size,self.pos_compound_size+self.neg_compound_size,self.compound_size))
        test_gene = np.zeros((self.gene_size,self.pos_gene_size+self.neg_gene_size,self.gene_size))
        for i in range(self.compound_size):
            compound[0,0,:] = self.assign_value_compound(compoundid)
        #embed_compound = self.sess.run([self.Dense_compound],feed_dict={self.})
