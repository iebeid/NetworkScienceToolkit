import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class TransEModel:
    def __init__(self, triples: list, entities: list, relations: list, margin: float, dim: int, b: int, epochs: int,
                 learning_rate: float) -> None:
        self.data = triples
        self.entities = entities
        self.relations = relations
        self.n_r = len(relations)
        self.n_e = len(entities)
        self.b = b
        self.k = dim
        self.margin = margin
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.initializer = tf.keras.initializers.RandomUniform(minval=float(-1 * (6 / np.sqrt(self.k))),
                                                               maxval=float(6 / np.sqrt(self.k)), seed=1234)
        self.parameters = []
        self.l = tf.nn.l2_normalize(tf.Variable(self.initializer(shape=(self.n_r, self.k)), name="l"), axis=1)
        self.e = tf.Variable(self.initializer(shape=(self.n_e, self.k)), name="e")
        self.parameters.append(self.l)
        self.parameters.append(self.e)

    def entity_by_relation(self):
        self.relational_dict = {}
        for r in self.relations:
            rel_entities = []
            for triple in self.data:
                h = triple[0]
                l = triple[1]
                t = triple[2]
                if l == r:
                    rel_entities.append(h)
                    rel_entities.append(t)
            rel_entities = list(set(rel_entities))
            self.relational_dict[r] = rel_entities

    def corrupt_dataset(self):
        self.relation_corrupt = {}
        for r in self.relations:
            corrupted_triples = []
            for triple in self.data:
                h = triple[0]
                l = triple[1]
                t = triple[2]
                if l == r:
                    h_dash = random.choice(self.relational_dict[l])
                    t_dash = random.choice(self.relational_dict[l])
                    corrupted_triples.append((h_dash, l, t))
                    corrupted_triples.append((h, l, t_dash))
            self.relation_corrupt[r] = corrupted_triples

    def prepare_data(self):
        self.entity_by_relation()
        self.corrupt_dataset()
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.data)

    def optimizer(self):
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        return optimizer

    def train(self):
        self.prepare_data()
        for _ in range(self.epochs):
            self.e = tf.nn.l2_normalize(self.e, axis=1)
            self.batches = self.train_dataset.batch(self.b)
            for batch in self.batches:
                for triple in batch:
                    with tf.GradientTape() as tape:
                        h = str(triple.numpy()[0].decode('utf-8'))
                        l = str(triple.numpy()[1].decode('utf-8'))
                        t = str(triple.numpy()[2].decode('utf-8'))
                        corrupted_triple = random.choice(self.relation_corrupt[l])
                        h_dash = corrupted_triple[0]
                        t_dash = corrupted_triple[2]
                        h_vector = tf.nn.embedding_lookup(self.e, self.entities.index(h))
                        t_vector = tf.nn.embedding_lookup(self.e, self.entities.index(t))
                        h_dash_vector = tf.nn.embedding_lookup(self.e, self.entities.index(h_dash))
                        t_dash_vector = tf.nn.embedding_lookup(self.e, self.entities.index(t_dash))
                        l_vector = tf.nn.embedding_lookup(self.l, self.relations.index(l))
                        distance1 = tf.math.sqrt(tf.reduce_sum(tf.square(tf.add(h_vector, l_vector) - t_vector)))
                        distance2 = tf.math.sqrt(tf.reduce_sum(tf.square(tf.add(h_dash_vector, l_vector) - t_dash_vector)))
                        loss = self.margin + distance1 - distance2
                    gradients = tape.gradient(loss, self.parameters)
                    self.optimizer().apply_gradients(list(zip(gradients, self.parameters)))
                    self.l = self.parameters[0]
                    self.e = self.parameters[1]