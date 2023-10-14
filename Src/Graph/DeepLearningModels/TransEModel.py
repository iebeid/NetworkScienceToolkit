import random
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

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
        self.l = tf.Variable(tf.nn.l2_normalize(tf.Variable(self.initializer(shape=(self.n_r, self.k))), axis=1),
                             name="l", trainable=True, dtype=tf.float32)
        self.e = tf.Variable(self.initializer(shape=(self.n_e, self.k)), name="e", trainable=True, dtype=tf.float32)

    def process_dataset(self):
        self.entity_per_relation = {}
        self.false_per_relation = {}
        self.true_per_relation={}
        for r in self.relations:
            rel_entities = []
            corrupted_triples = []
            true_triples=[]
            for triple in self.data:
                h = triple[0]
                l = triple[1]
                t = triple[2]
                if l == r:
                    rel_entities.append(h)
                    rel_entities.append(t)
            rel_entities = list(set(rel_entities))
            for triple in self.data:
                h = triple[0]
                l = triple[1]
                t = triple[2]
                if l == r:
                    h_dash = random.choice(rel_entities)
                    t_dash = random.choice(rel_entities)
                    corrupted_triples.append((h_dash, t))
                    corrupted_triples.append((h, t_dash))
            for triple in self.data:
                h = triple[0]
                l = triple[1]
                t = triple[2]
                if l == r:
                    true_triples.append((h,t))
            self.entity_per_relation[r] = rel_entities
            self.false_per_relation[r] = corrupted_triples
            self.true_per_relation[r]=true_triples
        # self.train_dataset = tf.data.Dataset.from_tensor_slices(self.data)

    # def transe_loss(self, entity_tensor, relation):
    #
    #     distance1 = tf.math.sqrt(tf.reduce_sum(tf.square(tf.add(h_vector, l_vector) - t_vector)))
    #     distance2 = tf.math.sqrt(tf.reduce_sum(tf.square(tf.add(h_dash_vector, l_vector) - t_dash_vector)))
    #     return self.margin + distance1 - distance2

    # def tensor_element_to_index(self, tensor):
    #
    #     index = tf.where(tf.equal(tensor, tensor_element))
    #
    #
    # def data_to_index(self, data_tensor):
    #     tf.map_fn(fn=lambda i: self.entities.index(str(i.numpy()[0].decode('utf-8'))), elems=data_tensor)
    #     T_batch = tf.map_fn(self.tensor_element_to_index, data_tensor)


    def optimizer(self):
        return tf.keras.optimizers.SGD(learning_rate=self.learning_rate)


    # def get_relation_corrupted_triple(self, r):
    #     return tf.convert_to_tensor(random.choice(self.false_per_relation[r]))

    def get_vector(self, entity):
        return tf.nn.embedding_lookup(self.e, self.entities.index(entity.numpy().decode('utf-8')))

    def data_to_index(self, data_tensor, relation):
        # vectors = tf.convert_to_tensor(tf.cast(tf.map_fn(fn=lambda i: self.get_vector(i), elems=data_tensor),dtype=tf.string))
        h_vector = tf.cast(tf.nn.embedding_lookup(self.e, self.entities.index(data_tensor.numpy()[0].decode('utf-8'))),dtype=tf.float32)
        t_vector = tf.cast(tf.nn.embedding_lookup(self.e, self.entities.index(data_tensor.numpy()[1].decode('utf-8'))),dtype=tf.float32)
        h_dash_vector = tf.cast(tf.nn.embedding_lookup(self.e, self.entities.index(data_tensor.numpy()[2].decode('utf-8'))),dtype=tf.float32)
        t_dash_vector = tf.cast(tf.nn.embedding_lookup(self.e, self.entities.index(data_tensor.numpy()[3].decode('utf-8'))),dtype=tf.float32)
        l_vector = tf.nn.embedding_lookup(self.l, self.relations.index(relation))
        return tf.convert_to_tensor([h_vector,t_vector,h_dash_vector,t_dash_vector, l_vector],dtype=tf.float32)

    # def entity_to_vector(self, ):
    def entity_to_index(self, entity_tensor):
        indices=tf.convert_to_tensor([int(self.entities.index(str(entity_tensor.numpy()[0].decode('utf-8')))),
                 int(self.entities.index(str(entity_tensor.numpy()[1].decode('utf-8')))),
                 int(self.entities.index(str(entity_tensor.numpy()[2].decode('utf-8')))),
                 int(self.entities.index(str(entity_tensor.numpy()[3].decode('utf-8'))))])
        return indices


    def transe_loss(self, entity_index_tensor, relation):
        vectors = tf.nn.embedding_lookup(self.e,entity_index_tensor)
        h_vector=vectors[0]
        t_vector=vectors[1]
        h_dash_vector = vectors[2]
        t_dash_vector = vectors[3]
        l_vector = tf.nn.embedding_lookup(self.l, self.relations.index(relation))
        distance1 = tf.math.sqrt(tf.reduce_sum(tf.square(tf.add(h_vector, l_vector) - t_vector)))
        distance2 = tf.math.sqrt(tf.reduce_sum(tf.square(tf.add(h_dash_vector, l_vector) - t_dash_vector)))
        return self.margin + distance1 - distance2



    def train(self):
        optimizer = self.optimizer()
        self.process_dataset()
        for epoch in tqdm(range(self.epochs)):
            start_time = time.perf_counter()
            self.e = tf.Variable(tf.nn.l2_normalize(self.e, axis=1), name="e", trainable=True, dtype=tf.float32)
            for r in tqdm(self.relations):
                self.S = tf.data.Dataset.from_tensor_slices(self.true_per_relation[r]).batch(self.b)
                for S_batch in self.S:
                    T_batch = tf.convert_to_tensor(random.choices(self.false_per_relation[r],k=self.b))
                    stack = tf.concat([S_batch, T_batch], axis=1)
                    stack_vectors = tf.map_fn(fn=lambda i: self.entity_to_index(i), elems=stack)
                    print()



    # def train(self):
    #     optimizer = self.optimizer()
    #     self.process_dataset()
    #     for epoch in range(self.epochs):
    #         start_time = time.perf_counter()
    #         self.e = tf.Variable(tf.nn.l2_normalize(self.e, axis=1), name="e", trainable=True, dtype=tf.float32)
    #         self.S = self.train_dataset.batch(self.b)
    #         for S_batch in tqdm(self.S):
    #             relations = tf.gather(tf.transpose(S_batch), 1)
    #             T_batch = tf.map_fn(self.get_relation_corrupted_triple, relations)
    #             stack = tf.concat([S_batch,T_batch],axis=1)
    #             print(stack)
    #             # print(T_batch)
    #             print("----")




            #     with tf.GradientTape() as tape:
            #         tape.watch(self.l)
            #         tape.watch(self.e)
            #         for triple in batch:
            #             h = str(triple.numpy()[0].decode('utf-8'))
            #             l = str(triple.numpy()[1].decode('utf-8'))
            #             t = str(triple.numpy()[2].decode('utf-8'))
            #             corrupted_triple = random.choice(self.relation_corrupt[l])
            #             h_dash = corrupted_triple[0]
            #             t_dash = corrupted_triple[2]
            #             h_vector = tf.nn.embedding_lookup(self.e, self.entities.index(h))
            #             t_vector = tf.nn.embedding_lookup(self.e, self.entities.index(t))
            #             h_dash_vector = tf.nn.embedding_lookup(self.e, self.entities.index(h_dash))
            #             t_dash_vector = tf.nn.embedding_lookup(self.e, self.entities.index(t_dash))
            #             l_vector = tf.nn.embedding_lookup(self.l, self.relations.index(l))
            #             self.loss = self.transe_loss(h_vector, h_dash_vector, t_vector, t_dash_vector, l_vector)
            #     gradients = tape.gradient(self.loss, [self.l, self.e])
            #     optimizer.apply_gradients(
            #         list(zip(gradients, [self.l, self.e])))
            #
            # end_time = time.perf_counter()
            # time_per_epoch = tf.constant(round((end_time - start_time), 3), dtype=tf.float32)
            # tf.print(" Epoch: " + tf.strings.as_string(epoch)
            #          + " Seconds/Epoch: " + tf.strings.as_string(time_per_epoch)
            #          + " Learning Rate: " + tf.strings.as_string(
            #     tf.constant(round(self.learning_rate, 3), dtype=tf.float32))
            #          + " Train Loss: " + tf.strings.as_string(self.loss)
            #          )





    #
    # def construct_corrupt_batch(self, batch):
    #
    #
    #
    # def vectorized_training(self):
    #     optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
    #     self.prepare_data()
    #     for epoch in tqdm(range(self.epochs)):
    #         start_time = time.perf_counter()
    #         self.e = tf.Variable(tf.nn.l2_normalize(self.e, axis=1), name="e", trainable=True, dtype=tf.float32)
    #         self.batches = self.train_dataset.batch(self.b)
    #         for batch in self.batches:
    #             # construct a corrupt batch
    #             with tf.GradientTape() as tape:
    #                 tape.watch(self.l)
    #                 tape.watch(self.e)


