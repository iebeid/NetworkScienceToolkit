import tensorflow as tf


class GraphConvolutionLayer:
    def __init__(self, id, input, graph, size, activation="tanh", initializer="GlorotUniform"):
        self.input = input
        self.graph = graph.degree_normalized_adjacency_tensorflow()
        self.initializer = eval("tf.keras.initializers." + initializer)
        self.activation = eval("tf.nn." + activation)
        self.W = tf.Variable(self.initializer(shape=(int(self.input.shape[1]), size)), name="AW_" + str(id))
        self.b = tf.Variable(self.initializer(shape=(int(self.input.shape[0]), size)), name="Ab_" + str(id))

    def compute(self):
        if self.activation is None:
            return tf.add(tf.matmul(tf.matmul(self.graph, self.input), self.W), self.b)
        else:
            return self.activation(tf.add(tf.matmul(tf.matmul(self.graph, self.input), self.W), self.b))