import tensorflow as tf


class DenseLayer:
    def __init__(self, id, input, size, activation="relu", initializer="GlorotUniform"):
        self.input = input
        self.initializer = eval("tf.keras.initializers." + initializer)
        self.activation = eval("tf.nn." + activation)
        self.W = tf.Variable(self.initializer(shape=(int(self.input.shape[1]), size)), name="W_" + str(id))
        self.b = tf.Variable(self.initializer(shape=(int(self.input.shape[0]), size)), name="b_" + str(id))

    def compute(self):
        if self.activation is None:
            return tf.add(tf.matmul(self.input, self.W), self.b)
        else:
            return self.activation(tf.add(tf.matmul(self.input, self.W), self.b))
