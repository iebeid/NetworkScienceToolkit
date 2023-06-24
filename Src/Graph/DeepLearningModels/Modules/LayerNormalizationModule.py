import tensorflow as tf


class LayerNormalizationModule:
    def __init__(self, input, epsilon):
        self.input = input
        self.scale = tf.Variable(tf.ones([self.size]), name="scale_" + str(self.id))
        self.beta = tf.Variable(tf.zeros([self.size]), name="beta_" + str(self.id))
        self.epsilon = epsilon

    def compute(self):
        mean, var = tf.nn.moments(self.input, [0])
        return (self.input - mean) / tf.sqrt(var + self.epsilon) * self.scale + self.beta