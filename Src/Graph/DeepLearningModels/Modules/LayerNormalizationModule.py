import tensorflow as tf


class LayerNormalizationModule:
    def __init__(self, input, epsilon):
        self.input = input
        self.scale = scale
        self.beta = beta
        self.epsilon = epsilon

    def operation(self):
        mean, var = tf.nn.moments(self.input, [0])
        return (self.input - mean) / tf.sqrt(var + self.epsilon) * self.scale + self.beta