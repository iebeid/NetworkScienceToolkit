import tensorflow as tf


class LayerNormalizationModule:
    def __init__(self, id, input, epsilon):
        self.id = id
        self.input = input
        self.scale = tf.Variable(tf.ones([int(self.input.shape.dims[0].value),1]), name="scale_" + str(self.id), trainable=True)
        self.beta = tf.Variable(tf.zeros([int(self.input.shape.dims[0].value),1]), name="beta_" + str(self.id), trainable=True)
        self.epsilon = epsilon

    def compute(self):
        mean, var = tf.nn.moments(self.input, [0])
        operation1=self.input - mean
        operation2=tf.sqrt(var + self.epsilon)
        operation3=operation1/operation2
        operation4=operation3 * self.scale
        operation5=operation4 + self.beta
        return operation5