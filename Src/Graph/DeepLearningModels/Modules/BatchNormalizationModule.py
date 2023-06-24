import tensorflow as tf

class BatchNormalizationModule:
    def __init__(self, input, batch_normalization_factor):
        self.input = input
        self.scale = tf.Variable(tf.ones([self.size]), name="scale_" + str(self.id))
        self.beta = tf.Variable(tf.zeros([self.size]), name="beta_" + str(self.id))
        self.batch_normalization_factor = batch_normalization_factor

    def compute(self):
        batch_mean1, batch_var1 = tf.nn.moments(self.input, [0])
        return tf.nn.sigmoid(
            self.scale * (self.input - batch_mean1) / tf.sqrt(batch_var1 + self.batch_normalization_factor) + self.beta)