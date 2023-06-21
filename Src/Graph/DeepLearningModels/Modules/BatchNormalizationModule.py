import tensorflow as tf

class BatchNormalizationModule:
    def __init__(self, input, batch_normalization_factor):
        self.input = input
        self.scale = scale
        self.beta = beta
        self.batch_normalization_factor = batch_normalization_factor

    def operation(self):
        batch_mean1, batch_var1 = tf.nn.moments(self.input, [0])
        return tf.nn.sigmoid(
            self.scale * (self.input - batch_mean1) / tf.sqrt(batch_var1 + self.batch_normalization_factor) + self.beta)