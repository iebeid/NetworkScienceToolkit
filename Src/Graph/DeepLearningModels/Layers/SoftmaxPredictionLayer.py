import tensorflow as tf


class SoftmaxPredictionLayer:
    def __init__(self, input):
        self.input = input

    def compute(self):
        self.output = tf.map_fn(fn=lambda i: tf.nn.softmax(i), elems=self.input)
