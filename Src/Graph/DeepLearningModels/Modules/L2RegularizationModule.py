import tensorflow as tf

class L2RegularizationModule:
    def __init__(self, rate, input):
        self.rate = rate
        self.input = input

    def operation(self):
        r = tf.keras.regularizers.L2(self.rate)
        return r(self.input)