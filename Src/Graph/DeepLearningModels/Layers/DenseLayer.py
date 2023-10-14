import tensorflow as tf


class DenseLayer:
    def __init__(self, id, input, size, activation="None", initializer="GlorotUniform()"):
        self.id=id
        self.input = input
        self.size=size
        try:
            self.initializer = eval("tf.keras.initializers." + initializer)
        except:
            self.initializer = None
        try:
            self.activation = eval("tf.nn." + activation)
        except:
            self.activation = None
        self.W = tf.Variable(self.initializer(shape=(int(self.input.shape.dims[1]), int(size))), name="W_" + str(self.id))
        self.b = tf.Variable(self.initializer(shape=(int(self.input.shape.dims[0]), int(size))), name="b_" + str(self.id))
        self.output = tf.Variable(self.initializer(shape=(int(self.input.shape.dims[0].value), int(self.size))),
                                  trainable=False)



    def compute(self):
        if not self.activation:
            self.output = tf.add(tf.matmul(self.input, self.W), self.b)
        else:
            self.output = self.activation(tf.add(tf.matmul(self.input, self.W), self.b))
