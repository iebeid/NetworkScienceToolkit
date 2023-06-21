import tensorflow as tf
from Src.Graph.DeepLearningModels.Modules.ScaledDotProductAttentionModule import ScaledDotProductAttentionModule


class MultiheadAttentionLayer:
    def __init__(self, Q, K, V, h, d_model, initializer="GlorotUniform"):
        self.initializer = eval("tf.keras.initializers." + initializer)
        self.Q = Q
        self.K = K
        self.V = V
        self.h = h
        self.d_model = d_model
        self.WQ = []
        self.WK = []
        self.WV = []
        for i in range(self.h):
            WQ = tf.Variable(self.initializer(shape=(self.d_model, int(self.K.shape[1]))), name="WQ_" + str(i))
            self.WQ.append(WQ)
        for i in range(self.h):
            WK = tf.Variable(self.initializer(shape=(self.d_model, int(self.K.shape[1]))), name="WK_" + str(i))
            self.WK.append(WK)
        for i in range(self.h):
            WV = tf.Variable(self.initializer(shape=(self.d_model, int(self.V.shape[1]))), name="WV_" + str(i))
            self.WV.append(WV)
        self.WO = tf.Variable(self.initializer(shape=(int(self.V.shape[1]) * self.h, self.d_model), name="WO_0"))

    def operation(self):
        sdpamts = []
        for i in range(self.h):
            sdpam = ScaledDotProductAttentionModule(tf.matmul(self.Q, self.WQ[i]), tf.matmul(self.K, self.WK[i]),
                                                    tf.matmul(self.V, self.WV[i]))
            head = sdpam.operation()
            sdpamts.append(head)
        return tf.matmul(tf.concat(sdpamts, axis=0), self.WO)
