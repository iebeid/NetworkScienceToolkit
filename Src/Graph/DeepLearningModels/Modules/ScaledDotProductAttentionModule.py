import numpy as np
import tensorflow as tf

class ScaledDotProductAttentionModule:
    def __init__(self, Q, K, V):
        self.Q=Q
        self.K=K
        self.V=V

    def operation(self):
        before_masking = tf.matmul(self.Q, tf.transpose(self.K))/tf.sqrt(self.Q.shape[1])
        masked_tensor = tf.where(before_masking < 0, -np.inf, before_masking)
        return tf.matmul(tf.nn.softmax(masked_tensor),self.V)
