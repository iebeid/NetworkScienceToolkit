import tensorflow as tf

class MaskedAccuracy:
    def __init__(self, labels, predictions, mask):
        self.labels=labels
        self.predictions=predictions
        self.mask=mask

    # Evaluation from Kipf et al 2017
    def compute(self):
        correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(self.mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        accuracy = tf.reduce_mean(accuracy_all)
        return accuracy
