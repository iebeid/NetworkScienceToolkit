import tensorflow as tf


class RegularizedMaskedCrossEntropy:
    def __init__(self, labels, predictions, mask, regularization_constant):
        self.labels = labels
        self.mask = mask
        self.predictions = predictions
        self.regularization_constant = regularization_constant

    def compute(self):
        return (-tf.reduce_mean(
            tf.reduce_sum(tf.boolean_mask(self.labels, self.mask) * tf.math.log(
                tf.boolean_mask(self.predictions, self.mask))))) + self.regularization_constant
