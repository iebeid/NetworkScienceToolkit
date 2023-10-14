from Src.Graph.DeepLearningModels.Layers.DenseLayer import DenseLayer


class FFN:

    def __init__(self, input, size, activation):
        self.input = input
        self.size = size
        self.activation = activation
        self.hidden = DenseLayer(1, self.input, self.size, activation="relu", initializer="GlorotUniform")
        self.output = DenseLayer(2, self.hidden, self.size, activation="none", initializer="GlorotUniform")

    def compute(self):
        self.hidden.compute()
        return self.output.compute()
