from Src.Graph.DeepLearningModels.Layers.DenseLayer import DenseLayer


class FFN:

    def __init__(self, input, size, activation):
        self.input = input
        self.size = size
        self.activation = activation
        self.layer1 = DenseLayer(1, self.input, self.size, activation="relu", initializer="GlorotUniform")
        self.layer2 = DenseLayer(2, self.layer1, self.size, activation="none", initializer="GlorotUniform")

    def compute(self):
        self.layer1.compute()
        return self.layer2.compute()
