import tensorflow as tf
from Src.Graph.DeepLearningModels.Modules.BatchNormalizationModule import BatchNormalizationModule
from Src.Graph.DeepLearningModels.Modules.LayerNormalizationModule import LayerNormalizationModule
from Src.Graph.DeepLearningModels.Modules.L2RegularizationModule import L2RegularizationModule


class GraphConvolutionLayer:
    def __init__(self, id, input, graph, size, train=True, embedding=False, dropout_rate=0.5, regularize=False,
                 regularization_rate=0.0005, residual=False, batch_norm=False, batch_norm_factor=0.001,
                 layer_norm=False, layer_norm_factor=0.00001, activation=None):
        self.input = input
        self.graph = graph.degree_normalized_adjacency_tensorflow()
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.activation = eval("tf.nn." + activation)
        self.id = id
        self.size = size
        self.train = train
        self.embedding = embedding
        self.dropout_rate = dropout_rate
        self.regularize = regularize
        self.regularization_rate = regularization_rate
        self.residual = residual
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.batch_norm_factor = batch_norm_factor
        self.layer_norm_factor = layer_norm_factor
        self.W = tf.Variable(self.initializer(shape=(int(self.input.shape.dims[1].value), int(self.size))), name="AW_" + str(self.id))
        self.b = tf.Variable(self.initializer(shape=(int(self.input.shape.dims[0].value), int(self.size))), name="Ab_" + str(self.id))
        self.M=None
        self.scale=None
        self.beta=None
        self.Z = None
        self.regularization_constant = 0
        if self.residual:
            self.M = tf.Variable(self.initializer(self.graph.N, self.size), name="M_" + str(self.id))
        if self.batch_norm:
            self.batch_norm_module = BatchNormalizationModule(self.Z, self.batch_norm_factor)
            self.scale = self.batch_norm_module.scale
            self.beta = self.batch_norm_module.beta
        if self.layer_norm:
            self.layer_norm_module = LayerNormalizationModule(self.Z, self.layer_norm_factor)
            self.scale = self.layer_norm_module.scale
            self.beta = self.layer_norm_module.beta

    def compute(self):
        if self.residual:
            self.residual_layer = tf.matmul(self.input, self.M)
        self.output = tf.add(tf.matmul(tf.matmul(self.graph, self.input), self.W), self.b)
        if self.batch_norm:
            bn_hidden_layer1 = self.batch_norm_module.compute()
            self.output = self.activation(bn_hidden_layer1)
        elif self.layer_norm:
            bn_hidden_layer1 = self.layer_norm_module.compute()
            self.output = self.activation(bn_hidden_layer1)
        else:
            self.output = self.activation(self.output)
        # Compute residual layer
        if self.residual:
            self.output = tf.add(self.output, self.residual_layer)
        # Compute l2 regularization for first layer only
        if self.regularize:
            l2_regularization_layer = L2RegularizationModule(self.regularization_rate, self.output)
            self.regularization_constant = l2_regularization_layer.operation()
        if self.embedding:
            self.node_embeddings = tf.nn.l2_normalize(self.output, axis=1)
        if self.train:
            # Dropout layer
            self.output = tf.nn.dropout(self.output, self.dropout_rate)
