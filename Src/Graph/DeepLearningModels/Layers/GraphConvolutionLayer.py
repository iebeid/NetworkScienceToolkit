import tensorflow as tf
from Src.Graph.DeepLearningModels.Modules.BatchNormalizationModule import BatchNormalizationModule
from Src.Graph.DeepLearningModels.Modules.LayerNormalizationModule import LayerNormalizationModule
from Src.Graph.DeepLearningModels.Modules.L2RegularizationModule import L2RegularizationModule


class GraphConvolutionLayer:
    def __init__(self, id, input, graph, size, embedding=False, dropout_rate=0.5, regularize=False,
                 regularization_rate=0.0005, residual=False, batch_norm=False, batch_norm_factor=0.001,
                 layer_norm=False, layer_norm_factor=0.00001, activation=None):
        self.input = input
        self.graph = graph.degree_normalized_adjacency_tensorflow()
        self.N = int(graph.N)
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.activation = eval("tf.nn." + activation)
        self.id = id
        self.size = int(size)
        self.embedding = embedding
        self.dropout_rate = dropout_rate
        self.regularize = regularize
        self.regularization_rate = regularization_rate
        self.residual = residual
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.batch_norm_factor = batch_norm_factor
        self.layer_norm_factor = layer_norm_factor
        self.W = tf.convert_to_tensor(tf.Variable(self.initializer(shape=(int(self.input.shape.dims[1].value), int(self.size))), name="AW_" + str(self.id), trainable=True))
        self.b = tf.Variable(self.initializer(shape=(int(self.input.shape.dims[0].value), int(self.size))), name="Ab_" + str(self.id), trainable=True)
        self.M=None
        self.scale=None
        self.beta=None
        self.output = tf.Variable(self.initializer(shape=(int(self.input.shape.dims[0].value), int(self.size))), trainable=False)
        self.regularization_constant = 0
        if self.residual:
            self.M = tf.Variable(self.initializer(shape=(int(self.size), int(self.size))), name="M_" + str(self.id) , trainable=True)
        if self.batch_norm:
            self.batch_norm_module = BatchNormalizationModule(self.id, self.output, self.batch_norm_factor)
            self.scale = self.batch_norm_module.scale
            self.beta = self.batch_norm_module.beta
        if self.layer_norm:
            self.layer_norm_module = LayerNormalizationModule(self.id, self.output, self.layer_norm_factor)
            self.scale = self.layer_norm_module.scale
            self.beta = self.layer_norm_module.beta

    def compute(self, train):
        if self.residual:
            self.residual_layer = tf.matmul(self.input, self.M)
        self.output = tf.add(tf.matmul(tf.matmul(self.graph, self.input), self.W), self.b)
        if self.batch_norm:
            self.output = self.activation(self.batch_norm_module.compute())
        elif self.layer_norm:
            self.output = self.activation(self.layer_norm_module.compute())
        else:
            self.output = self.activation(self.output)
        # Compute residual layer
        if self.residual:
            self.output = tf.add(self.output, self.residual_layer)
        # Compute l2 regularization for first layer only
        if self.regularize:
            self.regularization_constant = L2RegularizationModule(self.regularization_rate, self.output).compute()
        if self.embedding:
            self.node_embeddings = tf.nn.l2_normalize(self.output, axis=1)
        if train:
            # Dropout layer
            self.output = tf.nn.dropout(self.output, self.dropout_rate)
