import time
import numpy as np
import tensorflow as tf
from Src.Graph.Utils import AlgorithmicUtils
from Src.Graph.Utils import MathUtils
from Src.Graph.Utils import HierarchicalLabeler
from Src.Graph.DeepLearningModels.Layers.GraphConvolutionLayer import GraphConvolutionLayer
from Src.Graph.DeepLearningModels.Modules.BatchNormalizationModule import BatchNormalizationModule
from Src.Graph.DeepLearningModels.Modules.LayerNormalizationModule import LayerNormalizationModule
from Src.Graph.DeepLearningModels.Modules.L2RegularizationModule import L2RegularizationModule
from Src.Graph.DeepLearningModels.Layers.DenseLayer import DenseLayer
from Src.Graph.DeepLearningModels.Layers.SoftmaxPredictionLayer import SoftmaxPredictionLayer
from Src.Graph.DeepLearningModels.LossFunctions.RegularizedMaskedCrossEntropyLoss import RegularizedMaskedCrossEntropy
from Src.Graph.DeepLearningModels.EvaluationMethods.MaskedAccuracy import MaskedAccuracy

from numpy.random import seed

seed(1)
tf.random.set_seed(2)


class GCNModel:
    def __init__(self, graph, pararmeters):

        # Configuration
        self.dims = list(map(int, pararmeters["dims"].split(",")))
        self.regularization_rate = float(pararmeters["regularization_rate"])
        self.dropout_rate = float(pararmeters["dropout_rate"])
        self.activation = pararmeters["activation"]
        self.residual = eval(pararmeters["residual"])
        self.batch_norm = eval(pararmeters["batch_norm"])
        self.batch_norm_factor = float(pararmeters["batch_norm_factor"])
        self.layer_norm = eval(pararmeters["layer_norm"])
        self.layer_norm_epsilon = float(pararmeters["layer_norm_epsilon"])
        self.epochs = int(pararmeters["epochs"])
        self.learning_rate = float(pararmeters["learning_rate"])
        self.patience = int(pararmeters["patience"])
        self.train = eval(pararmeters["train"])
        self.split = int(pararmeters["split"])
        self.target = str(pararmeters["target"])

        # Model parameters
        self.model_parameters_dict = []

        # Input graph and node features
        self.graph = graph
        if self.target == "type":
            graph.prepare_graph_for_machine_learning_type(self.split)
            self.output_dim = len(self.graph.node_type_profile.keys())
        elif self.target == "label":
            graph.prepare_graph_for_machine_learning_label(self.split)
            self.output_dim = len(self.graph.node_label_profile.keys())
        if all(graph.node_features.values()):
            self.X = []
            for n, f in graph.node_features.items():
                self.X.append(eval(f))
            self.X = tf.Variable(tf.convert_to_tensor(np.array([np.array(xi) for xi in self.X]), dtype=tf.float32),
                                 name="X")
        else:
            initializer = tf.keras.initializers.Identity()
            self.X = tf.Variable(initializer((self.graph.N, self.dims[0])), name="X")
        # self.model_parameters_dict.append(self.X)

        # Model definition
        # GCN Layer 1
        self.Z_0 = GraphConvolutionLayer(0, self.X, self.graph, self.dims[0], embedding=False,
                                         dropout_rate=self.dropout_rate, regularize=True,
                                         regularization_rate=self.regularization_rate, residual=self.residual,
                                         batch_norm=self.batch_norm, batch_norm_factor=self.batch_norm_factor,
                                         layer_norm=self.layer_norm, layer_norm_factor=self.layer_norm_epsilon,
                                         activation=self.activation)
        if self.Z_0.W is not None:
            self.model_parameters_dict.append(self.Z_0.W)
        if self.Z_0.b is not None:
            self.model_parameters_dict.append(self.Z_0.b)
        if self.Z_0.M is not None:
            self.model_parameters_dict.append(self.Z_0.M)
        if self.Z_0.scale is not None:
            self.model_parameters_dict.append(self.Z_0.scale)
        if self.Z_0.beta is not None:
            self.model_parameters_dict.append(self.Z_0.beta)

        # GCN Layer 2
        self.Z_1 = GraphConvolutionLayer(1, self.Z_0.output, self.graph, self.dims[1], embedding=False,
                                         dropout_rate=self.dropout_rate, regularize=False,
                                         regularization_rate=self.regularization_rate, residual=self.residual,
                                         batch_norm=self.batch_norm, batch_norm_factor=self.batch_norm_factor,
                                         layer_norm=self.layer_norm, layer_norm_factor=self.layer_norm_epsilon,
                                         activation=self.activation)
        if self.Z_1.W is not None:
            self.model_parameters_dict.append(self.Z_1.W)
        if self.Z_1.b is not None:
            self.model_parameters_dict.append(self.Z_1.b)
        if self.Z_1.M is not None:
            self.model_parameters_dict.append(self.Z_1.M)
        if self.Z_1.scale is not None:
            self.model_parameters_dict.append(self.Z_1.scale)
        if self.Z_1.beta is not None:
            self.model_parameters_dict.append(self.Z_1.beta)

        # GCN Layer 3
        self.Z_2 = GraphConvolutionLayer(2, self.Z_1.output, self.graph, self.dims[2], embedding=True,
                                         dropout_rate=self.dropout_rate, regularize=False,
                                         regularization_rate=self.regularization_rate, residual=self.residual,
                                         batch_norm=self.batch_norm, batch_norm_factor=self.batch_norm_factor,
                                         layer_norm=self.layer_norm, layer_norm_factor=self.layer_norm_epsilon,
                                         activation=self.activation)
        if self.Z_2.W is not None:
            self.model_parameters_dict.append(self.Z_2.W)
        if self.Z_2.b is not None:
            self.model_parameters_dict.append(self.Z_2.b)
        if self.Z_2.M is not None:
            self.model_parameters_dict.append(self.Z_2.M)
        if self.Z_2.scale is not None:
            self.model_parameters_dict.append(self.Z_2.scale)
        if self.Z_2.beta is not None:
            self.model_parameters_dict.append(self.Z_2.beta)

        # Fully Connected Layer 4
        self.logits = DenseLayer(3, self.Z_2.output, self.output_dim, activation="None", initializer="GlorotUniform()")
        if self.logits.W is not None:
            self.model_parameters_dict.append(self.logits.W)
        if self.logits.b is not None:
            self.model_parameters_dict.append(self.logits.b)

        # Output Prediction Layer
        self.predictions = SoftmaxPredictionLayer(self.logits.output)

    def compile_model(self, train):
        # -----------------------------------------------------------------------------------------------
        # GCN Layer 1
        self.Z_0.compute(train)
        self.regularization_constant = self.Z_0.regularization_constant
        # -----------------------------------------------------------------------------------------------
        # GCN Layer 2
        self.Z_1.compute(train)
        # -----------------------------------------------------------------------------------------------
        # GCN Layer 3
        self.Z_2.compute(train)
        # -----------------------------------------------------------------------------------------------
        # Fully Connected Layer 4
        self.logits.compute()
        # -----------------------------------------------------------------------------------------------
        # Output prediction
        self.predictions.compute()
        # -----------------------------------------------------------------------------------------------

    # @tf.function
    def train_model(self):
        wait = 0
        best = 0.0
        for epoch in tf.range(self.epochs):
            start_time = time.perf_counter()
            with tf.GradientTape() as tape:
                self.compile_model(True)
                self.train_loss = RegularizedMaskedCrossEntropy(self.graph.y, self.predictions.output,
                                                                self.graph.train_mask,
                                                                self.regularization_constant).compute()
                self.train_accuracy = MaskedAccuracy(self.graph.y, self.predictions.output,
                                                     self.graph.train_mask).compute()
            self.gradients = tape.gradient(self.train_loss, self.model_parameters_dict)
            tf.keras.optimizers.Adam(learning_rate=self.learning_rate).apply_gradients(
                list(zip(self.gradients, self.model_parameters_dict)))
            self.compile_model(False)
            self.valid_loss = RegularizedMaskedCrossEntropy(self.graph.y, self.predictions.output,
                                                            self.graph.valid_mask,
                                                            self.regularization_constant).compute()
            self.valid_accuracy = MaskedAccuracy(self.graph.y, self.predictions.output, self.graph.valid_mask).compute()
            end_time = time.perf_counter()
            time_per_epoch = tf.constant(round((end_time - start_time), 3), dtype=tf.float32)
            tf.print(" Epoch: " + tf.strings.as_string(epoch)
                     + " Seconds/Epoch: " + tf.strings.as_string(time_per_epoch)
                     + " Learning Rate: " + tf.strings.as_string(
                tf.constant(round(self.learning_rate, 3), dtype=tf.float32))
                     + " Train Loss: " + tf.strings.as_string(self.train_loss)
                     + " Train Accuracy: " + tf.strings.as_string(self.train_accuracy)
                     + " Valid Loss: " + tf.strings.as_string(self.valid_loss)
                     + " Valid Accuracy: " + tf.strings.as_string(self.valid_accuracy)
                     )
            # Early stopping--------------------------------------------------------
            wait += 1
            if tf.greater(self.valid_accuracy, best):
                best = self.valid_accuracy
                wait = 0
            if tf.greater_equal(wait, self.patience):
                break

    def test_model(self):
        # Test on a split----------------------------------------------------------------------------
        self.compile_model(False)
        test_loss = RegularizedMaskedCrossEntropy(self.graph.y, self.predictions.output, self.graph.test_mask,
                                                  self.regularization_constant).compute()
        test_accuracy = MaskedAccuracy(self.graph.y, self.predictions.output, self.graph.test_mask).compute()
        print("Final Test Loss: " + str(test_loss.numpy()))
        print("Final Test Accuracy: " + str(test_accuracy.numpy()))

    def save_model(self, embedding_file, metadata_file):
        np.savetxt(embedding_file, list(self.Z_2.node_embeddings.numpy()), delimiter='\t', fmt='%f')
        np.savetxt(metadata_file, list(self.graph.nodes.keys()), delimiter='\t', fmt='%s')
