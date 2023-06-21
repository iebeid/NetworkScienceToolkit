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

from numpy.random import seed

seed(1)
tf.random.set_seed(2)

class GCNModel:
    def __init__(self, graph, pararmeters):
        self.graph=graph
        self.dims = list(pararmeters["dims"].split(","))
        self.regularization_rate = float(pararmeters["regularization_rate"])
        self.dropout_rate = float(pararmeters["dropout_rate"])
        self.activation = eval("tf.nn." + pararmeters["activation"])
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

        if self.target == "type":
            graph.prepare_graph_for_machine_learning_type(self.split)
            self.output_dim = len(self.graph.node_type_profile.keys())
        elif self.target=="label":
            graph.prepare_graph_for_machine_learning_label(self.split)
            self.output_dim = len(self.graph.node_label_profile.keys())

        self.model_parameters_dict = {}
        self.model_parameters_list = []

        # Input node features
        if all(graph.node_features.values()):
            X = []
            for n, f in graph.node_features.items():
                X.append(eval(f))
            X = tf.Variable(tf.convert_to_tensor(np.array([np.array(xi) for xi in X]), dtype=tf.float32), name="X")

        else:
            X = tf.Variable(self.identity_initializer(self.graph.N, self.graph.N), name="X")

        # Training tensors
        self.gradients = tf.Variable(self.zero_initializer(X.shape[0], X.shape[1]), name="gradients")
        self.train_loss = tf.Variable(self.zero_initializer(1, 1), name="train_loss")
        self.train_accuracy = tf.Variable(self.zero_initializer(1, 1), name="train_accuracy")
        self.valid_loss = tf.Variable(self.zero_initializer(1, 1), name="valid_loss")
        self.valid_accuracy = tf.Variable(self.zero_initializer(1, 1), name="valid_accuracy")
        self.model_parameters_dict["input"] = X
        self.model_parameters_list.append(X)

        # GCN Layers
        previous_C = X.shape[1]
        for i, hd in enumerate(self.dims):
            self.F = int(hd)
            W = tf.Variable(self.glorot_intializer(previous_C, self.F), name="W_" + str(i))
            b = tf.Variable(self.glorot_intializer(self.graph.N, self.F), name="b_" + str(i))
            self.model_parameters_dict[i] = {"W": W, "b": b}
            self.model_parameters_list.append(W)
            self.model_parameters_list.append(b)
            if self.residual:
                M = tf.Variable(self.glorot_intializer(self.graph.N, self.F), name="M_" + str(i))
                self.model_parameters_dict[i] = {"W": W, "b": b, "M": M}
                self.model_parameters_list.append(M)
            if self.batch_norm or self.layer_norm:
                scale = tf.Variable(tf.ones([self.F]), name="scale_" + str(i))
                beta = tf.Variable(tf.zeros([self.F]), name="beta_" + str(i))
                self.model_parameters_dict[i] = {"W": W, "b": b, "M": M, "scale": scale, "beta": beta}
                self.model_parameters_list.append(scale)
                self.model_parameters_list.append(beta)
            previous_C = self.F

        # Fully connected layer 4
        fully_connected_W = tf.Variable(self.glorot_intializer(previous_C, self.output_dim), name="fully_connected_W")
        fully_connected_b = tf.Variable(self.glorot_intializer(self.graph.N, self.output_dim), name="fully_connected_b")

        self.model_parameters_dict["output"] = {"W": fully_connected_W, "b": fully_connected_b}
        self.model_parameters_list.append(fully_connected_W)
        self.model_parameters_list.append(fully_connected_b)

    def glorot_intializer(self, in_d, out_d):
        init = tf.keras.initializers.GlorotUniform()
        return init(shape=(in_d, out_d))

    def identity_initializer(self, in_d, out_d):
        init = tf.keras.initializers.Identity()
        return init(shape=(in_d, out_d))

    def zero_initializer(self, in_d, out_d):
        init = tf.keras.initializers.Zeros()
        return init(shape=(in_d, out_d))

    # Loss--------------------------------------------------------------------------
    def regularized_masked_cross_entropy_loss(self, mask):
        return (-tf.reduce_mean(
            tf.reduce_sum(tf.boolean_mask(self.graph.y, mask) * tf.math.log(
                tf.boolean_mask(self.predictions, mask))))) + self.regularization_constant

    # Optimizer------------------------------------------------------------------------
    def optimizer(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        return optimizer

    # Evaluation from Kipf et al 2017-------------------------------------------------------
    def masked_accuracy(self, mask):
        correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.graph.y, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        accuracy = tf.reduce_mean(accuracy_all)
        return accuracy

    def gcn_layer(self, input, l, regularize, embedding):
        # GCN Layer
        l = l - 1
        gc = GraphConvolutionLayer(input, self.graph, self.model_parameters_dict[l]["W"], self.model_parameters_dict[l]["b"])
        # Residual layer
        if self.residual:
            self.residual_layer = tf.matmul(input, self.model_parameters_dict[l]["M"])
        Z = gc.compute()
        # Activation layer
        # Batch normalization layer
        if self.batch_norm:
            batch_normalization_layer = BatchNormalizationModule(Z, self.model_parameters_dict[l]["scale"],
                                                                self.model_parameters_dict[l]["beta"],
                                                                self.batch_norm_factor)
            bn_hidden_layer1 = batch_normalization_layer.operation()
            Z = self.activation(bn_hidden_layer1)
        elif self.layer_norm:
            batch_normalization_layer = LayerNormalizationModule(Z, self.layer_norm_epsilon)
            bn_hidden_layer1 = batch_normalization_layer.operation()
            Z = self.activation(bn_hidden_layer1)
        else:
            Z = self.activation(Z)
        # Residual layer
        if self.residual:
            Z = tf.add(Z, self.residual_layer)
        # Compute l2 regularization for first layer only
        if regularize:
            l2_regularization_layer = L2RegularizationModule(self.regularization_rate, Z)
            self.regularization_constant = l2_regularization_layer.operation()
        if embedding:
            self.node_embeddings = tf.nn.l2_normalize(Z, axis=1)
        if self.train:
            # Dropout layer
            Z = tf.nn.dropout(Z, self.dropout_rate)
        return Z

    def compile_model(self):
        # GCN Layer 1
        Z_0 = self.gcn_layer(self.model_parameters_dict["input"], 1, True, False)
        # -----------------------------------------------------------------------------------------------
        # GCN Layer 2
        Z_1 = self.gcn_layer(Z_0, 2, False, False)
        # -----------------------------------------------------------------------------------------------
        # GCN Layer 3
        Z_2 = self.gcn_layer(Z_1, 3, False, True)
        # -----------------------------------------------------------------------------------------------
        # Fully Connected Layer 4
        self.logits = DenseLayer(Z_2, self.model_parameters_dict["output"]["W"],
                                 self.model_parameters_dict["output"]["b"], tf.nn.relu).compute()
        # -----------------------------------------------------------------------------------------------
        # Output prediction
        self.predictions = tf.map_fn(fn=lambda i: tf.nn.softmax(i), elems=self.logits)
        # -----------------------------------------------------------------------------------------------

    @tf.function
    def train_model(self):
        wait = 0
        best = 0.0
        for epoch in tf.range(self.epochs):
            start_time = time.perf_counter()
            with tf.GradientTape() as tape:
                self.compile_model()
                self.train_loss = self.regularized_masked_cross_entropy_loss(self.graph.train_mask)
                self.train_accuracy = self.masked_accuracy(self.graph.train_mask)
            self.gradients = tape.gradient(self.train_loss, self.model_parameters_list)
            self.optimizer().apply_gradients(list(zip(self.gradients, self.model_parameters_list)))
            self.compile_model()
            self.valid_loss = self.regularized_masked_cross_entropy_loss(self.graph.valid_mask)
            self.valid_accuracy = self.masked_accuracy(self.graph.valid_mask)
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
        self.compile_model()
        test_loss = self.regularized_masked_cross_entropy_loss(self.graph.test_mask)
        test_accuracy = self.masked_accuracy(self.graph.test_mask)
        print("Final Test Loss: " + str(test_loss.numpy()))
        print("Final Test Accuracy: " + str(test_accuracy.numpy()))

    def save_model(self, embedding_file, metadata_file):
        np.savetxt(embedding_file, list(self.node_embeddings.numpy()), delimiter='\t', fmt='%f')
        np.savetxt(metadata_file, list(self.graph.nodes.keys()), delimiter='\t', fmt='%s')
