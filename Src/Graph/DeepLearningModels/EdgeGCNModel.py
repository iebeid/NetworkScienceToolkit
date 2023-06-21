import time
import numpy as np
import tensorflow as tf
from Src.Data.KarateClubNetwork import KarateClubNetwork
import Utils
from Modules import GraphConvolutionModule
from Modules import BatchNormalizationModule
from Modules import L2RegularizationModule
from Layers import DenseLayer
from numpy.random import seed

seed(1)
tf.random.set_seed(1)


class EdgeGCNModel:
    def __init__(self, graph, dims,
                 activation, residual, patience, epochs, split, train,
                 regularization_rate, dropout_rate, learning_rate,
                 batch_normalization, batch_normalization_factor):
        self.graph = graph
        self.edge_graph = graph.create_second_degree_graph()

        self.graph.prepare_graph_for_machine_learning_label(split)
        self.edge_graph.prepare_graph_for_machine_learning_label(split)

        self.train_mask1 = self.graph.train_mask
        self.valid_mask1 = self.graph.valid_mask
        self.test_mask1 = self.graph.test_mask
        self.train_mask2 = self.edge_graph.train_mask
        self.valid_mask2 = self.edge_graph.valid_mask
        self.test_mask2 = self.edge_graph.test_mask

        self.activation = activation
        self.residual = residual
        self.regularization_rate = regularization_rate
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_normalization = batch_normalization
        self.batch_normalization_factor = batch_normalization_factor
        self.patience = patience
        self.epochs = epochs
        self.train = train

        self.model_parameters_dict = {}
        self.model_parameters_list1 = []
        self.model_parameters_list2 = []

        input_dim = self.graph.N
        output_dim_1 = len(self.graph.node_label_profile.keys())
        output_dim_2 = len(self.edge_graph.node_label_profile.keys())
        # Input node features
        if all(graph.node_features.values()):
            X = []
            for n, f in graph.node_features.items():
                X.append(f)
            X = tf.convert_to_tensor(np.array([np.array(xi) for xi in X]), dtype=tf.float32)
        else:
            X = tf.Variable(self.identity_initializer(self.graph.N, input_dim), name="X")
        self.model_parameters_dict["input"] = X
        self.model_parameters_list1.append(X)
        # GCN Layers
        previous_hd = input_dim
        count = 0
        for i, hd in enumerate(dims):
            W = tf.Variable(self.glorot_intializer(previous_hd, hd), name="W_" + str(i))
            b = tf.Variable(self.glorot_intializer(self.graph.N, hd), name="b_" + str(i))
            M = tf.Variable(self.glorot_intializer(previous_hd, hd), name="M_" + str(i))
            scale = tf.Variable(tf.ones([hd]), name="scale_" + str(i))
            beta = tf.Variable(tf.zeros([hd]), name="beta_" + str(i))

            self.model_parameters_dict[i] = {"W": W, "b": b, "M": M, "scale": scale, "beta": beta}

            self.model_parameters_list1.append(W)
            self.model_parameters_list1.append(b)
            self.model_parameters_list1.append(M)
            if self.batch_normalization:
                self.model_parameters_list1.append(scale)
                self.model_parameters_list1.append(beta)
            previous_hd = hd
            count = i
        count = count + 1
        for j, hd in enumerate(dims):

            W = tf.Variable(self.glorot_intializer(previous_hd, hd), name="W_" + str(count + j))
            b = tf.Variable(self.glorot_intializer(self.edge_graph.N, hd), name="b_" + str(count + j))
            M = tf.Variable(self.glorot_intializer(previous_hd, hd), name="M_" + str(count + j))
            scale = tf.Variable(tf.ones([hd]), name="scale_" + str(count + j))
            beta = tf.Variable(tf.zeros([hd]), name="beta_" + str(count + j))

            self.model_parameters_dict[count + j] = {"W": W, "b": b, "M": M, "scale": scale, "beta": beta}

            self.model_parameters_list2.append(W)
            self.model_parameters_list2.append(b)
            self.model_parameters_list2.append(M)
            if self.batch_normalization:
                self.model_parameters_list2.append(scale)
                self.model_parameters_list2.append(beta)
            previous_hd = hd
            # count = count + 1
        # Fully connected layer 4
        fully_connected_W1 = tf.Variable(self.glorot_intializer(previous_hd, output_dim_1), name="fully_connected_W1")
        fully_connected_b1 = tf.Variable(self.glorot_intializer(self.graph.N, output_dim_1), name="fully_connected_b1")

        self.model_parameters_dict["output1"] = {"W": fully_connected_W1, "b": fully_connected_b1}
        self.model_parameters_list1.append(fully_connected_W1)
        self.model_parameters_list1.append(fully_connected_b1)

        fully_connected_W2 = tf.Variable(self.glorot_intializer(previous_hd, output_dim_2), name="fully_connected_W2")
        fully_connected_b2 = tf.Variable(self.glorot_intializer(self.edge_graph.N, output_dim_2),
                                         name="fully_connected_b2")

        self.model_parameters_dict["output2"] = {"W": fully_connected_W2, "b": fully_connected_b2}
        self.model_parameters_list2.append(fully_connected_W2)
        self.model_parameters_list2.append(fully_connected_b2)

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
    def regularized_masked_cross_entropy_loss(self, graph, predictions, r, mask):
        return (-tf.reduce_mean(
            tf.reduce_sum(tf.boolean_mask(graph.y, mask) * tf.math.log(
                tf.boolean_mask(predictions, mask))))) + r

    # Optimizer------------------------------------------------------------------------
    def optimizer(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        return optimizer

    # Evaluation from Kipf et al 2017-------------------------------------------------------
    def masked_accuracy(self, graph, predictions, mask):
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(graph.y, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        accuracy = tf.reduce_mean(accuracy_all)
        return accuracy

    def gcn_layer(self, input, graph, l, regularize, embedding):
        # GCN Layer
        l = l - 1
        gc = GraphConvolutionModule(input, graph, self.model_parameters_dict[l]["W"], self.model_parameters_dict[l]["b"])
        # Residual layer
        if self.residual:
            self.residual_layer = tf.matmul(input, self.model_parameters_dict[l]["M"])
        Z = gc.operation()
        # Activation layer
        # Batch normalization layer
        if self.batch_normalization:
            Z = self.activation(BatchNormalizationModule(Z, self.model_parameters_dict[l]["scale"],
                                                                self.model_parameters_dict[l]["beta"],
                                                                self.batch_normalization_factor).operation())
        else:
            Z = self.activation(Z)
        # Residual layer
        if self.residual:
            Z = tf.add(Z, self.residual_layer)
        # Compute l2 regularization for first layer only
        if regularize:
            self.regularization_constant = L2RegularizationModule(self.regularization_rate, Z).operation()
        if embedding:
            self.node_embeddings = tf.nn.l2_normalize(Z, axis=1)
        if self.train:
            # Dropout layer
            Z = tf.nn.dropout(Z, self.dropout_rate)
        return Z

    def compile_node_model(self):
        # Edge Graph Convolutional Neural Networks (E-GCN)
        # -----------------------------------------------------------------------------------------------
        # is a more pronounced pipeline that does
        # not only aggregate information from the first order node
        # neighborhood through the fast approximation of the convolution
        # operator on the graph signal
        # but also through agregating information on the edge neighborhood
        # which can also sometimes amount for 3 to 4 order neighborhood
        # the intention here is to learn simultanously edge embeddings
        # and if extended the model would be capable of learning motif embeddings
        # -----------------------------------------------------------------------------------------------
        # Node Information Aggregator
        # GCN Module 1
        Z_0 = self.gcn_layer(self.model_parameters_dict["input"], self.graph, 1, True, False)
        self.regularization_constant1 = self.regularization_constant
        # -----------------------------------------------------------------------------------------------
        # GCN Module 2
        Z_1 = self.gcn_layer(Z_0, self.graph, 2, False, False)
        # -----------------------------------------------------------------------------------------------
        # GCN Module 3
        Z_2 = self.gcn_layer(Z_1, self.graph, 3, False, True)
        # -----------------------------------------------------------------------------------------------
        self.final_node_embeddings = self.node_embeddings
        # Fully Connected Layer 4
        dense_layer = DenseLayer(Z_2, self.model_parameters_dict["output1"]["W"],
                                 self.model_parameters_dict["output1"]["b"], tf.nn.relu)
        self.logits = dense_layer.layer()
        # -----------------------------------------------------------------------------------------------
        # Output layers
        self.predictions1 = tf.map_fn(fn=lambda i: tf.nn.softmax(i), elems=self.logits)
        # -----------------------------------------------------------------------------------------------
    def compile_edge_model(self):
        # Edge Information Aggregator
        # Edge Embedding Lookup Step
        edge_input_features = []
        for e in self.graph.edges:
            s_n = self.graph.node_index[e[0]]
            t_n = self.graph.node_index[e[1]]
            if s_n < t_n:
                pair = tf.nn.embedding_lookup(self.final_node_embeddings, [s_n, t_n], max_norm=None, name=None)
                edge_vector = np.subtract(pair[0].numpy(), pair[1].numpy())
                edge_input_features.append(edge_vector)
        edge_input_features = tf.convert_to_tensor(np.array(edge_input_features), dtype=tf.float32)
        # -----------------------------------------------------------------------------------------------
        # GCN Module 4
        Z_0_edge = self.gcn_layer(edge_input_features, self.edge_graph, 4, True, False)
        self.regularization_constant2 = self.regularization_constant
        # -----------------------------------------------------------------------------------------------
        # GCN Module 5
        Z_1_edge = self.gcn_layer(Z_0_edge, self.edge_graph, 5, False, False)
        # -----------------------------------------------------------------------------------------------
        # GCN Module 6
        Z_2_edge = self.gcn_layer(Z_1_edge, self.edge_graph, 6, False, True)
        # -----------------------------------------------------------------------------------------------
        self.final_edge_embeddings = self.node_embeddings
        # Fully Connected Layer 4
        dense_layer_2 = DenseLayer(Z_2_edge, self.model_parameters_dict["output2"]["W"],
                                   self.model_parameters_dict["output2"]["b"], tf.nn.relu)
        self.logits2 = dense_layer_2.layer()
        # -----------------------------------------------------------------------------------------------
        # Output layers
        self.predictions2 = tf.map_fn(fn=lambda i: tf.nn.softmax(i), elems=self.logits2)
        # -----------------------------------------------------------------------------------------------


    def train_model(self):
        wait = 0
        best = 0.0
        for epoch in tf.range(self.epochs):
            start_time = time.perf_counter()
            with tf.GradientTape() as tape1:
                self.compile_node_model()
                self.train_loss1 = self.regularized_masked_cross_entropy_loss(self.graph, self.predictions1,
                                                                              self.regularization_constant1,
                                                                              self.train_mask1)
                self.train_accuracy1 = self.masked_accuracy(self.graph, self.predictions1, self.train_mask1)
            with tf.GradientTape() as tape2:
                self.compile_edge_model()
                self.train_loss2 = self.regularized_masked_cross_entropy_loss(self.edge_graph, self.predictions2,
                                                                              self.regularization_constant2,
                                                                              self.train_mask2)
                self.train_accuracy2 = self.masked_accuracy(self.edge_graph, self.predictions2, self.train_mask2)
            self.gradients1 = tape1.gradient(self.train_loss1, self.model_parameters_list1)
            self.gradients2 = tape2.gradient(self.train_loss2, self.model_parameters_list2)
            self.optimizer().apply_gradients(list(zip(self.gradients1, self.model_parameters_list1)))
            self.optimizer().apply_gradients(list(zip(self.gradients2, self.model_parameters_list2)))
            self.compile_node_model()
            self.compile_edge_model()
            self.valid_loss1 = self.regularized_masked_cross_entropy_loss(self.graph, self.predictions1,
                                                                              self.regularization_constant1,
                                                                              self.valid_mask1)
            self.valid_accuracy1 = self.masked_accuracy(self.graph, self.predictions1, self.valid_mask1)
            self.valid_loss2 = self.regularized_masked_cross_entropy_loss(self.edge_graph, self.predictions2,
                                                                              self.regularization_constant2,
                                                                              self.valid_mask2)
            self.valid_accuracy2 = self.masked_accuracy(self.edge_graph, self.predictions2, self.valid_mask2)
            end_time = time.perf_counter()
            time_per_epoch = tf.constant(round((end_time - start_time), 3), dtype=tf.float32)
            tf.print(" Epoch: " + tf.strings.as_string(epoch)
                     + " Seconds/Epoch: " + tf.strings.as_string(time_per_epoch)
                     + " Learning Rate: " + tf.strings.as_string(
                tf.constant(round(self.learning_rate, 3), dtype=tf.float32))
                     + " Train Loss 1: " + tf.strings.as_string(self.train_loss1)
                     + " Train Accuracy 1: " + tf.strings.as_string(self.train_accuracy1)
                     + " Valid Loss 1: " + tf.strings.as_string(self.valid_loss1)
                     + " Valid Accuracy 1: " + tf.strings.as_string(self.valid_accuracy1)
                     + " Train Loss 2: " + tf.strings.as_string(self.train_loss2)
                     + " Train Accuracy 2: " + tf.strings.as_string(self.train_accuracy2)
                     + " Valid Loss 2: " + tf.strings.as_string(self.valid_loss2)
                     + " Valid Accuracy 2: " + tf.strings.as_string(self.valid_accuracy2)
                     )
            # Early stopping--------------------------------------------------------
            wait += 1
            if tf.greater(self.valid_accuracy1, best):
                best = self.valid_accuracy1
                wait = 0
            if tf.greater_equal(wait, self.patience):
                break

    def test_model(self):
        # Test on a split----------------------------------------------------------------------------
        self.compile_node_model()
        self.compile_edge_model()
        test_loss1 = self.regularized_masked_cross_entropy_loss(self.graph, self.predictions1,
                                                                              self.regularization_constant1,
                                                                              self.test_mask1)
        test_accuracy1 = self.masked_accuracy(self.graph, self.predictions1, self.test_mask1)
        print("Final Test Loss 1: " + str(test_loss1.numpy()))
        print("Final Test Accuracy 1: " + str(test_accuracy1.numpy()))
        test_loss2 = self.regularized_masked_cross_entropy_loss(self.edge_graph, self.predictions2,
                                                                              self.regularization_constant2,
                                                                              self.test_mask2)
        test_accuracy2 = self.masked_accuracy(self.edge_graph, self.predictions2, self.test_mask2)
        print("Final Test Loss 2: " + str(test_loss2.numpy()))
        print("Final Test Accuracy 2: " + str(test_accuracy2.numpy()))

    def save_model(self, embedding_file1, metadata_file1, embedding_file2, metadata_file2):
        np.savetxt(embedding_file1, list(self.final_node_embeddings.numpy()), delimiter='\t', fmt='%f')
        np.savetxt(metadata_file1, list(self.graph.nodes.keys()), delimiter='\t', fmt='%s')
        np.savetxt(embedding_file2, list(self.final_edge_embeddings.numpy()), delimiter='\t', fmt='%f')
        np.savetxt(metadata_file2, list(self.edge_graph.nodes.keys()), delimiter='\t', fmt='%s')


@Utils.measure_execution_time
def main():
    print("Main Program")
    print("Tensorflow version: " + tf.__version__)
    if tf.test.is_built_with_cuda():
        print("Tensorflow built with CUDA support")
    else:
        print("Tensorflow is NOT built with CUDA support")
    print(tf.config.list_physical_devices("CPU"))
    print(tf.config.list_physical_devices("GPU"))
    # Load the Karate Club Network-----------------------------------------------------------
    karate_data = KarateClubNetwork("Data/Karate/karate.txt",
                                    "Data/Karate/karate-node-labels.txt")
    karate_data.load()
    karate_data.graph.info()
    print(karate_data.graph.node_type_profile)
    hidden_1_dim = 16
    hidden_2_dim = 16
    hidden_3_dim = 16
    epochs = 400
    regularization_rate = 0.0005
    dropout_rate = 0.5
    learning_rate = 0.001
    batch_normalization_factor = 0.001
    batch_normalization = False
    residual = True
    train = True
    patience = 400
    split = 70
    activation = tf.nn.tanh

    model_nodes = EdgeGCNModel(karate_data.graph,
                               [hidden_1_dim, hidden_2_dim, hidden_3_dim],
                               activation, residual, patience, epochs, split, train,
                               regularization_rate, dropout_rate, learning_rate,
                               batch_normalization, batch_normalization_factor)
    model_nodes.train_model()
    model_nodes.test_model()
    model_nodes.save_model("Data/Karate/karate_node_embedding.tsv", "Data/Karate/karate_node_metadata.tsv","Data/Karate/karate_edge_embedding.tsv", "Data/Karate/karate_edge_metadata.tsv")


if __name__ == '__main__':
    main()
