############################
# Author: Islam Akef Ebeid #
############################
import time
import numpy as np
import tensorflow as tf


class RGCNModel:
    def __init__(self, graph):
        # Input data information-----------------------------------------------------------------
        print("Input data information-----------------------------------------------------------------")
        print(graph.node_label_profile)
        print(graph.schema.keys())
        number_of_classes = int(len(graph.node_label_profile.keys()))
        number_of_nodes = graph.N
        graph.create_relational_adjacencies()
        # Configuration-----------------------------------------------------------------------------
        input_dim = number_of_nodes
        hidden_1_dim = 32
        hidden_2_dim = 32
        output_dim = number_of_classes
        num_bases = 20
        epochs = 500
        regularization_rate = 0.0005
        dropout_rate = 0.5
        learn_rate = 0.001
        batch_normalization_factor = 0.001
        batch_normalization = False
        patience = 50
        split = 60
        activation = tf.nn.relu
        # Training data-------------------------------------------------------------------------------------
        y = []
        x = []
        index = []
        for n, info in graph.nodes.items():
            x.append(str(n))
            index.append(graph.node_index[n])
            y_probabilty_vector = list(np.zeros((number_of_classes)))
            y_probabilty_vector[int(info["label"])] = 1.0
            y.append(y_probabilty_vector)
        y = tf.reshape(tf.convert_to_tensor(y, dtype=tf.float32), shape=(number_of_nodes, number_of_classes))
        train_samples, valid_samples, test_samples = graph.balanced_node_type_sampler(split)
        train_mask = np.zeros((number_of_nodes), dtype=int)
        train_mask[train_samples] = 1
        train_mask = train_mask.astype(bool)
        valid_mask = np.zeros((number_of_nodes), dtype=int)
        valid_mask[valid_samples] = 1
        valid_mask = valid_mask.astype(bool)
        test_mask = np.zeros((number_of_nodes), dtype=int)
        test_mask[test_samples] = 1
        test_mask = test_mask.astype(bool)

        # Model intialization and input-------------------------------------------------------------------------------
        def glorot_intializer(in_d, out_d):
            init = tf.keras.initializers.GlorotUniform()
            return init(shape=(in_d, out_d))

        def identity_initializer(in_d, out_d):
            init = tf.keras.initializers.Identity()
            return init(shape=(in_d, out_d))

        def zero_initializer(in_d, out_d):
            init = tf.keras.initializers.Zeros()
            return init(shape=(in_d, out_d))

        # Input features
        X = tf.Variable(glorot_intializer(number_of_nodes, input_dim), name="X")

        # RGCN Layer 1
        W_0 = tf.Variable(glorot_intializer(input_dim, hidden_1_dim), name="W_0")
        b_0 = tf.Variable(glorot_intializer(number_of_nodes, hidden_1_dim), name="b_0")

        # RGCN Layer 1 Relation specific parameters
        relation_specific_parameters_0 = {}
        V_b_list_0 = {}
        for i in range(num_bases):
            V_b = tf.Variable(glorot_intializer(input_dim, hidden_1_dim), name="V_b_0_" + str(i))
            V_b_list_0[i] = V_b
        for j, v in enumerate(graph.schema.values()):
            schema_source = v["source"]
            schema_target = v["target"]
            schema_type = v["type"]
            # W_r = tf.Variable(zero_initializer(input_dim, hidden_1_dim), name="W_r_0_" + str(j))
            a_r_b = tf.Variable(glorot_intializer(1, num_bases), name="a_r_b_0_" + str(j))
            b_r = tf.Variable(glorot_intializer(number_of_nodes, hidden_1_dim), name="b_r_0_" + str(j))
            relation_specific_parameters_0[str(schema_source + schema_type + schema_target)] = [a_r_b,b_r]


        # print([item for sublist in list(relation_specific_parameters_0.values()) for item in sublist])

        # RGCN Layer 2
        W_1 = tf.Variable(glorot_intializer(hidden_1_dim, hidden_2_dim), name="W_1")
        b_1 = tf.Variable(glorot_intializer(number_of_nodes, hidden_2_dim), name="b_1")

        # RGCN Layer 2 Relation specific parameters
        relation_specific_parameters_1 = {}
        V_b_list_1 = {}
        for i in range(num_bases):
            V_b = tf.Variable(glorot_intializer(hidden_1_dim, hidden_2_dim), name="V_b_1_" + str(i))
            V_b_list_1[i] = V_b
        for j, v in enumerate(graph.schema.values()):
            schema_source = v["source"]
            schema_target = v["target"]
            schema_type = v["type"]
            # W_r = tf.Variable(glorot_intializer(hidden_1_dim, hidden_2_dim), name="W_r_1_" + str(j))
            a_r_b = tf.Variable(glorot_intializer(1, num_bases), name="a_r_b_1_" + str(j))
            b_r = tf.Variable(glorot_intializer(number_of_nodes, hidden_2_dim), name="b_r_1_" + str(j))
            relation_specific_parameters_1[str(schema_source + schema_type + schema_target)] = [a_r_b,b_r]

        # Fully connected layer 3
        W_2 = tf.Variable(glorot_intializer(hidden_2_dim, output_dim), name="W_2")
        b_2 = tf.Variable(glorot_intializer(number_of_nodes, output_dim), name="b_2")

        # Batch normalization weights
        scale0 = tf.Variable(tf.ones([hidden_1_dim]))
        beta0 = tf.Variable(tf.zeros([hidden_1_dim]))

        scale1 = tf.Variable(tf.ones([hidden_2_dim]))
        beta1 = tf.Variable(tf.zeros([hidden_2_dim]))

        # TODO:
        # 1- dynamic layers
        # 2- reorganize in classes
        # 3- minibatch
        # Model building-----------------------------------------------------------------------------------
        def model(graph, X, W_0, W_1, W_2, b_0, b_1, b_2, scale0, scale1, beta0,
                  beta1, relation_specific_parameters_0, relation_specific_parameters_1, V_b_list_0, V_b_list_1,
                  regularization_rate, dropout_rate, batch_normalization,
                  batch_normalization_factor, train=True):

            def l2_regularization_layer(rate, input):
                r = tf.keras.regularizers.L2(rate)
                return r(input)

            def batch_normalization_layer(input, scale, beta, batch_normalization_factor):
                batch_mean1, batch_var1 = tf.nn.moments(input, [0])
                input_hat = (input - batch_mean1) / tf.sqrt(batch_var1 + batch_normalization_factor)
                bn1 = scale * input_hat + beta
                return tf.nn.sigmoid(bn1)

            def relational_gcn_layer(input, graph, dim1, dim2, W, b, relation_specific_parameters, V_b_list):
                output = tf.zeros([graph.N, dim2], tf.float32)
                for r, AD_relational in graph.relation_degree_normalized_adjacency_matrices.items():
                    a_r_list = relation_specific_parameters[r][0]
                    a_r_list = tf.squeeze(a_r_list,axis=0)
                    W_r = tf.zeros([dim1, dim2], tf.float32)
                    for i in range(num_bases):
                        a_r_b = a_r_list[i]
                        V_b = V_b_list[i]
                        W_r = tf.add(W_r, tf.multiply(a_r_b, V_b))
                    b_r = relation_specific_parameters[r][1]
                    select = tf.matmul(AD_relational, input)
                    convolve = tf.matmul(select, W_r)
                    hidden_layer_a = tf.add(convolve, b_r)
                    hidden_layer_b = tf.add(tf.matmul(select, W), b)
                    agg = tf.add(hidden_layer_a, hidden_layer_b) / graph.relations_adjacency_node_counts[r]
                    output = tf.add(output, agg)
                return output

            def dense_layer(input, W, b):
                return tf.add(tf.matmul(input, W), b)

            # First GCN layer 1
            Z_0 = relational_gcn_layer(X, graph, input_dim, hidden_1_dim, W_0, b_0, relation_specific_parameters_0,
                                       V_b_list_0)
            # Activation layer 1
            # Batch normalization layer 1
            if batch_normalization:
                bn_hidden_layer = batch_normalization_layer(Z_0, scale0, beta0, batch_normalization_factor)
                Z_0 = activation(bn_hidden_layer)
            else:
                Z_0 = activation(Z_0)
            # Compute l2 regularization for first layer only
            regularization_constant = l2_regularization_layer(regularization_rate, Z_0)
            if train:
                # Dropout layer 1
                Z_0 = tf.nn.dropout(Z_0, dropout_rate)
            # Second GCN layer 2
            Z_1 = relational_gcn_layer(Z_0, graph, hidden_1_dim, hidden_2_dim, W_1, b_1, relation_specific_parameters_1,
                                       V_b_list_1)
            # Activaion layer 2
            # Batch normalization layer 2
            if batch_normalization:
                bn_hidden_layer = batch_normalization_layer(Z_1, scale1, beta1, batch_normalization_factor)
                Z_1 = activation(bn_hidden_layer)
            else:
                Z_1 = activation(Z_1)
            # Extract embeddings if needed
            embeddings = tf.nn.l2_normalize(Z_1, axis=1)
            if train:
                # Dropout layer 2
                Z_1 = tf.nn.dropout(Z_1, dropout_rate)
            # Fully Connected Layer 3
            logits = activation(dense_layer(Z_1, W_2, b_2))
            # Output layers
            predictions = tf.map_fn(fn=lambda i: tf.nn.softmax(i), elems=logits)
            if train:
                return predictions, regularization_constant
            else:
                return predictions, regularization_constant, embeddings

        # Loss--------------------------------------------------------------------------
        def regularized_masked_cross_entropy_loss(prediction, y, r, mask):
            return (-tf.reduce_mean(
                tf.reduce_sum(tf.boolean_mask(y, mask) * tf.math.log(tf.boolean_mask(prediction, mask))))) + r

        # Optimizer------------------------------------------------------------------------
        def optimizer(learn_rate):
            optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
            return optimizer

        # Evaluation from Kipf et al 2017-------------------------------------------------------
        def masked_accuracy(prediction, y, mask):
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy_all = tf.cast(correct_prediction, tf.float32)
            mask = tf.cast(mask, dtype=tf.float32)
            mask /= tf.reduce_mean(mask)
            accuracy_all *= mask
            accuracy = tf.reduce_mean(accuracy_all)
            return accuracy

        # Train model-----------------------------------------------------------------------------
        @tf.function
        def train_full_batch(graph, X, W_0, W_1, W_2, b_0, b_1, b_2, M_0, M_1, M_2,
                             scale0, scale1, beta0, beta1, relation_specific_parameters_0, relation_specific_parameters_1,
                             V_b_list_0, V_b_list_1,
                             y, optimizer, train_mask, valid_mask, regularization_rate,
                             dropout_rate,
                             batch_normalization, batch_normalization_factor, patience):
            wait = 0
            best = 0.0
            for epoch in tf.range(epochs):
                start_time = time.perf_counter()
                with tf.GradientTape() as tape:
                    prediction, regularization_constant = model(graph, X, W_0, W_1, W_2,
                                                                b_0, b_1, b_2, M_0, M_1, M_2, scale0, scale1,
                                                                beta0, beta1, relation_specific_parameters_0,
                                                                relation_specific_parameters_1, V_b_list_0, V_b_list_1,
                                                                regularization_rate,
                                                                dropout_rate, batch_normalization,
                                                                batch_normalization_factor, True)
                    train_loss = regularized_masked_cross_entropy_loss(prediction, y, regularization_constant, train_mask)
                    train_accuracy = masked_accuracy(prediction, y, train_mask)
                if batch_normalization:
                    params = [X, W_0, W_1, W_2, b_0, b_1, b_2, scale0, scale1, beta0, beta1] + list(
                        V_b_list_0.values()) + list(V_b_list_1.values()) + [
                                 item for sublist in list(
                            relation_specific_parameters_0.values()) for item in
                                 sublist] + [item for sublist
                                             in list(
                            relation_specific_parameters_1.values()) for
                                             item in sublist]
                else:
                    params = [X, W_0, W_1, W_2, b_0, b_1, b_2] + list(V_b_list_0.values()) + list(V_b_list_1.values()) + [
                        item for sublist in list(
                            relation_specific_parameters_0.values()) for item in sublist] + [item for sublist in list(
                        relation_specific_parameters_1.values()) for item in sublist]
                gradients = tape.gradient(train_loss, params)
                optimizer.apply_gradients(list(zip(gradients, params)))
                prediction, regularization_constant, _ = model(graph, X, W_0, W_1, W_2,
                                                               b_0, b_1, b_2, scale0, scale1,
                                                               beta0, beta1, relation_specific_parameters_0,
                                                               relation_specific_parameters_1, V_b_list_0, V_b_list_1,
                                                               regularization_rate,
                                                               dropout_rate, batch_normalization,
                                                               batch_normalization_factor, False)
                valid_loss = regularized_masked_cross_entropy_loss(prediction, y, regularization_constant, valid_mask)
                valid_accuracy = masked_accuracy(prediction, y, valid_mask)
                end_time = time.perf_counter()
                time_per_epoch = tf.constant(round((end_time - start_time), 3), dtype=tf.float32)
                tf.print(" Epoch: " + tf.strings.as_string(epoch)
                         + " Seconds/Epoch: " + tf.strings.as_string(time_per_epoch)
                         + " Learning Rate: " + tf.strings.as_string(tf.constant(round(learn_rate, 3), dtype=tf.float32))
                         + " Train Loss: " + tf.strings.as_string(train_loss)
                         + " Train Accuracy: " + tf.strings.as_string(train_accuracy)
                         + " Valid Loss: " + tf.strings.as_string(valid_loss)
                         + " Valid Accuracy: " + tf.strings.as_string(valid_accuracy)
                         )
                # Early stopping--------------------------------------------------------
                wait += 1
                if tf.greater(valid_accuracy, best):
                    best = valid_accuracy
                    wait = 0
                if tf.greater_equal(wait, patience):
                    break

        # Train---------------------------------------------------------------------------------
        train_full_batch(graph, X, W_0, W_1, W_2, b_0, b_1, b_2,
                         scale0,
                         scale1,
                         beta0, beta1, relation_specific_parameters_0, relation_specific_parameters_1, V_b_list_0,
                         V_b_list_1, y,
                         optimizer(learn_rate),
                         train_mask,
                         valid_mask, regularization_rate,
                         dropout_rate, batch_normalization, batch_normalization_factor, patience)
        # Test on a split----------------------------------------------------------------------------
        prediction, regularization_constant, final_embedding = model(graph, X, W_0, W_1, W_2,
                                                                     b_0, b_1, b_2, scale0, scale1,
                                                                     beta0, beta1, relation_specific_parameters_0,relation_specific_parameters_1,
                                                                     V_b_list_0, V_b_list_1,
                                                                     regularization_rate,
                                                                     dropout_rate, batch_normalization,
                                                                     batch_normalization_factor, False)
        test_loss = regularized_masked_cross_entropy_loss(prediction, y, regularization_constant, test_mask)
        test_accuracy = masked_accuracy(prediction, y, test_mask)
        print("Final Test Loss: " + str(test_loss.numpy()))
        print("Final Test Accuracy: " + str(test_accuracy.numpy()))
        # final_embedding = list(final_embedding.numpy())
        # np.savetxt(embedding_file, final_embedding, delimiter='\t', fmt='%f')
        # np.savetxt(metadata_file, nodes, delimiter='\t', fmt='%s')