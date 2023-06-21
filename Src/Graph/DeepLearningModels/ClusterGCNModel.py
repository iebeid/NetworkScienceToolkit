import time
import random
import numpy as np
import tensorflow as tf
from Graph import GraphObject


class ClusterGCNModel:
    def __init__(self,netgraph_graph):

        # Load graph
        netgraph_graph.info()
        print(netgraph_graph.node_type_profile)
        print(netgraph_graph.schema.keys())
        number_of_classes = int(len(netgraph_graph.node_type_profile.keys()))
        number_of_nodes = netgraph_graph.N
        print("Number of classes: " + str(number_of_classes))


        # Configuration
        input_dim = number_of_nodes
        hidden_1_dim = 32
        hidden_2_dim = 32
        output_dim = number_of_classes
        epochs = 50
        regularization_rate = 0.0005
        dropout_rate = 0.5
        learn_rate = 0.01
        cluster_gcn = True
        residual_connections = False
        batch_normalization_factor = 0.001
        batch_normalization = False
        patience = 100
        split = 70
        number_of_clusters = 4  # the number of clusters/subgraphs
        clusters_per_batch = 2  # combine two cluster per batch




        # Training Data
        y = []
        x = []
        index = []
        nodes = list(netgraph_graph.nodes.keys())
        for n, info in netgraph_graph.nodes.items():
            x.append(str(n))
            index.append(netgraph_graph.node_index[n])
            y_probabilty_vector = list(np.zeros((number_of_classes)))
            y_probabilty_vector[int(info["type"])] = 1.0
            # y_probabilty_vector[int(info["label"])] = 1.0
            y.append(y_probabilty_vector)
        y = tf.reshape(tf.convert_to_tensor(y, dtype=tf.float32), shape=(number_of_nodes, number_of_classes))
        # train_samples, valid_samples, test_samples = graph.balanced_node_label_sampler(split)
        train_samples, valid_samples, test_samples = netgraph_graph.balanced_node_type_sampler(split)
        train_mask = np.zeros((number_of_nodes), dtype=int)
        train_mask[train_samples] = 1
        train_mask = train_mask.astype(bool)
        valid_mask = np.zeros((number_of_nodes), dtype=int)
        valid_mask[valid_samples] = 1
        valid_mask = valid_mask.astype(bool)
        test_mask = np.zeros((number_of_nodes), dtype=int)
        test_mask[test_samples] = 1
        test_mask = test_mask.astype(bool)


        # TODO:
        # 1- dynamic layers
        # 2- reorganize in classes
        def model(AD_inv, X, W_0, W_1, W_2, M_0, M_1, M_2, b_0, b_1, b_2, scale0, scale1, scale2, beta0, beta1, beta2,
                  regularization_rate, regularization_constant, dropout_rate, residual_connections, batch_normalization,
                  batch_normalization_factor, train=True):

            def l2_regularization_layer(rate, input):
                r = tf.keras.regularizers.L2(rate)
                return r(input)

            def batch_normalization_layer(input, scale, beta, batch_normalization_factor):
                batch_mean1, batch_var1 = tf.nn.moments(input, [0])
                input_hat = (input - batch_mean1) / tf.sqrt(batch_var1 + batch_normalization_factor)
                bn1 = scale * input_hat + beta
                return tf.nn.sigmoid(bn1)

            def gcn_layer(activation, input, AD_inv, W, M, b, scale, beta, residual_connections, batch_normalization_factor, batch_normalization,
                          train):
                graph_convolution_approximation = tf.matmul(tf.matmul(AD_inv, input), W)
                hidden_layer = tf.add(graph_convolution_approximation, b)
                activation_layer = 0.0
                if train:
                    if batch_normalization:
                        bn_hidden_layer = batch_normalization_layer(hidden_layer, scale, beta, batch_normalization_factor)
                        activation_layer = activation(bn_hidden_layer)
                    else:
                        activation_layer = activation(hidden_layer)
                residual_layer = tf.matmul(input, M)
                if residual_connections:
                    output = tf.add(activation_layer, residual_layer)
                else:
                    output = activation_layer
                return output

            # Three GCN layers
            Z_0 = gcn_layer(tf.nn.relu, X, AD_inv, W_0, M_0, b_0, scale0, beta0, batch_normalization_factor,
                            batch_normalization, train)

            if train:
                regularization_constant = l2_regularization_layer(regularization_rate, Z_0)
                Z_0 = tf.nn.dropout(Z_0, dropout_rate)
            Z_1 = gcn_layer(tf.nn.relu, Z_0, AD_inv, W_1, M_1, b_1, scale1, beta1, batch_normalization_factor,
                            batch_normalization, train)
            embeddings = tf.nn.l2_normalize(Z_1, axis=1)
            if train:
                Z_1 = tf.nn.dropout(Z_1, dropout_rate)
            logits = gcn_layer(tf.nn.relu, Z_1, AD_inv, W_2, M_2, b_2, scale2, beta2, batch_normalization_factor,
                               batch_normalization, train)
            # Output layers
            predictions = tf.map_fn(fn=lambda i: tf.nn.softmax(i), elems=logits)
            if train:
                return predictions
            else:
                return predictions, embeddings

        # Loss
        def regularized_masked_cross_entropy_loss(prediction, y, r, mask):
            return (-tf.reduce_mean(
                tf.reduce_sum(tf.boolean_mask(y, mask) * tf.math.log(tf.boolean_mask(prediction, mask))))) + r

        # Optimizer
        def optimizer(learn_rate):
            optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
            return optimizer

        # Evaluation from Kipf et al 2017
        def masked_accuracy(prediction, y, mask):
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy_all = tf.cast(correct_prediction, tf.float32)
            mask = tf.cast(mask, dtype=tf.float32)
            mask /= tf.reduce_mean(mask)
            accuracy_all *= mask
            accuracy = tf.reduce_mean(accuracy_all)
            return accuracy

        def glorot_intializer(in_d, out_d):
            init = tf.keras.initializers.GlorotUniform()
            return init(shape=(in_d, out_d))

        def identity_initializer(in_d, out_d):
            init = tf.keras.initializers.Identity()
            return init(shape=(in_d, out_d))

        regularization_constant = tf.Variable(0.0, trainable=False)

        final_embedding = tf.Variable(glorot_intializer(netgraph_graph.N, hidden_2_dim), trainable=False)

        # Train model. GPU mode through the use of tf.function will copy everything to GPU memory.
        # @tf.function
        def train_full_batch(graph, input_dim, hidden_1_dim, hidden_2_dim, output_dim, y, optimizer, train_mask, valid_mask, regularization_rate, dropout_rate, batch_normalization, batch_normalization_factor, patience):

            # -----------------
            # In training parameters
            wait = 0
            best = 0.0
            # -----------------
            # Cluster GCN specific procedure
            for epoch in tf.range(epochs):
                start_time = time.perf_counter()

                # randomly choose q partitions
                chosen_clusters = random.sample(list(netgraph_graph.node_label_profile.keys()), k=2)
                print(chosen_clusters)
                # extract the corresponding subgraph
                nodes = []
                subgraph_nodes = {}
                for n, label in netgraph_graph.node_label.items():
                    for l in chosen_clusters:
                        if label != l:
                            continue
                        nodes.append(n)
                        subgraph_nodes[n] = {
                            "alt_id": "none",
                            "type": netgraph_graph.node_type[n],
                            "label": label,
                            "attributes": netgraph_graph.node_attributes[n],
                            "features": netgraph_graph.node_features[n]
                        }
                nodes = list(set(nodes))
                temp_nodes = nodes
                subgraph_edges = {}
                edge_index = 0
                for n in nodes:
                    neighbors = netgraph_graph.tasks.get_neighbors(n)
                    temp_nodes.remove(n)
                    intersection = list(set.intersection(set(temp_nodes), set(neighbors)))
                    if len(intersection) != 0:
                        for e in intersection:
                            edge_type = None
                            for n1, n2, w, r in netgraph_graph.edges:
                                if n1 == n and n2 == e:
                                    edge_type = r
                            subgraph_edges[edge_index] = {"source": n, "target": e, "type": edge_type,
                                                          "weight": float(1)}
                            edge_index = edge_index + 1
                    temp_nodes.append(n)
                subgraph_schema = {}
                for edge_id, values in subgraph_edges.items():
                    source_node_type = str(subgraph_nodes[values["source"]]["type"])
                    relation_node_type = str(values["type"])
                    target_node_type = str(subgraph_nodes[values["target"]]["type"])
                    subgraph_schema[source_node_type + relation_node_type + target_node_type] = {"source": source_node_type,
                                                                                                 "target": target_node_type,
                                                                                                 "type": relation_node_type,
                                                                                                 "weight": float(1)}
                subgraph = GraphObject(subgraph_edges, nodes=subgraph_nodes, schema=subgraph_schema, undirected=True,
                                    link_single_nodes=False)
                print(subgraph.info())
                subgraph_degree_normalized_adjacency = subgraph.tasks.degree_normalized_adjacency()
                subgraph_AD_inv = tf.constant(
                    tf.convert_to_tensor(np.asarray(subgraph_degree_normalized_adjacency, dtype=np.float32), dtype=tf.float32))

                #---------------------------------------------------------------------------------

                # Initialization
                # Model intialization and input


                # degree_normalized_adjacency = graph.tasks.degree_normalized_adjacency()
                # AD_inv = tf.constant(tf.convert_to_tensor(np.asarray(degree_normalized_adjacency, dtype=np.float32), dtype=tf.float32))
                # AD_inv = graph.tasks.degree_normalized_adjacency_tensorflow()
                X = tf.Variable(identity_initializer(subgraph.N, input_dim))
                W_0 = tf.Variable(glorot_intializer(input_dim, hidden_1_dim))
                b_0 = tf.Variable(glorot_intializer(subgraph.N, hidden_1_dim))
                M_0 = tf.Variable(glorot_intializer(input_dim, hidden_1_dim))
                W_1 = tf.Variable(glorot_intializer(hidden_1_dim, hidden_2_dim))
                b_1 = tf.Variable(glorot_intializer(subgraph.N, hidden_2_dim))
                M_1 = tf.Variable(glorot_intializer(hidden_1_dim, hidden_2_dim))
                W_2 = tf.Variable(glorot_intializer(hidden_2_dim, output_dim))
                b_2 = tf.Variable(glorot_intializer(subgraph.N, output_dim))
                M_2 = tf.Variable(glorot_intializer(hidden_2_dim, output_dim))
                scale0 = tf.Variable(tf.ones([hidden_1_dim]))
                beta0 = tf.Variable(tf.zeros([hidden_1_dim]))
                scale1 = tf.Variable(tf.ones([hidden_2_dim]))
                beta1 = tf.Variable(tf.zeros([hidden_2_dim]))
                scale2 = tf.Variable(tf.ones([output_dim]))
                beta2 = tf.Variable(tf.zeros([output_dim]))



                with tf.GradientTape() as tape:
                    prediction = model(subgraph_AD_inv, X, W_0, W_1, W_2, M_0, M_1, M_2,
                                       b_0, b_1, b_2, scale0, scale1, scale2,
                                       beta0, beta1, beta2, regularization_rate, regularization_constant,
                                       dropout_rate, batch_normalization, batch_normalization_factor, True)
                    train_loss = regularized_masked_cross_entropy_loss(prediction, y, regularization_constant, train_mask)
                    train_accuracy = masked_accuracy(prediction, y, train_mask)
                if batch_normalization:
                    params = [X, W_0, W_1, W_2, M_0, M_1, M_2, b_0, b_1, b_2, scale0, scale1, scale2, beta0, beta1, beta2]
                else:
                    params = [X, W_0, W_1, W_2, M_0, M_1, M_2, b_0, b_1, b_2]
                gradients = tape.gradient(train_loss, params)
                optimizer.apply_gradients(list(zip(gradients, params)))
                prediction, embeddings = model(subgraph_AD_inv, X, W_0, W_1, W_2, M_0, M_1, M_2,
                                      b_0, b_1, b_2, scale0, scale1, scale2,
                                      beta0, beta1, beta2, regularization_rate, regularization_constant,
                                      dropout_rate, batch_normalization,
                                      batch_normalization_factor, False)
                final_embedding = tf.nn.embedding_lookup(final_embedding,)
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
                # Early stopping
                wait += 1
                if tf.greater(valid_loss, best):
                    best = valid_loss
                    wait = 0
                if tf.greater_equal(wait, patience):
                    break
                if tf.greater_equal(valid_accuracy, 0.9):
                    break

        train_full_batch(y,optimizer(learn_rate),train_mask,valid_mask,regularization_rate,dropout_rate,batch_normalization,batch_normalization_factor,patience)
        # Test on a split
        prediction, final_embedding = model(regularization_rate, dropout_rate, batch_normalization, batch_normalization_factor, False)
        test_loss = regularized_masked_cross_entropy_loss(prediction, y, regularization_constant, test_mask)
        test_accuracy = masked_accuracy(prediction, y, test_mask)
        print("Final Test Loss: " + str(test_loss.numpy()))
        print("Final Test Accuracy: " + str(test_accuracy.numpy()))
        final_embedding = list(final_embedding.numpy())
        models_directory = "../data/embedding/"
        embedding_file = models_directory + "karate_club_network/embedding/" + "karate_embedding.tsv"
        metadata_file = models_directory + "karate_club_network/embedding/" + "karate_metadata.tsv"
        np.savetxt(embedding_file, final_embedding, delimiter='\t', fmt='%f')
        np.savetxt(metadata_file, nodes, delimiter='\t', fmt='%s')