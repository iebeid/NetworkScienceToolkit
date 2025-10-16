import pandas as pd
import stellargraph as sg
from sklearn import model_selection
import tensorflow as tf
import numpy as np


def encode_node_type(node_type):
    node_type_code = ""
    if node_type == "Gene ontology":
        node_type_code = "[go]"
    if node_type == "Chemical ontology":
        node_type_code = "[co]"
    if node_type == "Substructure":
        node_type_code = "[ss]"
    if node_type == "Target":
        node_type_code = "[ta]"
    if node_type == "Tissue":
        node_type_code = "[ti]"
    if node_type == "Pathway":
        node_type_code = "[pa]"
    if node_type == "Disease":
        node_type_code = "[di]"
    if node_type == "Chemical Compound/Drug":
        node_type_code = "[dr]"
    if node_type == "Side effect":
        node_type_code = "[se]"
    if node_type == "N/A":
        node_type_code = "[na]"
    return str(node_type_code)


def load_graph(node_file,edge_file):
    nodes_df = pd.read_csv(node_file, sep=",", header=None, encoding='utf-8')
    edges_df = pd.read_csv(edge_file, sep=",", header=None, encoding='utf-8')

    nodes_list = []
    node_type_list = []
    output_layer = []
    nodes_and_types_list = []
    edges_source_list = []
    edges_target_list = []

    for index, line in nodes_df.iterrows():
        node_id = int(str(line[0]).rstrip())
        node_type = str(line[1]).rstrip()
        node_type = encode_node_type(node_type)
        nodes_list.append(node_id)
        node_type_id = 0


        if node_type == "[go]":
            node_type_id = 1
            output_layer.append([node_id, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if node_type == "[co]":
            node_type_id = 2
            output_layer.append([node_id, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if node_type == "[ss]":
            node_type_id = 3
            output_layer.append([node_id, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if node_type == "[ta]":
            node_type_id = 4
            output_layer.append([node_id, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if node_type == "[ti]":
            node_type_id = 5
            output_layer.append([node_id, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if node_type == "[pa]":
            node_type_id = 6
            output_layer.append([node_id, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if node_type == "[di]":
            node_type_id = 7
            output_layer.append([node_id, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if node_type == "[dr]":
            node_type_id = 8
            output_layer.append([node_id, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if node_type == "[se]":
            node_type_id = 9
            output_layer.append([node_id, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        node_type_list.append(node_type_id)
        nodes_and_types_list.append([node_id,node_type_id])

    for index, line in edges_df.iterrows():
        source = int(str(line[0]).rstrip())
        target = int(str(line[1]).rstrip())
        edges_source_list.append(source)
        edges_target_list.append(target)

    nodes_df = pd.DataFrame({"type": node_type_list}, index=nodes_list)
    edges_df = pd.DataFrame({"source": edges_source_list, "target": edges_target_list})

    graph = sg.StellarGraph(nodes_df, edges_df)

    return graph, nodes_and_types_list, output_layer


def main():
    node_file = "Chem2Bio2RDF/fixed_nodes_ids.csv"
    edge_file = "Chem2Bio2RDF/fixed_edges_ids.csv"
    graph, targets, output_layer = load_graph(node_file, edge_file)
    print("Number of nodes {} and number of edges {} in graph.".format(graph.number_of_nodes(), graph.number_of_edges()))
    train_targets, test_targets = model_selection.train_test_split(targets, train_size=0.5)

    train_targets = np.array(train_targets)
    test_targets = np.array(test_targets)

    generator = sg.mapper.FullBatchNodeGenerator(graph, method="gcn")

    # two layers of GCN, each with hidden dimension 16
    gcn = sg.layer.GCN(layer_sizes=[16, 16], generator=generator)
    x_inp, x_out = gcn.build()

    print(x_inp)
    print(x_out)

    # use TensorFlow Keras to add a layer to compute the (one-hot) predictions
    predictions = tf.keras.layers.Dense(units=16, activation="softmax")(x_out)

    # use the input and output tensors to create a TensorFlow Keras model
    model = tf.keras.Model(inputs=x_inp, outputs=predictions)

    # prepare the model for training with the Adam optimiser and an appropriate loss function
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # train the model on the train set
    model.fit(generator.flow(train_targets[:,0], train_targets[:,1]), epochs=5)

    # check model generalisation on the test set
    (loss, accuracy) = model.evaluate(generator.flow(test_targets[:,0], test_targets[:,1]))
    print(f"Test set: loss = {loss}, accuracy = {accuracy}")


if __name__ == "__main__":
    main()