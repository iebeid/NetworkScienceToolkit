import argparse
from models.data_utils import *

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def parse_args():
    parser = argparse.ArgumentParser(description="Make rel2vec embeddings") 
    parser.add_argument('--node-list', nargs='?', default='data/c2b2rdf/nodes.txt',
                        help='Input neighbor and pata data')
    parser.add_argument('--edge-list', nargs='?', default='data/c2b2rdf/edges.txt',
                        help='Input neighbor and pata data')
    parser.add_argument('--output-dim', type=int, default=200, help='Dimension of embedding vectors.')
    parser.add_argument('--output', nargs='?', default="emb/c2b2rdf/r2vectors.txtnp", help='Where to save embedding vectors.')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size (number of nodes to process at once)')
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--pos-targets', nargs='?', default='data/c2b2rdf/train_pos.csv')
    parser.add_argument('--neg-targets', nargs='?', default='data/c2b2rdf/train_neg.csv')

    return parser.parse_args()

class RelEmbedding(Layer):
    def __init__(self, embed_dim, num_nodes, num_edge_types, **kwargs):
        self.embed_dim = embed_dim
        self.num_edge_types = num_edge_types
        self.num_nodes = num_nodes
        super(RelEmbedding, self).__init__(**kwargs)
    def build(self, input_shape):
        self.input_to_hidden = self.add_weight(shape=(self.num_nodes, self.embed_dim), trainable=True)
        self.hidden_to_predict = Dense(units=(self.num_nodes * self.num_edge_types), use_bias = False)
        self.sm = Softmax()
        self.rs = Reshape((self.num_nodes, self.num_edge_types))    
        super(RelEmbedding, self).build(input_shape)  # Be sure to call this at the end
    def call(self, heads):
        x = tf.gather(self.input_to_hidden, heads)
        x = self.hidden_to_predict(x)
        x = self.sm(x)
        x = self.rs(x)
        return x
    def compute_output_shape(self, input_shape):
        out_dim = (self.num_nodes, self.num_edge_types)
        return out_dim

def make_ds_from_triples(triples, batch_size, num_nodes, num_edge_types):
    heads = triples[ : , 0]
    rel_tails = triples[ : , 1:3]
    targets_ds = tf.data.Dataset.from_tensor_slices(rel_tails)
    targets_ds = targets_ds.map(lambda rel_tail: tf.reshape(tf.one_hot((rel_tail[0] * num_edge_types) + rel_tail[1], num_nodes * num_edge_types), (num_nodes, num_edge_types)),  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    head_ds = tf.data.Dataset.from_tensor_slices(heads)
    head_ds = head_ds.map(lambda head: tf.cast(head, tf.int32), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    total_ds = tf.data.Dataset.zip((head_ds, targets_ds))
    return total_ds.batch(batch_size)

def make_loss(relation_weight, node_weight):
    def loss(true, predicted):
        # Where true and predicted are size (nodes, edges)
        # First off, it gets points for guessing the correct tail AND relation. 
        ce1 = tf.keras.losses.CategoricalCrossentropy()(true, predicted)
        # It also gets partial credit for guessing the correct tail by itself, even if it didn't get the relation 100%.
        true_tails = tf.reduce_sum(true, axis=2)
        predicted_tails = tf.reduce_sum(predicted, axis=2)
        ce2 = tf.keras.losses.CategoricalCrossentropy()(true_tails, predicted_tails)
        # Finally, it also gets partial credit for guessing the relation type. Do what you can, model. 
        true_relations = tf.reduce_sum(true, axis=1)
        predicted_relations = tf.reduce_sum(predicted, axis=1)
        ce3 = tf.keras.losses.CategoricalCrossentropy()(true_relations, predicted_relations)
        return ce1 + (node_weight * ce2) + (relation_weight * ce3)
    return loss

def main(args):

    batch_size = args.batch_size
    epochs = args.epochs
    nodes = load_dict(args.node_list)
    edges = load_dict(args.edge_list)
    num_nodes = len(nodes)
    num_edge_types = len(edges)
    print("Loading triples from files...")
    total_triples = load_triples(args.pos_targets)
    print("Found %d triples!" % len(total_triples))
    train_ds = make_ds_from_triples(tf.convert_to_tensor(total_triples), batch_size, num_nodes, num_edge_types)

    print("Building model...")
    in_layer = Input(shape=(), dtype=tf.int32)
    rel_layer = RelEmbedding(args.output_dim, num_nodes, num_edge_types)
    rel = rel_layer(in_layer)
    model = Model(inputs=in_layer, outputs=rel)
    model.compile(loss=make_loss(1.0/num_nodes, 1.0/num_edge_types), optimizer=Adam(lr=0.01), metrics=['categorical_accuracy'])
    print(model.summary())
    model.fit(x = train_ds, epochs=epochs, verbose=1)

    # Save the embeddings.
    weights = rel_layer.get_weights()[0]
    save_features(args.output, weights)

if __name__ == "__main__":
    args = parse_args()
    main(args)