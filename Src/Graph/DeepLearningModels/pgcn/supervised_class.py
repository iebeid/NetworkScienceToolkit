import argparse
import os
from models.data_utils import *
from tensorflow.keras.models import Model
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Add, Flatten, Dropout, Lambda
from scipy.sparse import identity
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from layers import *
from tensorflow.keras.optimizers import Adam
from random import shuffle

from functools import partial

import datetime

from memory_profiler import profile

from math import ceil, floor, pow 


def parse_args():
    parser = argparse.ArgumentParser(description="Train on a node classification task") 
    parser.add_argument('--input-data', nargs='?', default='emb/aifb/stackmeta.pbz2',
                        help='Input neighbor and pata data')
    parser.add_argument('--node-list', nargs='?', default='data/aifb/nodes.txt', help='List of nodes')
    parser.add_argument('--output-dim', type=int, default=16, help='Dimension of embedding vectors.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (number of nodes to process at once)')
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--train-targets', nargs='?', default='data/aifb/train_class.csv')
    parser.add_argument('--test-targets', nargs='?', default='data/aifb/test_class.csv')
    parser.add_argument('--ids', dest='ids', action='store_true',
                        help='Boolean specifying if to use node ids when training (one-hot vectors for each node). Default is false.') 
    parser.add_argument('--nologs', dest='logs', action='store_false',
                        help='Boolean specifying NOT to make tensorboard logs. Default is to make logs. ')
    parser.set_defaults(ids=False)
    parser.set_defaults(logs=True)
    return parser.parse_args()    

def printpd(pd):
    # Get the average number of each path type: 
    minPNum = 999999999
    minP = None
    maxPNum = 0 
    maxP = None
    totalP = 0
    for path, data in pd.items():
        count = data['count']
        name = data['name']
        totalP += count
        if count < minPNum:
            minPNum = count
            minP = name
        if count > maxPNum:
            maxPNum = count
            maxP = name
    print("minPath is %s with count %d" % (minP, minPNum))
    print("maxPath is %s with count %d" % (maxP, maxPNum))
    print("%d total and %d average count" % (totalP, totalP / len(pd)))
    return minPNum, maxPNum

def process_nodeid(nodeid, neighbor_feats, dims, pathid):
    feat = neighbor_feats[pathid][nodeid]
    if feat == None:
        dim = dims[pathid]
        dtype = tf.float32
        if pathid >= (len(dims) / 2):
            dtype = tf.int32
        feat = tf.zeros((1, dim), dtype=dtype)
    return feat

def make_ds_from_node_targets(node_targets, neighbor_feats, pd, output_types, batch_size, tids):
    process_pyfunc = (lambda triple: process_triple_for_dataset(neighbor_feats, pd, triple))
    target_map_ds = tf.data.Dataset.from_tensor_slices(node_targets).repeat(None)
    node_ds = target_map_ds.map(lambda node_target: node_target[0],  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    path_type_dims = get_path_dims(pd)
    num_path_types = len(path_type_dims)
    id_dims = get_id_dims(pd)
    dims = path_type_dims + id_dims
    dtypes = ([tf.float32] * num_path_types) + ([tf.int32] * num_path_types)
    path_pyfuncs = [(partial(process_nodeid, neighbor_feats=neighbor_feats, dims=dims, pathid=pathid)) for pathid in range(num_path_types * 2)]
    path_ds = []
    for pathid in range(num_path_types * 2):
        dtype = dtypes[pathid]
        p_ds = node_ds.map(lambda node: tf.py_function(func = path_pyfuncs[pathid], inp=[node], Tout=dtype), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        shape = (None, dims[pathid])
        p_ds = p_ds.map(lambda tens: tf.ensure_shape(tens, shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        path_ds.append(p_ds)
    neighbor_ds = tf.data.Dataset.zip(tuple(path_ds))
    target_ds = target_map_ds.map(lambda node_target: tf.one_hot(node_target[1], len(tids)))
    ds = tf.data.Dataset.zip((neighbor_ds, target_ds))
    ds = ds.padded_batch(batch_size, drop_remainder=True).repeat(None).prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def main(args):
    batch_size = args.batch_size
    epochs = args.epochs
    patience = int(epochs / 5)
    node_dict = load_dict(args.node_list)
    saved_data = load_cpickle(args.input_data)
    pd = saved_data["path_data"]
    path_type_dims = get_path_dims(pd)
    id_dims = get_id_dims(pd)
    num_edge_types = len(saved_data["edge_dict"])
    neighbor_feats = saved_data["neighbor_feats"]
    num_nodes = len(neighbor_feats[0])    
    doids = args.ids # If the graph is too big, one-hots for each node becomes infeasible in memory. 
    print("Doing ids? %r" % doids)
    ragged = False
    path_type_dims = get_path_dims(pd)
    id_dims = get_id_dims(pd)
    node_target_pairs, tids = load_target_vecs(args.train_targets, node_dict)
    # Make validation sets. 
    shuffle(node_target_pairs)
    train_val_split = 0.2
    val_idx = int(len(node_target_pairs) * train_val_split)
    val_inputs = tf.convert_to_tensor(node_target_pairs[0:val_idx], dtype=tf.int32)
    train_inputs = tf.convert_to_tensor(node_target_pairs[val_idx:], dtype=tf.int32)
    print("Train and val input shapes: %s and %s" % (str(train_inputs.shape), str(val_inputs.shape)))
    max_batches = 1000
    train_steps = min((ceil((len(node_target_pairs) - val_idx) / batch_size)), max_batches)
    val_steps = min((ceil(val_idx / batch_size)), max_batches * train_val_split)
    
    print("Are we ragged? %r" % ragged)
    feat_in_layers = [Input(shape=(None, path_dim), name="nodef%d" % i) for i, path_dim in enumerate(path_type_dims)]
    id_in_layers = [Input(shape=(None, id_dim), name="idf%d" % i, dtype='int32') for i, id_dim in enumerate(id_dims)]
    in_layers = feat_in_layers + id_in_layers
    print("%d input layers with %d paths" % (len(in_layers), len(path_type_dims)))
    pgcn_layer = PathGCN(path_dims=path_type_dims, id_dims=id_dims, num_nodes=len(node_dict), ragged=ragged, output_dim=args.output_dim, message_passer="dense", activation="relu", l1=0.0001, dropout=0.3, do_ids=doids)
    pgcn = pgcn_layer(in_layers)
    classifier = Dense(units=len(tids), activation="softmax")(pgcn)
    
    print("Building model...")
    model = Model(inputs=in_layers, outputs=classifier)
    model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=Adam(lr=0.0001), metrics=['categorical_accuracy'])
    print(model.summary())
    
    patience = int(epochs / 5)
    log_dir="logs/class/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=patience)
    callbacks = [stopping_callback]
    if(args.logs):
        print("Saving logs to %s" % log_dir)
        callbacks.append(tensorboard_callback)

    
    output_types = tuple([layer.dtype for layer in in_layers])
    
    train_ds = make_ds_from_node_targets(train_inputs, neighbor_feats, pd, output_types, batch_size, tids)
    val_ds = make_ds_from_node_targets(val_inputs, neighbor_feats, pd, output_types, batch_size, tids)

    print("Starting fitting...")
    print("Train and val steps: %d and %d" % (train_steps, val_steps))
    model.fit(x = train_ds, epochs=epochs, validation_data = val_ds, steps_per_epoch=train_steps, validation_steps=val_steps, verbose=1, callbacks=callbacks)  
    print("Done fitting!")  
    test_targets, test_tids = load_target_vecs(args.test_targets, node_dict, tids)
    assert tids == test_tids
    test_ds = make_ds_from_node_targets(tf.convert_to_tensor(test_targets), neighbor_feats, pd, output_types, batch_size, tids)

    results = (model.evaluate(x=test_ds, steps=len(test_targets)))
    print("Results:")
    print(results)
    
if __name__ == "__main__":
    args = parse_args()

    main(args)
