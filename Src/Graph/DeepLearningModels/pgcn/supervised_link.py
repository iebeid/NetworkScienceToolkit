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
from functools import *


import datetime

from math import ceil, floor, pow 

def parse_args():
    parser = argparse.ArgumentParser(description="Train on link prediction task.") 
    parser.add_argument('--input-data', nargs='?', default='emb/Toy/stackmeta.pbz2',
                        help='Input neighbor and pata data')
    parser.add_argument('--output-dim', type=int, default=500, help='Dimension of embedding vectors.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (number of nodes to process at once)')
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--pos-targets', nargs='?', default='data/Toy/train_pos.csv')
    parser.add_argument('--neg-targets', nargs='?', default='data/Toy/train_neg.csv')
    parser.add_argument('--test-pos-targets', nargs='?', default='data/Toy/test_pos.csv')
    parser.add_argument('--test-neg-targets', nargs='?', default='data/Toy/test_neg.csv')
    parser.add_argument('--ids', dest='ids', action='store_true',
                        help='Boolean specifying if to use node ids when training (one-hot vectors for each node). Default is false.') 
    parser.add_argument('--nologs', dest='logs', action='store_false',
                        help='Boolean specifying NOT to make tensorboard logs. Default is to make logs. ')
    parser.add_argument('--nofeats', dest='nofeats', action='store_false',
                        help='Boolean specifying NOT to use node features. Turn on ids if you enable this. Default is to use node features. ')
    parser.set_defaults(ids=False)
    parser.set_defaults(logs=True)
    parser.set_defaults(nofeats=False)
    return parser.parse_args()

def decoder_loss(labels, predicted_score):
    neg_mul = tf.ones_like(labels) - labels
    pos_mul = labels
    losses = (pos_mul * tf.math.log(predicted_score)) + (neg_mul * tf.math.log(1 - predicted_score))
    final = -tf.reduce_mean(losses)
    return final

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

def process_triple_path(triple, neighbor_feats, dims, pathid, nofeats, num_path_types):
    trip = triple.numpy()
    head = trip[0]
    tail = trip[2]
    headf = neighbor_feats[pathid][head]
    tailf = neighbor_feats[pathid][tail]
    if nofeats and pathid < num_path_types:
        headf = tf.zeros_like(headf)
        tailf = tf.zeros_like(tailf)
    dim = dims[pathid]
    dtype = tf.float32
    if pathid >= (len(dims) / 2):
        dtype = tf.int32
    if headf == None:
        headf = tf.zeros((1, dim), dtype=dtype)
    if tailf == None:
        tailf = tf.zeros((1, dim), dtype=dtype)
    return tf.ragged.stack([headf, tailf], axis=0).to_tensor()

def make_ds_from_triples(triples, neighbor_feats, pd, output_types, batch_size, nofeats):
    print("Triples given with input shape %s" % str(triples.shape)) 
    process_pyfunc = (lambda triple: process_triple_for_dataset(neighbor_feats, pd, triple))
    triple_ds = tf.data.Dataset.from_tensor_slices(triples).shuffle(1000, reshuffle_each_iteration=True).repeat(None)
    edge_ds = triple_ds.map(lambda triple: tf.cast(triple[1], tf.int32), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    path_type_dims = get_path_dims(pd)
    num_path_types = len(path_type_dims)
    id_dims = get_id_dims(pd)
    dims = path_type_dims + id_dims
    dtypes = ([tf.float32] * num_path_types) + ([tf.int32] * num_path_types)
    triple_p_ds = []
    path_pyfuncs = [(partial(process_triple_path, neighbor_feats=neighbor_feats, dims=dims, pathid=pathid, nofeats=nofeats, num_path_types=num_path_types)) for pathid in range(num_path_types * 2)]
    for pathid in range(num_path_types * 2):
        dtype = dtypes[pathid]
        p_ds = triple_ds.map(lambda triple: tf.py_function(func = path_pyfuncs[pathid], inp=[triple], Tout=dtype), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        shape = (2, None, dims[pathid])
        p_ds = p_ds.map(lambda tens: tf.ensure_shape(tens, shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        triple_p_ds.append(p_ds)
    neighbor_ds = tf.data.Dataset.zip(tuple([edge_ds] + triple_p_ds))
    target_ds = triple_ds.map(lambda triple: (tf.cast(triple[3], tf.float32)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = tf.data.Dataset.zip((neighbor_ds, target_ds))
    ds = ds.padded_batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def main(args):
    batch_size = args.batch_size

    epochs = args.epochs
    patience = int(epochs / 5)
    print("Loading neighbor feats...")
    saved_data = load_cpickle(args.input_data)
    pd = saved_data["path_data"]
    path_type_dims = get_path_dims(pd)
    id_dims = get_id_dims(pd)
    num_edge_types = len(saved_data["edge_dict"])
    neighbor_feats = saved_data["neighbor_feats"]
    num_nodes = len(neighbor_feats[0])
    print("Loading triples from files...")
    pos_triples = load_triples(args.pos_targets)
    neg_triples = load_triples(args.neg_targets)
    total_triples = []
    print("Splitting train/test triples")
    for p, triples in enumerate([neg_triples, pos_triples]):
        for triple in triples:
            total_triples.append((triple[0], triple[1], triple[2], p)) 
    shuffle(total_triples)
    del pos_triples
    del neg_triples

    test_pos_triples = load_triples(args.test_pos_targets)
    test_neg_triples = load_triples(args.test_neg_targets)
    test_triples = []
    for p, triples in enumerate([test_neg_triples, test_pos_triples]):
        for triple in triples:
            test_triples.append((triple[0], triple[1], triple[2], p)) 
    shuffle(test_triples)
    has_test = True
    if len(test_triples) == 0:
        print("------WARNING! No triples specified as testing triples! Manually creating testing triples--------")
        has_test = False
    del test_pos_triples
    del test_neg_triples


    # Make validation sets..
    if not has_test:
        test_split = 0.1
        test_idx = int(len(total_triples) * test_split)
        test_triples = total_triples[:test_idx]
        total_triples = total_triples[test_idx:]

    val_split = 0.1
    val_idx = int(len(total_triples) * val_split)
    val_triples = tf.convert_to_tensor(total_triples[:val_idx], dtype=tf.int32)
    train_triples = tf.convert_to_tensor(total_triples[val_idx:], dtype=tf.int32)
    test_triple_tens = tf.convert_to_tensor(test_triples, dtype=tf.int32)
    max_batches = 100
    do_ids = args.ids or args.nofeats

    train_steps = min(ceil(len(train_triples) / batch_size), max_batches)
    val_steps = min(ceil(len(val_triples) / batch_size), max_batches * val_split)
    print("Doing %d and %d steps per train and test epoch" % (train_steps, val_steps))
    edge_type_layer = Input(shape=(1,), dtype='int32')
    feat_in_layers = [Input(shape=(2, None, path_dim), name="nodef%d" % i, batch_size = batch_size) for i, path_dim in enumerate(path_type_dims)]
    id_in_layers = [Input(shape=(2, None, id_dim), name="idf%d" % i, dtype='int32', batch_size=batch_size) for i, id_dim in enumerate(id_dims)]
    encoder_in_layers = feat_in_layers + id_in_layers
    in_layers = [edge_type_layer] + encoder_in_layers
    print("%d input layers with %d paths" % (len(in_layers), len(path_type_dims)))
    pgcn = PathGCN(path_dims=path_type_dims, id_dims=id_dims, num_nodes=num_nodes, output_dim=args.output_dim, message_passer="dense", dropout=0.3, do_ids = do_ids)
    encoder = TripleEncoder(pgcn=pgcn)(encoder_in_layers)
    decoder = TripleDiagonalDecoder(num_types=num_edge_types)([edge_type_layer] + encoder)
    normalization = Activation("sigmoid")(decoder)
    
    print("Building model...")
    model = Model(inputs=in_layers, outputs=normalization)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=Adam(lr=0.0005), metrics=['binary_accuracy'])
    print(model.summary())
    
    patience = int(epochs / 5)
    log_dir="logs/link/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    callbacks = [stopping_callback]
    if(args.logs):
        print("Logging into %s" % log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='5, 25')
        callbacks.append(tensorboard_callback)

    print("Prepping data")
    output_types = tuple([layer.dtype for layer in in_layers])
    nofeats = args.nofeats
    train_ds = make_ds_from_triples(train_triples, neighbor_feats, pd, output_types, batch_size, nofeats)
    val_ds = make_ds_from_triples(val_triples, neighbor_feats, pd, output_types, batch_size, nofeats)
    test_ds = make_ds_from_triples(test_triple_tens, neighbor_feats, pd, output_types, batch_size, nofeats)
    print("Starting fitting...")
    model.fit(x = train_ds, epochs=epochs, validation_data=val_ds, validation_steps = val_steps, steps_per_epoch=train_steps, verbose=1, callbacks=callbacks)    
    print("Done fitting")
    test_steps = min(ceil(len(test_triples) / batch_size), max_batches)
    base_results = (model.evaluate(x = test_ds, steps=test_steps))

    print("Base Results:")
    print(base_results)

    
if __name__ == "__main__":
    args = parse_args()

    main(args)
