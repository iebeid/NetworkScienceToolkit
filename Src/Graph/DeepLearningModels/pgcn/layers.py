import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
class MLP(Layer):
    def __init__(self, hidden_dim1, hidden_dim2, output_dim, activation="relu", l1=0.001, dropout=0, **kwargs):
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout
        self.l1 = l1
        super(MLP, self).__init__(**kwargs)
    def build(self, input_shape):
        self.h1 = (Dense(units=self.hidden_dim1, activation=self.activation, kernel_regularizer=regularizers.l1(self.l1), bias_regularizer=regularizers.l1(self.l1)))
        self.h2 = (Dense(units=self.hidden_dim2, activation=self.activation, kernel_regularizer=regularizers.l1(self.l1), bias_regularizer=regularizers.l1(self.l1)))
        self.d1 = Dropout(self.dropout)
        self.d2 = Dropout(self.dropout)
        if self.activation is not None:
            self.a_layer = Activation(self.activation)
        else:
            self.a_layer = Lambda(lambda x : x)
        super(MLP, self).build(input_shape)  # Be sure to call this at the end
    def call(self, input_tens):
        x = self.h1(input_tens)
        x = self.d1(x)
        x = self.h2(x)
        x = self.d2(x)
        x = self.a_layer(x)
        return x
    def compute_output_shape(self, input_shape):
        out_dim = (self.output_dim,)
        return out_dim # Batch size * output dimension
class PathGCN(Layer):
    def __init__(self, output_dim, path_dims, id_dims, num_nodes, message_passer, do_ids, ragged = False, activation="relu", l1 = 0.000, threshold=0.1, dropout=0.1, **kwargs):
        self.path_dims = path_dims
        self.id_dims = id_dims
        self.output_dim = output_dim
        self.activation = activation
        self.mp = message_passer
        self.num_nodes = num_nodes
        self.l1 = l1
        self.threshold = threshold
        self.dropout = dropout
        self.ragged = ragged
        self.do_ids = do_ids
        print("PGCN with ids?: %r" % self.do_ids)
        super(PathGCN, self).__init__(**kwargs)
        if ragged:
            self._supports_ragged_inputs = True 
    def build(self, input_shapes):
        # Shape is array of (batch_size, num_neighbors_each_path_type (None), path_input_dim)
        self.path_layers = []
        self.id_layers = []
        self.id_reshape_layers = []
        self.path_weights = self.add_weight(shape=(len(self.path_dims),), trainable=False, initializer='ones', name='path weighs')
        self.dropout_layers = [Dropout(self.dropout) for x in range(len(input_shapes))]
        for idx, path_dim in enumerate(input_shapes[:len(self.path_dims)]):
            # weight = self.add_weight(name="idx%d" % idx, shape=(path_dim[1], path_dim[2], self.output_dim), initializer='uniform', trainable=True)
            #self.path_weights.append(weight)
            id_dim = self.id_dims[idx]
            total_dim = path_dim[2]
            if self.do_ids:
                self.id_reshape_layers.append(Reshape((-1, id_dim * self.num_nodes)))
                total_dim += (id_dim * self.num_nodes)
            if self.mp is "mlp":
                self.path_layers.append(MLP(input_shape=(path_dim[0], path_dim[1], total_dim), hidden_dim1 = 500, hidden_dim2 = 300, output_dim=self.output_dim, activation=self.activation, dropout=self.dropout, l1=self.l1))
            elif self.mp is "dense":
                dlayer = Dense(input_shape=(path_dim[0], path_dim[1], total_dim), units=self.output_dim, kernel_regularizer=regularizers.l1(self.l1), bias_regularizer=regularizers.l1(self.l1))
                self.path_layers.append(dlayer)
            else:
                print("Unknown message passer %s!" % self.mp)
        self.sm_layer = Softmax()
        if self.activation == "None" or self.activation is None:
            self.activation_layer = None
        else:
            self.activation_layer = Activation(self.activation)
        super(PathGCN, self).build(input_shapes)  # Be sure to call this at the end
    def subaggr(self, i, feat_tensor, id_tensor):
        if self.do_ids:
            one_hotted = K.one_hot(id_tensor, self.num_nodes)
            one_hotted_rs = self.id_reshape_layers[i](one_hotted)
            total_in_vec = K.concatenate([one_hotted_rs, feat_tensor])
        else:
            total_in_vec = feat_tensor
        if self.ragged and type(total_in_vec) is tf.RaggedTensor:
            total_in_vec = total_in_vec.to_tensor()
        out = self.path_layers[i](total_in_vec)
        sub_aggregated = tf.keras.backend.mean(out, axis=1)
        return sub_aggregated
    def call(self, inputs):
        outputs_per_path = []
        # Inputs is array of (possibly ragged) tensors, each tensor is (batch_size, num_neighbors_each_path_type, path_input_dim)
        for i in range(int(len(inputs) / 2)):
            feat_tensor = inputs[i]
            id_tensor = inputs[i + len(self.path_dims)]
            subagg = self.dropout_layers[i](self.subaggr(i, feat_tensor, id_tensor))
            outputs_per_path.append(subagg)       

        outputs_tens = tf.stack(outputs_per_path, axis=2)
        # Scale by path weights ##EDIT Don't do this, it severly decrease performance?!?!? For some reason?!?!?!
        aggregated = tf.reduce_sum(outputs_tens, axis=2)
        if self.activation_layer == None:
            return aggregated

        return self.activation_layer(aggregated) # For some reason activating this makes it worse. I don't know, man. 
    def compute_output_shape(self, input_shapes):
        out_dim = (self.output_dim,)
        print("Output dim: %s" % str(out_dim))
        return out_dim # Batch size * output dimension

class TripleEncoder(Layer):
    def __init__(self, pgcn, **kwargs):
        self.pgcn = pgcn
        super(TripleEncoder, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shapes):
        # Input shapes here should be (batch_size, 2, num_neighbors_each_path_type, path/id_dim)
        super(TripleEncoder, self).build(input_shapes) 
    def call(self, inputs):
        head_tens = []
        tail_tens = []
        for in_tens in inputs:
            unstacked = tf.unstack(in_tens, axis=1)
            head_tens.append(unstacked[0])
            tail_tens.append(unstacked[1])
        return [self.pgcn(head_tens), self.pgcn(tail_tens)]
    def compute_output_shape(self, input_shapes):
        return [self.pgcn.output_dim, self.pgcn.output_dim]  

class TripleDiagonalDecoder(Layer):
    def __init__(self, num_types, **kwargs):
        self.num_types = num_types
        super(TripleDiagonalDecoder, self).__init__(**kwargs)
    def build(self, input_shapes):
        encoder_shape = input_shapes[1][1]
        self.diags = self.add_weight(shape=(self.num_types, encoder_shape), trainable=True, initializer='uniform') 
        super(TripleDiagonalDecoder, self).build(input_shapes)  
    def call(self, inputs):
        rel = inputs[0]
        head_embed, tail_embed = tf.expand_dims(inputs[1], axis=1), tf.expand_dims(inputs[2], axis=2)
        raw_diag = (tf.gather(self.diags, rel))
        diag = tf.squeeze(raw_diag, axis=1)
        # Diag is a 1xd tensor that represents the diagonal. Turn it into a dxd matrix.
        mat = tf.linalg.diag(diag) 
        score = tf.matmul(tf.matmul(head_embed, mat), tail_embed)
        final = tf.squeeze(tf.squeeze(score, axis=2), axis=1)
        return final
    def compute_output_shape(self, input):
        return 1
class TripleMatrixDecoder(Layer):
    def __init__(self, num_types, **kwargs):
        self.num_types = num_types
        super(TripleMatrixDecoder, self).__init__(**kwargs)
    def build(self, input_shapes):
        encoder_shape = input_shapes[1][1]
        self.mats = self.add_weight(shape=(self.num_types, encoder_shape, encoder_shape), trainable=True, initializer='uniform') 
        super(TripleMatrixDecoder, self).build(input_shapes)  
    def call(self, inputs):
        rel = inputs[0]
        head_embed, tail_embed = tf.expand_dims(inputs[1], axis=1), tf.expand_dims(inputs[2], axis=2)
        mat = tf.squeeze((tf.gather(self.mats, rel)), axis=1)
        score = tf.matmul(tf.matmul(head_embed, diag), tail_embed)
        final = tf.squeeze(tf.squeeze(score, axis=2), axis=1)
        return final
    def compute_output_shape(self, input):
        return 1
