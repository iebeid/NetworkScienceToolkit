import stellargraph as sg
from stellargraph.mapper import CorruptedGenerator, FullBatchNodeGenerator
from stellargraph.layer import GCN, DeepGraphInfomax

import pandas as pd
from sklearn import model_selection, preprocessing
from IPython.display import display, HTML

import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, callbacks

dataset = sg.datasets.Cora()
display(HTML(dataset.description))
G, node_classes = dataset.load()


print(G.info())

fullbatch_generator = FullBatchNodeGenerator(G)

corrupted_generator = CorruptedGenerator(fullbatch_generator)
gen = corrupted_generator.flow(G.nodes())

def make_gcn_model():
    # function because we want to create a second one with the same parameters later
    return GCN(
        layer_sizes=[16, 16],
        activations=["relu", "relu"],
        generator=fullbatch_generator,
        dropout=0.4,
    )


pretrained_gcn_model = make_gcn_model()


infomax = DeepGraphInfomax(pretrained_gcn_model, corrupted_generator)
x_in, x_out = infomax.in_out_tensors()

dgi_model = Model(inputs=x_in, outputs=x_out)
dgi_model.compile(
    loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=optimizers.Adam(lr=1e-3)
)

epochs = 500

dgi_es = callbacks.EarlyStopping(monitor="loss", patience=50, restore_best_weights=True)
dgi_history = dgi_model.fit(gen, epochs=epochs, verbose=0, callbacks=[dgi_es])

sg.utils.plot_history(dgi_history)

node_classes.value_counts().to_frame()

train_classes, test_classes = model_selection.train_test_split(
    node_classes, train_size=8, stratify=node_classes, random_state=1
)
val_classes, test_classes = model_selection.train_test_split(
    test_classes, train_size=500, stratify=test_classes
)

train_classes.value_counts().to_frame()

target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_classes)
val_targets = target_encoding.transform(val_classes)
test_targets = target_encoding.transform(test_classes)

train_gen = fullbatch_generator.flow(train_classes.index, train_targets)
test_gen = fullbatch_generator.flow(test_classes.index, test_targets)
val_gen = fullbatch_generator.flow(val_classes.index, val_targets)

pretrained_x_in, pretrained_x_out = pretrained_gcn_model.in_out_tensors()

pretrained_predictions = tf.keras.layers.Dense(
    units=train_targets.shape[1], activation="softmax"
)(pretrained_x_out)

pretrained_model = Model(inputs=pretrained_x_in, outputs=pretrained_predictions)
pretrained_model.compile(
    optimizer=optimizers.Adam(lr=0.01), loss="categorical_crossentropy", metrics=["acc"],
)

prediction_es = callbacks.EarlyStopping(
    monitor="val_acc", patience=50, restore_best_weights=True
)

pretrained_history = pretrained_model.fit(
    train_gen,
    epochs=epochs,
    verbose=0,
    validation_data=val_gen,
    callbacks=[prediction_es],
)

sg.utils.plot_history(pretrained_history)

