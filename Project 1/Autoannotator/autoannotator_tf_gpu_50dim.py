import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers

# scanorama or pca
train_features = np.load('../arrays/train_features_scanorama.npy')
val_features = np.load('../arrays/val_features_scanorama.npy')
train_labels = np.load('../arrays/train_labels_scanorama.npy')
val_labels = np.load('../arrays/val_labels_scanorama.npy')
sample_weights = np.load('../arrays/sample_weights_scanorama.npy')

def create_model(input_size, num_classes, hidden_layers, dropout_rate, l2_reg, learning_rate):

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_size,)))
    model.add(layers.BatchNormalization())

    for units in hidden_layers:

        model.add(
            layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_reg)
            )
        )

        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

input_size = train_features.shape[1] # 50
num_classes = len(np.unique(train_labels)) # 30
hidden_layers = [256, 128, 64]
dropout_rate = 0.2
l2_reg = 1e-4
learning_rate = 1e-4
epochs = 250
batch_size = 256

strategy = tf.distribute.MirroredStrategy()
print("# GPUs: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = create_model(input_size, num_classes, hidden_layers, dropout_rate, l2_reg, learning_rate)

global_batch_size = batch_size * strategy.num_replicas_in_sync
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels, sample_weights)).batch(global_batch_size).cache().prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).batch(global_batch_size).cache().prefetch(tf.data.AUTOTUNE)

model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=2)

model.save("/gpfs/scratch/blukacsy/granulomas_final_tf_nn_50dim_scanorama_v1.h5")

