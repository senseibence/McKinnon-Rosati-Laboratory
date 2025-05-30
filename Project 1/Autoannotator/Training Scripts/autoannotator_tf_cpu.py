import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers

train_features = np.load('../arrays/train_features.npy')
val_features = np.load('../arrays/val_features.npy')
train_labels = np.load('../arrays/train_labels.npy')
val_labels = np.load('../arrays/val_labels.npy')
sample_weights = np.load('../arrays/sample_weights.npy')

def create_model(input_size, num_classes, hidden_layers, dropout_rate, l2_reg):

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
    return model

input_size = train_features.shape[1] # 23693
num_classes = len(np.unique(train_labels)) # 30
hidden_layers = [8192, 4096, 2048, 1024]
dropout_rate = 0.5
l2_reg = 1e-4
learning_rate=1e-5
epochs = 1000
batch_size = 512

model = create_model(input_size, num_classes, hidden_layers, dropout_rate, l2_reg)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(train_features, train_labels, sample_weight=sample_weights, validation_data=(val_features, val_labels), epochs=epochs, batch_size=batch_size, verbose=2)

model.save("/gpfs/scratch/blukacsy/granulomas_final_tf_nn_v1.h5")