import os 
os.environ["KERAS_BACKEND"] = "jax"

import keras as ks
import numpy as np

train_features = np.load('/gpfs/scratch/blukacsy/train_features.npy')
val_features = np.load('/gpfs/scratch/blukacsy/val_features.npy')
train_labels = np.load('/gpfs/scratch/blukacsy/train_labels.npy')
val_labels = np.load('/gpfs/scratch/blukacsy/val_labels.npy')
sample_weights = np.load('/gpfs/scratch/blukacsy/sample_weights.npy')

def create_model(input_size, num_classes, hidden_layers, dropout_rate, learning_rate_schedule, weight_decay):

    model = ks.Sequential()
    model.add(ks.Input(shape=(input_size,)))
    model.add(ks.layers.BatchNormalization())

    for units in hidden_layers:
        model.add(ks.layers.Dense(units, activation=ks.layers.LeakyReLU(negative_slope=0.02)))
        model.add(ks.layers.BatchNormalization())
        model.add(ks.layers.Dropout(dropout_rate))

    model.add(ks.layers.Dense(num_classes, activation=ks.layers.Softmax()))

    model.compile(
        optimizer=ks.optimizers.AdamW(learning_rate=learning_rate_schedule, weight_decay=weight_decay), 
        loss=ks.losses.SparseCategoricalCrossentropy(), 
        metrics=[ks.metrics.SparseCategoricalAccuracy(name="accuracy")], 
        weighted_metrics=[ks.metrics.SparseCategoricalAccuracy(name="weighted_accuracy")]
    )

    return model

input_size = train_features.shape[1]
num_classes = len(np.unique(train_labels))
hidden_layers = [1024, 256]
dropout_rate = 0.5
learning_rate = 1e-3
weight_decay = 1e-4
epochs = 1000
warmup_epochs = 100
batch_size = 256

steps_per_epoch = int(np.ceil(len(train_features)/batch_size))
total_steps = epochs * steps_per_epoch
warmup_steps = warmup_epochs * steps_per_epoch
decay_steps = total_steps - warmup_steps
learning_rate_schedule = ks.optimizers.schedules.CosineDecay(initial_learning_rate=0.0, decay_steps=decay_steps, alpha=0.0, warmup_target=learning_rate, warmup_steps=warmup_steps)

model = create_model(input_size, num_classes, hidden_layers, dropout_rate, learning_rate_schedule, weight_decay)

train_dataset = ks.data.Dataset.from_tensor_slices((train_features, train_labels, sample_weights)).cache().shuffle(len(train_features)).batch(batch_size).prefetch(ks.data.AUTOTUNE)
val_dataset = ks.data.Dataset.from_tensor_slices((val_features, val_labels)).cache().batch(batch_size).prefetch(ks.data.AUTOTUNE)

model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=2)

model.save("/gpfs/scratch/blukacsy/granulomas30_jax_v1.keras") 