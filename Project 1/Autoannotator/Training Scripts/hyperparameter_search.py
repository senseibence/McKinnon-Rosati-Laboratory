import os
os.environ["KERAS_BACKEND"] = "jax"

import json
import keras as ks
import jax
import numpy as np
from tensorflow import data
import optuna
from optuna.integration import KerasPruningCallback

ks.mixed_precision.set_global_policy("mixed_float16")

devices = jax.devices("gpu")
data_parallel = ks.distribution.DataParallel(devices=devices)
ks.distribution.set_distribution(data_parallel)

train_features = np.load("/gpfs/scratch/blukacsy/granulomas30_train_features.npy")
val_features = np.load("/gpfs/scratch/blukacsy/granulomas30_val_features.npy")
train_labels = np.load("/gpfs/scratch/blukacsy/granulomas30_train_labels.npy")
val_labels = np.load("/gpfs/scratch/blukacsy/granulomas30_val_labels.npy")
weights = np.load("/gpfs/scratch/blukacsy/granulomas30_weights.npy")
class_weights = dict(enumerate(weights))

features = train_features.shape[1]
num_samples = train_features.shape[0]
num_classes = len(np.unique(train_labels))
epochs = 1000

def create_model(params):

    hidden_layers = params["hidden_layers"]
    dropout_rate = params["dropout_rate"]
    negative_slope = params["negative_slope"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]
    warmup_epochs = params["warmup_epochs"]

    model = ks.Sequential()
    model.add(ks.Input(shape=(features,)))
    model.add(ks.layers.BatchNormalization())

    for units in hidden_layers:
        model.add(ks.layers.Dense(units, activation=ks.layers.LeakyReLU(negative_slope=negative_slope)))
        model.add(ks.layers.BatchNormalization())
        if (dropout_rate > 0): model.add(ks.layers.Dropout(dropout_rate))

    model.add(ks.layers.Dense(num_classes))
    model.add(ks.layers.Activation("softmax", dtype="float32"))

    steps_per_epoch = int(np.ceil(len(train_features)/batch_size))
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    decay_steps = total_steps - warmup_steps
    learning_rate_schedule = ks.optimizers.schedules.CosineDecay(initial_learning_rate=0.0, decay_steps=decay_steps, alpha=0.0, warmup_target=learning_rate, warmup_steps=warmup_steps)

    model.compile(
        optimizer=ks.optimizers.AdamW(learning_rate=learning_rate_schedule, weight_decay=weight_decay),
        loss=ks.losses.SparseCategoricalCrossentropy(),
        metrics=[ks.metrics.SparseCategoricalAccuracy(name="accuracy")],
        weighted_metrics=[ks.metrics.SparseCategoricalAccuracy(name="weighted_accuracy")]
    )

    return model

def objective(trial):

    depth = trial.suggest_int("depth", 1, 4)
    hidden_layers = [trial.suggest_int(f"units_{i}", num_classes, features, log=True) for i in range(depth)]

    hyperparameters = {
        "hidden_layers" : hidden_layers,
        "dropout_rate" : trial.suggest_float("dropout_rate", 0, 0.95),
        "negative_slope" : trial.suggest_float("negative_slope", 1e-4, 5e-1, log=True),
        "learning_rate" : trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
        "weight_decay" : trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "batch_size" : trial.suggest_int("batch_size", 1, num_samples, log=True),
        "warmup_epochs" : trial.suggest_int("warmup_epochs", 5, 50)
    }

    model = create_model(hyperparameters)
    batch_size = hyperparameters["batch_size"]

    train_dataset = data.Dataset.from_tensor_slices((train_features, train_labels)).cache().shuffle(len(train_features), reshuffle_each_iteration=True).batch(batch_size).prefetch(data.AUTOTUNE)
    val_dataset = data.Dataset.from_tensor_slices((val_features, val_labels)).cache().batch(batch_size).prefetch(data.AUTOTUNE)

    early_stopping_callback = ks.callbacks.EarlyStopping(monitor="val_loss", patience=200, verbose=0, restore_best_weights=True, start_from_epoch=0)
    pruning_callback = KerasPruningCallback(trial, "val_loss")

    history = model.fit(train_dataset, validation_data=val_dataset, class_weight=class_weights, epochs=epochs, callbacks=[early_stopping_callback, pruning_callback], verbose=0)

    loss = min(history.history["val_loss"])
    ks.backend.clear_session()
    return loss

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(multivariate=True, group=False), pruner=optuna.pruners.HyperbandPruner(min_resource=50, max_resource=epochs, reduction_factor=3))
study.optimize(objective, n_trials=100)

print("val_loss:", study.best_value)
print("hyperparameters:", study.best_params)

params = study.best_params
hidden_layers = [params[f"units_{i}"] for i in range(params["depth"])]
params["hidden_layers"] = hidden_layers
batch_size = params["batch_size"]

model = create_model(params)

train_features = np.concatenate([train_features, val_features])
train_labels = np.concatenate([train_labels, val_labels])
train_dataset = data.Dataset.from_tensor_slices((train_features, train_labels)).cache().shuffle(len(train_features), reshuffle_each_iteration=True).batch(batch_size).prefetch(data.AUTOTUNE)

early_stopping_callback = ks.callbacks.EarlyStopping(monitor="loss", patience=200, verbose=1, restore_best_weights=True, start_from_epoch=0)
model.fit(train_dataset, class_weight=class_weights, epochs=epochs, callbacks=[early_stopping_callback], verbose=2)

model.save("/gpfs/scratch/blukacsy/granulomas30_jax_v1.keras")

with open("/gpfs/scratch/blukacsy/granulomas30_hyperparameters.json", "w") as file: 
    json.dump(params, file, indent=4)
