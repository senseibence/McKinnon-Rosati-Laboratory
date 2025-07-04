import os
os.environ["KERAS_BACKEND"] = "jax"

import json
import sys
import keras as ks
import jax
import numpy as np
from tensorflow import data
from sklearn.utils.class_weight import compute_class_weight
import optuna
from optuna.integration import KerasPruningCallback

ks.mixed_precision.set_global_policy("mixed_float16")

devices = jax.devices("gpu")
data_parallel = ks.distribution.DataParallel(devices=devices)
ks.distribution.set_distribution(data_parallel)

inputs = sys.argv[1:]
dataset_name = str(inputs[0])
group_name = str(inputs[1])

train_features = np.load(f"/gpfs/scratch/blukacsy/{dataset_name}_train_features_hvg_{group_name}.npy")
val_features = np.load(f"/gpfs/scratch/blukacsy/{dataset_name}_val_features_hvg_{group_name}.npy")
train_labels = np.load(f"/gpfs/scratch/blukacsy/{dataset_name}_train_labels_hvg_{group_name}.npy")
val_labels = np.load(f"/gpfs/scratch/blukacsy/{dataset_name}_val_labels_hvg_{group_name}.npy")
weights = np.load(f"/gpfs/scratch/blukacsy/{dataset_name}_weights_hvg_{group_name}.npy")
class_weights = dict(enumerate(weights))

num_classes = len(np.unique(train_labels))
num_samples = train_features.shape[0]
features = train_features.shape[1]

core_train_dataset = data.Dataset.from_tensor_slices((train_features, train_labels)).cache().shuffle(num_samples, reshuffle_each_iteration=True)
core_val_dataset = data.Dataset.from_tensor_slices((val_features, val_labels)).cache()

epochs = 1000
min_batch_size = max(1, num_samples//500)

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

    model.add(ks.layers.Dense(num_classes, dtype="float32"))

    steps_per_epoch = int(np.ceil(num_samples/batch_size))
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    decay_steps = total_steps - warmup_steps
    learning_rate_schedule = ks.optimizers.schedules.CosineDecay(initial_learning_rate=0.0, decay_steps=decay_steps, alpha=0.0, warmup_target=learning_rate, warmup_steps=warmup_steps)

    model.compile(
        optimizer=ks.optimizers.AdamW(learning_rate=learning_rate_schedule, weight_decay=weight_decay),
        loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
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
        "batch_size" : trial.suggest_int("batch_size", min_batch_size, num_samples, log=True),
        "warmup_epochs" : trial.suggest_int("warmup_epochs", 5, 50)
    }

    model = create_model(hyperparameters)
    batch_size = hyperparameters["batch_size"]

    train_dataset = core_train_dataset.batch(batch_size).prefetch(data.AUTOTUNE)
    val_dataset = core_val_dataset.batch(batch_size).prefetch(data.AUTOTUNE)

    early_stopping_callback = ks.callbacks.EarlyStopping(monitor="val_loss", patience=200, verbose=0, restore_best_weights=True, start_from_epoch=0)
    pruning_callback = KerasPruningCallback(trial, "val_loss")

    history = model.fit(train_dataset, validation_data=val_dataset, class_weight=class_weights, epochs=epochs, callbacks=[early_stopping_callback, pruning_callback], verbose=0)

    val_loss_history = history.history["val_loss"]
    best_epoch = int(np.argmin(val_loss_history))
    trial.set_user_attr("best_epoch", best_epoch)
    loss = min(val_loss_history)

    ks.backend.clear_session()
    return loss

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False), pruner=optuna.pruners.HyperbandPruner(min_resource=50, max_resource=epochs, reduction_factor=3))
study.optimize(objective, n_trials=300)

params = study.best_params
hidden_layers = [params[f"units_{i}"] for i in range(params["depth"])]
params["hidden_layers"] = hidden_layers
batch_size = params["batch_size"]

train_features = np.concatenate([train_features, val_features])
train_labels = np.concatenate([train_labels, val_labels])
num_samples = train_features.shape[0]

train_dataset = data.Dataset.from_tensor_slices((train_features, train_labels)).cache().shuffle(num_samples, reshuffle_each_iteration=True).batch(batch_size).prefetch(data.AUTOTUNE)

weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
weights = np.array(weights)
class_weights = dict(enumerate(weights))

model = create_model(params)

epochs = study.best_trial.user_attrs["best_epoch"] + 1
model.fit(train_dataset, class_weight=class_weights, epochs=epochs, verbose=2)

model.save(f"/gpfs/scratch/blukacsy/{dataset_name}_hvg_{group_name}_jax_v1.keras")

with open(f"/gpfs/scratch/blukacsy/{dataset_name}_hvg_{group_name}_hyperparameters.json", "w") as file:
    json.dump(params, file, indent=4)