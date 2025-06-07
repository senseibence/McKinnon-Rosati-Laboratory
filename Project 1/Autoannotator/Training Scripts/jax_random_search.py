import os
os.environ["KERAS_BACKEND"] = "jax"

import json
import time
import random
import numpy as np
import keras as ks
import jax
from tensorflow import data
from sklearn.model_selection import KFold

ks.mixed_precision.set_global_policy("mixed_float16")

devices = jax.devices("gpu")
data_parallel = ks.distribution.DataParallel(devices=devices)
ks.distribution.set_distribution(data_parallel)

DATA_DIR = "/gpfs/scratch/blukacsy"

train_features = np.load(os.path.join(DATA_DIR, "train_features.npy"))
val_features = np.load(os.path.join(DATA_DIR, "val_features.npy"))
train_labels = np.load(os.path.join(DATA_DIR, "train_labels.npy"))
val_labels = np.load(os.path.join(DATA_DIR, "val_labels.npy"))
weights = np.load(os.path.join(DATA_DIR, "weights.npy"))
class_weights = dict(enumerate(weights))

all_features = np.concatenate([train_features, val_features], axis=0)
all_labels = np.concatenate([train_labels, val_labels], axis=0)


def create_model(input_size, num_classes, hidden_layers, dropout_rate,
                 lr_schedule, weight_decay):
    model = ks.Sequential()
    model.add(ks.Input(shape=(input_size,)))
    model.add(ks.layers.BatchNormalization())
    for units in hidden_layers:
        model.add(ks.layers.Dense(units,
                                  activation=ks.layers.LeakyReLU(negative_slope=0.02)))
        model.add(ks.layers.BatchNormalization())
        model.add(ks.layers.Dropout(dropout_rate))
    model.add(ks.layers.Dense(num_classes))
    model.add(ks.layers.Activation("softmax", dtype="float32"))

    model.compile(
        optimizer=ks.optimizers.AdamW(learning_rate=lr_schedule,
                                      weight_decay=weight_decay),
        loss=ks.losses.SparseCategoricalCrossentropy(),
        metrics=[ks.metrics.SparseCategoricalAccuracy(name="accuracy")],
        weighted_metrics=[ks.metrics.SparseCategoricalAccuracy(name="weighted_accuracy")]
    )
    return model


def sample_hyperparams():
    depth = random.randint(1, 4)
    hidden_layers = [random.choice([64, 128, 256, 512, 1024, 2048, 4096])
                     for _ in range(depth)]
    dropout = random.uniform(0.2, 0.9)
    lr = 10 ** random.uniform(-6, -3)
    weight_decay = 10 ** random.uniform(-6, -3)
    batch_size = random.choice([128, 256, 512, 1024])
    warmup_epochs = random.randint(5, 50)
    epochs = random.randint(200, 1000)
    return {
        "hidden_layers": hidden_layers,
        "dropout": dropout,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "warmup_epochs": warmup_epochs,
        "epochs": epochs,
    }


def train_and_evaluate(params, fold_data=None):
    if fold_data is None:
        X_train, y_train = train_features, train_labels
        X_val, y_val = val_features, val_labels
    else:
        X_train, y_train, X_val, y_val = fold_data

    input_size = X_train.shape[1]
    num_classes = len(np.unique(train_labels))

    steps_per_epoch = int(np.ceil(len(X_train) / params["batch_size"]))
    total_steps = params["epochs"] * steps_per_epoch
    warmup_steps = params["warmup_epochs"] * steps_per_epoch
    decay_steps = total_steps - warmup_steps
    lr_schedule = ks.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=decay_steps,
        alpha=0.0,
        warmup_target=params["learning_rate"],
        warmup_steps=warmup_steps,
    )

    model = create_model(input_size, num_classes, params["hidden_layers"],
                         params["dropout"], lr_schedule, params["weight_decay"])

    train_ds = (data.Dataset.from_tensor_slices((X_train, y_train))
                .cache()
                .shuffle(len(X_train), reshuffle_each_iteration=True)
                .batch(params["batch_size"])
                .prefetch(data.AUTOTUNE))
    val_ds = (data.Dataset.from_tensor_slices((X_val, y_val))
              .cache()
              .batch(params["batch_size"])
              .prefetch(data.AUTOTUNE))

    callback = ks.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, verbose=0,
        restore_best_weights=True, start_from_epoch=50
    )

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        class_weight=class_weights,
                        epochs=params["epochs"],
                        callbacks=[callback],
                        verbose=0)

    val_loss = min(history.history["val_loss"])
    return model, val_loss


def run_search(runtime_hours=48, k_folds=None):
    start = time.time()
    best_loss = float("inf")
    best_model = None
    best_params = None
    trial = 0

    while time.time() - start < runtime_hours * 3600:
        params = sample_hyperparams()
        trial += 1
        if k_folds:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            losses = []
            for train_idx, val_idx in kf.split(all_features):
                fold_data = (all_features[train_idx], all_labels[train_idx],
                             all_features[val_idx], all_labels[val_idx])
                _, fold_loss = train_and_evaluate(params, fold_data)
                losses.append(fold_loss)
            val_loss = float(np.mean(losses))
            model, _ = train_and_evaluate(params)
        else:
            model, val_loss = train_and_evaluate(params)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_params = params
            model.save(os.path.join(DATA_DIR, "best_jax_model.keras"))
            with open(os.path.join(DATA_DIR, "best_jax_params.json"), "w") as f:
                json.dump(best_params, f, indent=2)
        print(f"Trial {trial}: val_loss={val_loss:.4f}, best={best_loss:.4f}")

    return best_model, best_params


if __name__ == "__main__":
    run_search()

