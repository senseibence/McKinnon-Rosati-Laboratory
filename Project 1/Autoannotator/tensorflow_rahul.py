"""
Tensorflow + ReLU

1. Loads and annotates the .h5ad file using Scanpy.
2. Maps cluster labels to dict annotations.
3. Splits the data into training, validation, and test sets.
4. Normalizes the data using StandardScaler.
5. Trains a neural network with dropout, ReLU activation, early stopping, and sample weights.
6. Evaluates performance on the test set and plots a confusion matrix.
7. Saves the trained model as 'tensorflow_rm_neural_network_model.h5'.


"""
import numpy as np
import scanpy as sc
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.sparse
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------------------------
# 1. Load the dataset and annotate it
# ------------------------------------------------------------------

clustered_file = ""

# Read the .h5ad file
adata = sc.read_h5ad(clustered_file)
adata.uns['log1p']["base"] = None  # Bug fix for log1p

cluster_type = 'my_clust_1'

# Need to change depending on dataset
annotation_dict = {
    '9': 'CAP1',
    '24': 'CAP2',
    '9b': 'VEC',
    '27': 'LEC',

    '17': 'Ciliated',
    '15': 'Secretory',
    '22': 'AT1',
    '6': 'AT2',
    '12': 'AT2-t1',
    '19': 'AT2-t2',

    '18': 'AF1',
    '14': 'AF2',
    '25': 'Pericyte',

    '20': 'Mesothelial',

    '3': 'B1',
    '3b': 'B2',

    '0': 'Th1',
    '8': 'T_naive',
    '11': 'T_ex',
    '77': 'Treg',

    '11b': 'NK',

    '4a': 'AM',
    '4': 'M-t1',
    '10': 'M-lc',
    '7': 'M-t2',
    '7b': 'M-C1q',
    '7c': 'iMon',

    '23': 'pDC',
    '13': 'DC',
    '5b': 'N1',
    '5': 'N2',
}

# Apply the annotation dictionary
adata.obs['cell_type_edit'] = [annotation_dict[clust] for clust in adata.obs[cluster_type]]

# Map cell types to integer labels (the "replacement_dict")
replacement_dict = {
    'AT2': 0,
    'B1': 1,
    'M-t1': 2,
    'DC': 3,
    'Th1': 4,
    'M-t2': 5,
    'Secretory': 6,
    'AM': 7,
    'N1': 8,
    'M-C1q': 9,
    'AT2-t2': 10,
    'AF2': 11,
    'VEC': 12,
    'CAP1': 13,
    'N2': 14,
    'AT2-t1': 15,
    'Pericyte': 16,
    'pDC': 17,
    'Ciliated': 18,
    'NK': 19,
    'AT1': 20,
    'T_naive': 21,
    'Treg': 22,
    'M-lc': 23,
    'Mesothelial': 24,
    'T_ex': 25,
    'CAP2': 26,
    'LEC': 27,
    'iMon': 28,
    'B2': 29
}

adata.obs['celltype'] = adata.obs['cell_type_edit'].replace(replacement_dict)

# ------------------------------------------------------------------
# 2. Data exploration (optional prints)
# ------------------------------------------------------------------

# Quick checks
print("Unique cell types:", adata.obs['cell_type_edit'].unique())
print("Number of unique cell types:", adata.obs['cell_type_edit'].nunique())
print("Unique values in 'celltype':", adata.obs['celltype'].unique())

# Count the number of cells in each cell type
cell_type_counts = adata.obs['cell_type_edit'].value_counts()
print("\nNumber of cells in each cell type:")
print(cell_type_counts)

# ------------------------------------------------------------------
# 3. Prepare the data for training
# ------------------------------------------------------------------

# Extract gene expression values (X) and labels (y)
# Convert the data to dense array if it's sparse
if scipy.sparse.issparse(adata.X):
    X = adata.X.toarray()
else:
    X = adata.X
y = adata.obs['celltype'].values

# Split into train/test -> then train/val
train_features, test_features, train_labels, test_labels = train_test_split(
    X, y, test_size=0.2, random_state=42
)
train_features, val_features, train_labels, val_labels = train_test_split(
    train_features, train_labels, test_size=0.25, random_state=42
)

# Convert to numpy arrays (redundant if already numpy, but just to be safe)
train_features = np.array(train_features)
val_features = np.array(val_features)
test_features = np.array(test_features)

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

# Print shape info
print("\nTraining features shape:", train_features.shape)
print("Validation features shape:", val_features.shape)
print("Test features shape:", test_features.shape)

# ------------------------------------------------------------------
# 4. Handle class imbalance with sample weights
# ------------------------------------------------------------------
sample_weights = compute_sample_weight(class_weight='balanced', y=train_labels)

# ------------------------------------------------------------------
# 5. Feature scaling (StandardScaler)
# ------------------------------------------------------------------
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# ------------------------------------------------------------------
# 6. Build the final neural network
# ------------------------------------------------------------------

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

input_dim = train_features.shape[1]
num_classes = len(np.unique(train_labels))

# Model definition
model = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ------------------------------------------------------------------
# 7. Train the model
# ------------------------------------------------------------------
history = model.fit(
    train_features,
    train_labels,
    epochs=50,
    batch_size=128,
    validation_data=(val_features, val_labels),
    sample_weight=sample_weights,
    callbacks=[early_stopping]
)

# ------------------------------------------------------------------
# 8. Evaluate on test set
# ------------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(test_features, test_labels, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predictions
predictions = model.predict(test_features)
predicted_labels = np.argmax(predictions, axis=1)

# ------------------------------------------------------------------
# 9. Metrics and Confusion Matrix
# ------------------------------------------------------------------
def evaluate_model(y_true, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n------ {model_name} ------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

evaluate_model(test_labels, predicted_labels, "Final Neural Network")

print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels, zero_division=0))

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

plot_confusion_matrix(test_labels, predicted_labels, "Final Neural Network")

# ------------------------------------------------------------------
# 10. Plot training history
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# ------------------------------------------------------------------
# 11. Save the trained model
# ------------------------------------------------------------------
model.save('final_neural_network_model.h5')
print("\nModel saved as 'final_neural_network_model.h5'.")
