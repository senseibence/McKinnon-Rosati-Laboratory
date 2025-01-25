import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

train_features = np.load('../arrays/train_features.npy')
test_features = np.load('../arrays/test_features.npy')
val_features = np.load('../arrays/val_features.npy')
train_labels = np.load('../arrays/train_labels.npy')
test_labels = np.load('../arrays/test_labels.npy')
val_labels = np.load('../arrays/val_labels.npy')
sample_weights = np.load('../arrays/sample_weights.npy')

# Normalize features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# Convert labels to TensorFlow tensors
y_train = tf.convert_to_tensor(train_labels, dtype=tf.int32)
y_val = tf.convert_to_tensor(val_labels, dtype=tf.int32)
y_test = tf.convert_to_tensor(test_labels, dtype=tf.int32)

# Define the neural network with Sigmoid activations
input_size = train_features.shape[1]
num_classes = len(np.unique(train_labels))

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(input_size,)),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(num_classes)
])

# Loss and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Training loop
num_epochs = 50
batch_size = 32

# Fit the model
model.fit(train_features, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(val_features, val_labels))

# Evaluate the model on validation set
val_loss, val_accuracy = model.evaluate(val_features, val_labels, verbose=0)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save the model
model.save('Tensorflow_NeuralNetModel.h5')

# Define the function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7.7, 6))
    sb.heatmap(cm, annot=False, cmap='Blues', cbar=True,
               xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Confusion matrix plot
y_pred = model.predict(test_features)
nn_pred = np.argmax(y_pred, axis=1)  # Convert logits to class labels

# Call the updated confusion matrix function
plot_confusion_matrix(y_test.numpy(), nn_pred, "Neural Network")

# Metrics of the Neural Network
print(classification_report(y_test.numpy(), nn_pred))

###########################################################################

#Use this framework to create a big model

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

# Normalize features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# Convert labels to TensorFlow tensors
y_train = tf.convert_to_tensor(train_labels, dtype=tf.int32)
y_val = tf.convert_to_tensor(val_labels, dtype=tf.int32)
y_test = tf.convert_to_tensor(test_labels, dtype=tf.int32)

# Define a giant neural network with more layers and different activation functions
input_size = train_features.shape[1]
num_classes = len(np.unique(train_labels))

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(input_size,)),
    tf.keras.layers.Dense(1024, activation='tanh'),
    tf.keras.layers.Dense(512, activation='sigmoid'),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(num_classes)
])

# Loss and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Training loop
num_epochs = 50
batch_size = 32

# Fit the model
model.fit(train_features, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(val_features, val_labels))

# Evaluate the model on validation set
val_loss, val_accuracy = model.evaluate(val_features, val_labels, verbose=0)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Define the function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7.7, 6))
    sb.heatmap(cm, annot=False, cmap='Blues', cbar=True,
               xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Confusion matrix plot
y_pred = model.predict(test_features)
nn_pred = np.argmax(y_pred, axis=1)  # Convert logits to class labels

# Call the updated confusion matrix function
plot_confusion_matrix(y_test.numpy(), nn_pred, "Neural Network")

# Metrics of the Neural Network
print(classification_report(y_test.numpy(), nn_pred))

# Save the model
model.save('Tensorflow_GiantNeuralNetModel.h5')
