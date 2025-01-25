import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

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

# Convert to PyTorch tensors
X_train = torch.tensor(train_features, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)
X_val = torch.tensor(val_features, dtype=torch.float32)
y_val = torch.tensor(val_labels, dtype=torch.long)
X_test = torch.tensor(test_features, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)

# Define the neural network with Sigmoid activations
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.sigmoid1 = nn.Sigmoid()
        self.layer2 = nn.Linear(128, 64)
        self.sigmoid2 = nn.Sigmoid()
        self.output = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.sigmoid1(self.layer1(x))  # Apply sigmoid activation after the first layer
        x = self.sigmoid2(self.layer2(x))  # Apply sigmoid activation after the second layer
        x = self.output(x)                 # Output layer (no activation applied here)
        return x


# Initialize the model
input_size = train_features.shape[1]
num_classes = len(set(train_labels))  # Number of unique labels
model = NeuralNet(input_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_predictions = torch.argmax(val_outputs, axis=1)
        val_accuracy = (val_predictions == y_val).float().mean()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), 'Pytorch_NeuralNetModel.pth')

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
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)  # Logits from the neural network
    nn_pred = torch.argmax(test_outputs, axis=1).numpy()  # Convert logits to class labels

# Call the updated confusion matrix function
plot_confusion_matrix(y_test.numpy(), nn_pred, "Neural Network")

# Metrics of the Neural Network
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = torch.argmax(test_outputs, axis=1)
    print(classification_report(y_test.numpy(), test_predictions.numpy()))