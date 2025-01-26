import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.metrics as metrics

name = "airway_epithelium"

test_features = np.load(f"../../Arrays/test_features_{name}.npy")
test_labels = np.load(f"../../Arrays/test_labels_{name}.npy")

binary = False
if len(np.unique(test_labels)) == 2: binary = True

model = keras.models.load_model(f"../../Models/granulomas_final_tf_nn_{name}_v1.h5") 

print(model.summary())

prediction = model.predict(test_features)
max_indices = np.argmax(prediction, axis=1)
if binary: prediction = prediction[:, 1]

def overall_metrics(y_true, y_pred, average='weighted'):

    results = {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'precision': metrics.precision_score(y_true, y_pred, average=average),
        'recall': metrics.recall_score(y_true, y_pred, average=average),
        'f1_score': metrics.f1_score(y_true, y_pred, average=average),
    }

    return results

def class_metrics(y_true, y_pred):
    return metrics.classification_report(y_true, y_pred)

def create_confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(y_true, y_pred):
    confusion_matrix = create_confusion_matrix(y_true, y_pred)

    if binary:
        print('\nconfusion matrix:')
        print('true negatives:', confusion_matrix[0][0])
        print('false positives:', confusion_matrix[0][1])
        print('false negatives:', confusion_matrix[1][0])
        print('true positives:', confusion_matrix[1][1])
        print('total class 0:', np.sum(confusion_matrix[0]))
        print('total class 1:', np.sum(confusion_matrix[1]))

    plt.figure(figsize=(8, 6))
    sb.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Reds', cbar=True, xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def roc_auc_ovr(y_true, y_score):
    return metrics.roc_auc_score(y_true, y_score, multi_class='ovr')

def roc_auc_ovo(y_true, y_score):
    return metrics.roc_auc_score(y_true, y_score, multi_class='ovo')

def average_precision(y_true, y_score): 
    return metrics.average_precision_score(y_true, y_score, average='weighted')

def balanced_accuracy(y_true, y_pred):
    return metrics.balanced_accuracy_score(y_true, y_pred)

print("\n\n\n")
print("overall metrics:", overall_metrics(test_labels, max_indices))
print("class metrics:", class_metrics(test_labels, max_indices))
print("roc_auc ovr:", roc_auc_ovr(test_labels, prediction))
print("roc_auc ovo:", roc_auc_ovo(test_labels, prediction))
print("average precision:", average_precision(test_labels, prediction))
print("balanced accuracy:", balanced_accuracy(test_labels, max_indices))
plot_confusion_matrix(test_labels, max_indices)
print("\n\n\n")