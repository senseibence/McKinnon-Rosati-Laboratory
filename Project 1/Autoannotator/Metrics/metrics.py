import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.metrics as metrics

test_features = np.load('../../Arrays/test_features.npy')
test_labels = np.load('../../Arrays/test_labels.npy')

model = keras.models.load_model("../../Models/granulomas_final_tf_nn_v3.h5") 

print(model.summary())

prediction = model.predict(test_features)
max_indices = np.argmax(prediction, axis=1)

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

    plt.figure(figsize=(8, 6))
    sb.heatmap(confusion_matrix, annot=False, cmap='Reds', cbar=True, xticklabels=np.unique(np.arange(30)), yticklabels=np.unique(np.arange(30)))
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

print("overall metrics:", overall_metrics(test_labels, max_indices))
print("class metrics:", class_metrics(test_labels, max_indices))
print("roc_auc ovr:", roc_auc_ovr(test_labels, prediction))
print("roc_auc ovo:", roc_auc_ovo(test_labels, prediction))
print("average precision:", average_precision(test_labels, prediction))
print("balanced accuracy:",balanced_accuracy(test_labels, max_indices))
plot_confusion_matrix(test_labels, max_indices)