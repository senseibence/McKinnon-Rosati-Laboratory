import numpy as np
import keras as ks
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.metrics as metrics
import scanpy as sc

# name = "subset"

# test_features = np.load(f"../../Arrays/test_features_hvg_{name}.npy")
# test_labels = np.load(f"../../Arrays/test_labels_hvg_{name}.npy")

# binary = False
# if len(np.unique(test_labels)) == 2: binary = True

# model = ks.models.load_model(f"../../Models/granulomas30_hvg_{name}_jax_v1.keras", custom_objects={'LeakyReLU': ks.layers.LeakyReLU}, compile=False)

# print(model.summary())

# prediction = model.predict(test_features)
# max_indices = np.argmax(prediction, axis=1)
# if binary: prediction = prediction[:, 1]

adata_global_test = sc.read_h5ad("C:\\Users\\bence\\Projects\\BIO446\\McKinnon-Rosati-Laboratory\\Project 1\\Data\\adata_global_test.h5ad")
adata_global_test_hvg = adata_global_test[:, adata_global_test.var['highly_variable'] ].copy()

# test_features = adata_global_test_hvg.X
test_labels = adata_global_test_hvg.obs['celltype'].values

# prediction = model.predict(test_features)
# max_indices = np.argmax(prediction, axis=1)

max_indices = np.load("../../Arrays/max_indices.npy")

def overall_metrics(y_true, y_pred, average='weighted'):

    results = {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'precision': metrics.precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': metrics.recall_score(y_true, y_pred, average=average),
        'f1_score': metrics.f1_score(y_true, y_pred, average=average),
    }

    return results

def class_metrics(y_true, y_pred):
    return metrics.classification_report(y_true, y_pred, zero_division=0)

def create_confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(y_true, y_pred):
    confusion_matrix = create_confusion_matrix(y_true, y_pred)

    # if binary:
    #     print('\nconfusion matrix:')
    #     print('true negatives:', confusion_matrix[0][0])
    #     print('false positives:', confusion_matrix[0][1])
    #     print('false negatives:', confusion_matrix[1][0])
    #     print('true positives:', confusion_matrix[1][1])
    #     print('total class 0:', np.sum(confusion_matrix[0]))
    #     print('total class 1:', np.sum(confusion_matrix[1]))

    plt.figure(figsize=(12, 10))
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
print(overall_metrics(test_labels, max_indices))
print()
print(class_metrics(test_labels, max_indices))
# print("roc_auc ovr:", roc_auc_ovr(test_labels, prediction))
# print("roc_auc ovo:", roc_auc_ovo(test_labels, prediction))
# print("average precision:", average_precision(test_labels, prediction))
print("balanced accuracy:", balanced_accuracy(test_labels, max_indices))
plot_confusion_matrix(test_labels, max_indices)
print("\n\n\n")