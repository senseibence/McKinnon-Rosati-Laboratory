# metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


def evaluate_classification_metrics(y_true, y_pred, average='weighted'):

    metrics_dict = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    return metrics_dict


def evaluate_confusion_matrix(y_true, y_pred):

    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred, target_names=None):

    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def evaluate_roc_auc(y_true, y_score):

    return roc_auc_score(y_true, y_score, multi_class='ovr')


def evaluate_average_precision(y_true, y_score):

    return average_precision_score(y_true, y_score, average='weighted')
