import numpy as np
import joblib
import xgboost as xgb

train_features = np.load('train_features.npy')
test_features = np.load('test_features.npy')
val_features = np.load('val_features.npy')
train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')
val_labels = np.load('val_labels.npy')
sample_weights = np.load('sample_weights.npy')

XGB_decisionTree = xgb.XGBClassifier(
    n_estimators=1500,           
    learning_rate=0.02,
    max_depth=7,
    colsample_bytree=0.7,
    subsample=0.7,
    gamma=0.2,
    objective='multi:softmax',
    eval_metric=['mlogloss', 'merror'],
    early_stopping_rounds=30,
)

XGB_decisionTree = XGB_decisionTree.fit(X=train_features, y=train_labels, eval_set=[(val_features, val_labels)], sample_weight=sample_weights)
joblib.dump(XGB_decisionTree, 'XGBmodel.pkl')