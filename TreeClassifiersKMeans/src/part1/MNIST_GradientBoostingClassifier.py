import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Reference
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
# Params
# loss: {‘deviance’, ‘exponential’}, default=’deviance’
# learning_rate: float, default=0.1
# n_estimators: int, default=100
# subsample: float, default=1.0
# criterion: {‘friedman_mse’, ‘squared_error’, ‘mse’, ‘mae’}, default=’friedman_mse’
# min_samples_split: int or float, default=2
# min_samples_leaf: int or float, default=1
# min_weight_fraction_leaf: float, default=0.0
# max_depth: int, default=3

""" MNIST Data """
print("Loading MNIST data")
# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

# rescale the data, use the traditional train/test split
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X.astype(np.float32))

full_data = True
if full_data:
    # (60K: Train) and (10K: Test)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
else:
    lenX = len(X) / 10
    print("Using 10% of MNIST data")
    X_train, X_test = X[:6000], X[6000:7000]
    y_train, y_test = y[:6000], y[6000:7000]
print("Loaded MNIST data")

""" Hyper Parameters"""
hyper_params_gbc = {
    'learning_rates': [1.0, 0.1, 0.01],
    'n_estimators': [50, 100, 150],
    'max_features': ['sqrt', 'log2'],
    # 'max_features': ['sqrt', 'log2', None],
}


def get_params(hyper_params):
    learning_rates_list = hyper_params['learning_rates']
    n_estimators_list = hyper_params['n_estimators']
    max_features_list = hyper_params['max_features']
    for learning_rate in learning_rates_list:
        for n_estimators in n_estimators_list:
            for max_features in max_features_list:
                yield learning_rate, n_estimators, max_features


params = get_params(hyper_params_gbc)
values_params_table = []
for i, j, k in params:
    start_time = time.time()
    clf = GradientBoostingClassifier(learning_rate=i, n_estimators=j, max_features=k, criterion="friedman_mse")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_test = y_test.astype(float)
    y_pred = y_pred.astype(float)
    accuracy = accuracy_score(y_test, y_pred)
    run_time = round(time.time() - start_time, 3)
    values_params_table.append((i, j, k, accuracy, run_time))
    print("Parameters: ", "learning_rate: ", i, ", n_estimators: ", j, ", max_features:", k,
          ", accuracy:", accuracy, ", run_time:", run_time, sep="")

cols = ["learning_rate", "n_estimators", "max_features", "accuracy", "run_time"]
df = pd.DataFrame(values_params_table, columns=cols)
print(df.to_string())
