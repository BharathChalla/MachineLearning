import time

import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.datasets import fetch_openml
from sklearnex.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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
    print("Using 10% of MNIST data")
    X_train, X_test = X[:6000], X[6000:7000]
    y_train, y_test = y[:6000], y[6000:7000]
    print("Loaded 10% of MNIST data")
print("Loaded MNIST data")

""" Hyper Parameters"""
hyper_params_rfc = {
    'n_estimators': [50, 100, 150],
    'criterion': ["gini", "entropy"],
    'max_features': ['sqrt', 'log2', None],
}


def get_params(hyper_params):
    n_estimators_list = hyper_params['n_estimators']
    criterion_list = hyper_params['criterion']
    max_features_list = hyper_params['max_features']
    for n_estimator in n_estimators_list:
        for criterion in criterion_list:
            for max_features in max_features_list:
                yield n_estimator, criterion, max_features


params = get_params(hyper_params_rfc)
values_params_table = []
for i, j, k in params:
    start_time = time.time()
    clf = RandomForestClassifier(n_estimators=i, criterion=j, max_features=k, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_test = y_test.astype(float)
    y_pred = y_pred.astype(float)
    accuracy = accuracy_score(y_test, y_pred)
    run_time = round(time.time() - start_time, 3)
    values_params_table.append((i, j, k, accuracy, run_time))
    print("Parameters: ", "n_estimators:", i, ", criterion:", j, ", max_features:", k,
          ", accuracy:", accuracy, ", run_time:", run_time, sep="")

cols = ["n_estimators", "criterion", "max_features", "accuracy", "run_time"]
df = pd.DataFrame(values_params_table, columns=cols)
print(df.to_string())
