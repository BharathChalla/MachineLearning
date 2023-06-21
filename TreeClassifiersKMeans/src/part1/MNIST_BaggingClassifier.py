import time

import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.datasets import fetch_openml
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
print("Loaded MNIST data")

""" Hyper Parameters"""
hyper_params_bc = {
    'base_estimators': [DecisionTreeClassifier()],  # RandomForestClassifier(), SVC()],
    'n_estimators': [5, 10, 15, 20, 25],
    'bootstrap': [True, False],
}


def get_params(hyper_params):
    base_estimators_list = hyper_params['base_estimators']
    n_estimators_list = hyper_params['n_estimators']
    bootstrap_list = hyper_params['bootstrap']
    for base_estimator in base_estimators_list:
        for n_estimator in n_estimators_list:
            for bootstrap in bootstrap_list:
                yield base_estimator, n_estimator, bootstrap


params = get_params(hyper_params_bc)
values_params_table = []
for i, j, k in params:
    start_time = time.time()
    ensemble_clf = BaggingClassifier(base_estimator=i, n_estimators=j, bootstrap=k, n_jobs=-1)
    ensemble_clf.fit(X_train, y_train)
    y_pred = ensemble_clf.predict(X_test)
    y_test = y_test.astype(float)
    y_pred = y_pred.astype(float)
    accuracy = accuracy_score(y_test, y_pred)
    run_time = round(time.time() - start_time, 3)
    values_params_table.append((i, j, k, accuracy, run_time))
    print("Parameters: ", "base_estimators:", i, ", n_estimators:", j, ", bootstrap:", k,
          ", accuracy:", accuracy, ", run_time:", run_time, sep="")

cols = ["base_estimators", "n_estimators", "bootstrap", "accuracy", "run_time"]
df = pd.DataFrame(values_params_table, columns=cols)
print(df.to_string())
