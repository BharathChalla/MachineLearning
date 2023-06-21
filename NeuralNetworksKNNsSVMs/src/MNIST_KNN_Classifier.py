import time

import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn import clone
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearnex.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

""" MNIST Data """
print("Loading MNIST data")
# Load data from https://www.openml.org/d/554
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, cache=True)
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
    X_train, X_test = X[:6000], X[6000:7000]
    y_train, y_test = y[:6000], y[6000:7000]

print("Loaded MNIST data")

""" Hyper Parameters"""
hyper_params_knn = {
    'n_neighbors': [3, 5, 9],
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance'],
}


def get_params(hyper_params):
    n_neighbors_list = hyper_params['n_neighbors']
    metrics_list = hyper_params['metric']
    weights_list = hyper_params['weights']
    for nn in n_neighbors_list:
        for m in metrics_list:
            for w in weights_list:
                yield nn, m, w


def find_best_params(estimator, hyper_params):
    estimator_name = type(estimator).__name__
    print("Estimator: ", estimator_name)
    params = get_params(hyper_params)
    best_params = {}
    best_error_rate = 1.0
    params_error_rates = []
    for nn, m, w in params:
        start_time = time.time()
        print("Training Model: ", estimator_name, end='')
        print(" - Parameters: ", "n_neighbors:", nn, "metric:", m, "weights:", w, )
        model = clone(estimator)
        model.set_params(n_neighbors=nn, metric=m, weights=w)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        error_rate = 1 - accuracy
        print("TEST ERROR RATE(%):", error_rate)
        model_run_time = (round(time.time() - start_time, 3))
        params_error_rates.append([nn, m, w, error_rate, model_run_time])
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_params = {'n_neighbors': nn, 'metric': m, 'weights': w, }
        print("--- Model Run Time %s seconds ---" % model_run_time)
        print()

    column_names = ["n_neighbors", "metric", "weights", "ErrorRate", "RunTime(sec)"]
    df = pd.DataFrame(params_error_rates, columns=column_names)
    print(df.to_string())
    print("Best Error Rate: ", best_error_rate)
    print("Best Hyper-Parameters: ", best_params)


mlp_clf = KNeighborsClassifier()
find_best_params(mlp_clf, hyper_params_knn)
