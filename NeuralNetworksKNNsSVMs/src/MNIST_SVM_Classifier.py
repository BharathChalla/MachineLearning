import time

import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn import clone
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearnex.svm import SVC

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
C_range_linear = np.logspace(0, 2, 3)
hyper_params_C_linear = {
    'C': C_range_linear
}
hyper_params_C_gamma_kernel = {
    'C': [1, 5, 10],
    'gamma': [1e-4, 1e-3, 1e-2],
    'kernel': ['rbf', 'sigmoid'],  # 'poly', 'degree'
}


def get_params(hyper_params):
    c_list = hyper_params['C']
    gamma_list = hyper_params['gamma']
    kernel_list = hyper_params['kernel']
    for k in kernel_list:
        for C in c_list:
            for g in gamma_list:
                yield k, C, g


def find_best_params_linear(estimator, hyper_params):
    estimator_name = type(estimator).__name__
    print("Estimator: ", estimator_name)
    params = hyper_params['C']
    best_params = {}
    best_error_rate = 1.0
    params_error_rates = []
    for C in params:
        start_time = time.time()
        print("Training Model: ", estimator_name, end='')
        print(" - Parameters: ", "C:", C)
        model = clone(estimator)
        model.set_params(C=C)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        error_rate = 1 - accuracy
        print("TEST ERROR RATE(%):", error_rate)
        model_run_time = (round(time.time() - start_time, 3))
        params_error_rates.append(["linear", C, "Default", error_rate, model_run_time])
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_params = {'C': C}
        print("--- Model Run Time %s seconds ---" % model_run_time)
        print()

    column_names = ["Kernel", "C", "gamma", "ErrorRate", "RunTime(sec)"]
    df = pd.DataFrame(params_error_rates, columns=column_names)
    print(df.to_string())
    print("Best Error Rate: ", best_error_rate)
    print("Best Hyper-Parameters: ", best_params)


def find_best_params(estimator, hyper_params):
    estimator_name = type(estimator).__name__
    print("Estimator: ", estimator_name)
    params = get_params(hyper_params)
    best_params = {}
    best_error_rate = 1.0
    params_error_rates = []
    for k, C, g, in params:
        start_time = time.time()
        print("Training Model: ", estimator_name, end='')
        print(" - Parameters: ", "kernel:", k, "C:", C, "gamma:", g, )
        model = clone(estimator)
        model.set_params(kernel=k, C=C, gamma=g)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        error_rate = 1 - accuracy
        print("TEST ERROR RATE(%):", error_rate)
        model_run_time = (round(time.time() - start_time, 3))
        params_error_rates.append([k, C, g, error_rate, model_run_time])
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_params = {'C': C, 'gamma': g, 'kernel': k}
        print("--- Model Run Time %s seconds ---" % model_run_time)
        print()

    column_names = ["Kernel", "C", "gamma", "ErrorRate", "RunTime(sec)"]
    df = pd.DataFrame(params_error_rates, columns=column_names)
    print(df.to_string())
    print("Best Error Rate: ", best_error_rate)
    print("Best Hyper-Parameters: ", best_params)


linear_svc_clf = SVC(kernel='linear', random_state=2021)
find_best_params_linear(linear_svc_clf, hyper_params_C_linear)

svc_clf = SVC(random_state=2021)
find_best_params(svc_clf, hyper_params_C_gamma_kernel)
