import sys
import time
from os import path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

"""Dataset Params"""
c_list = [300, 500, 1000, 1500, 1800]
d_list = [100, 1000, 5000]
e_list = ["train", "valid", "test"]

""" Hyper Parameters"""
hyper_params_dtc = {
    'criterion': ["gini", "entropy"],
    'splitter': ["best", "random"],
    'max_features': ['sqrt', 'log2', None],
}


def get_column_names(df):
    num_cols = len(list(df))
    rng = range(1, num_cols + 1)
    new_cols = ['col' + str(i) for i in rng]
    return new_cols[:num_cols]


def get_dataset_paths(data_path):
    for c in c_list:
        for d in d_list:
            dataset_paths = []
            for e in e_list:
                path_suffix = e + '_c' + str(c) + '_d' + str(d) + '.csv'
                dataset_paths.append(path.join(data_path, path_suffix))
            yield c, d, dataset_paths


def get_train_valid_dataframes(dataset_path):
    print(dataset_path)
    train_data_path = dataset_path[0]
    valid_data_path = dataset_path[1]
    test_data_path = dataset_path[2]
    train_df = pd.read_csv(train_data_path)
    valid_df = pd.read_csv(valid_data_path)
    test_df = pd.read_csv(test_data_path)
    # print("train: ", len(train_df), "valid: ", len(valid_df), "test: ", len(test_df))
    return train_df, valid_df, test_df


def get_data_with_column_names(train_df, valid_df, test_df):
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1:]
    X_valid = valid_df.iloc[:, :-1]
    y_valid = valid_df.iloc[:, -1:]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1:]

    X_train.columns = get_column_names(X_train)
    y_train.columns = get_column_names(y_train)
    X_valid.columns = get_column_names(X_valid)
    y_valid.columns = get_column_names(y_valid)
    X_test.columns = get_column_names(X_test)
    y_test.columns = get_column_names(y_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_cross_validation_and_groups(data_len):
    groups = np.array([[1] * data_len, [2] * data_len])  # 1 - train, 2 - validation
    groups = groups.reshape(data_len * 2, 1)
    group_k_fold = GroupKFold(n_splits=2)

    # data_splits = group_k_fold.split(X_train_valid, y_train_valid, groups=groups)
    # print(group_k_fold)
    # for train_index, test_index in data_splits:
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X_train_valid[train_index], X_train_valid[test_index]
    #     y_train, y_test = y_train_valid[train_index], y_train_valid[test_index]
    return group_k_fold, groups


def main():
    arg_list = sys.argv
    if len(arg_list) != 2:
        print("Pass the CNF Boolean data path as an argument")
        return

    data_path = arg_list[1]  #
    dataset_name = path.basename(path.normpath(data_path))
    print("DatasetName:", dataset_name)
    grid_search_clf = None

    dataset_paths = get_dataset_paths(data_path)
    params_table = []

    params_column_names = ["c", "d"]
    keys = sorted(hyper_params_dtc.keys())  # Best params are sorted by keys
    for k in keys:
        params_column_names.append(k)
    # params_column_names.append("Best Score")
    params_column_names.append("Accuracy")
    params_column_names.append("F1-Score")

    for c, d, dataset_path in dataset_paths:
        train_df, valid_df, test_df = get_train_valid_dataframes(dataset_path)
        X_train, y_train, X_valid, y_valid, X_test, y_test = get_data_with_column_names(train_df, valid_df, test_df)

        X_train_valid = X_train.append(X_valid, ignore_index=True, sort=False)
        y_train_valid = y_train.append(y_valid, ignore_index=True, sort=False)
        # print(X_train_valid.shape)
        # print(y_train_valid.shape)

        data_len = len(X_train)
        cv, groups = get_cross_validation_and_groups(data_len)

        grid_search_clf = GridSearchCV(DecisionTreeClassifier(random_state=2021), hyper_params_dtc, cv=cv, verbose=1,
                                       n_jobs=-1, return_train_score=True)
        grid_search_clf.fit(X_train_valid, y_train_valid, groups=groups)
        print("Best Params: ", grid_search_clf.best_params_)
        print("Best Estimator: ", grid_search_clf.best_estimator_)
        print("Best Score: ", grid_search_clf.best_score_)
        # best_clf = grid_search_clf.best_estimator_
        # y_pred = best_clf.predict(X_test)

        # Retraining the model with best Params.
        params = grid_search_clf.best_params_
        clf_with_best_params = DecisionTreeClassifier(**params)
        clf_with_best_params.fit(X_train_valid, y_train_valid)
        y_pred = clf_with_best_params.predict(X_test)

        print(pd.DataFrame(grid_search_clf.cv_results_).to_string())

        y_test = y_test.astype(float)
        y_pred = y_pred.astype(float)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("Accuracy:", accuracy, ", f1_score:", f1, sep="")
        params_table.append([c, d, *params.values(), accuracy, f1])

    df = pd.DataFrame(params_table, columns=params_column_names)
    print(df.to_string())


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- Model Run Time %s seconds ---" % (round(time.time() - start_time, 3)))
