import os
import random
import re
import string
import sys
import time
from os import path
from string import punctuation

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


def get_txt_files(path_to_txt_files, shuffle=False, seed=131):
    txt_files = []
    for root, dirs, files in os.walk(path_to_txt_files):
        for _file in files:
            if '.txt' in _file:
                txt_files.append(os.path.join(root, _file))
    if shuffle:
        random.seed(seed)
        random.shuffle(txt_files)
    return txt_files


def clean_data(email_data):
    email_data = re.sub(r'http\S+', ' ', email_data)
    email_data = re.sub("\d+", " ", email_data)
    email_data = email_data.replace('\n', ' ')
    email_data = email_data.translate(str.maketrans("", "", punctuation))
    email_data = email_data.lower()
    return email_data


def get_words_list(text_data_files):
    words = []
    for text_data_file in text_data_files:
        with open(text_data_file, 'r', encoding='latin-1') as file:
            data = file.read()
            data = clean_data(data)
            words.extend(data.split())

    printable_chars = list(string.printable)
    prepositions = ['and', 'the', 'or', 'are', 'in', 'to', 'be', 'is', 'as', 'by', 'if', 'will', 'as', 'for', 'on',
                    'it', 'we', 'than', 'this', 'an', 'of']
    words = [i for i in words if i not in printable_chars]
    words = [i for i in words if i not in prepositions]
    return words


def get_data_vocab_counts(vocabulary, text_data_files):
    df = pd.DataFrame(columns=vocabulary)

    count = 0
    for text_data_file in text_data_files:
        words_list = []
        with open(text_data_file, 'r', encoding='latin-1') as file:
            data = file.read()
            data = clean_data(data)
            words_list.extend(data.split())

        printable_chars = list(string.printable)
        prepositions = ['and', 'the', 'or', 'are', 'in', 'to', 'be', 'is', 'as', 'by', 'if', 'will', 'as', 'for', 'on',
                        'it', 'we', 'than', 'this', 'an', 'of']
        words_list = [i for i in words_list if i not in printable_chars]
        words_list = [i for i in words_list if i not in prepositions]

        row = [0] * len(vocabulary)

        for i in range(len(vocabulary)):
            if vocabulary[i] == 'weight-zero':
                row[i] = 1
            else:
                if vocabulary[i] in words_list:
                    row[i] = 1
                else:
                    if vocabulary[i] in words_list:
                        row[i] = 1
                    else:
                        row[i] = 0
        df.loc[count] = row
        count += 1
    return df.values


def get_metrics(labels, preds):
    true_negative = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    for i in range(len(labels)):
        if labels[i] == preds[i]:
            if labels[i] == 1:
                true_negative = true_negative + 1
            else:
                true_positive = true_positive + 1
        else:
            if labels[i] == 1:
                false_negative = false_negative + 1
            else:
                false_positive = false_positive + 1

    accuracy = float(true_negative + true_positive) / len(labels) * 100
    precision = float(true_positive) / (true_positive + false_positive) * 100
    recall = float(true_positive) / (true_positive + false_negative) * 100
    f1_score = 2 * float((precision * recall) / (precision + recall))
    return accuracy, precision, recall, f1_score


def main():
    arg_list = sys.argv
    if len(arg_list) != 2:
        print("Pass the Ham Spam data path as an argument")
        return

    data_path = arg_list[1]  #
    dataset_name = path.basename(path.normpath(data_path))
    print(dataset_name)
    train_data_path = path.join(data_path, "train")
    test_data_path = path.join(data_path, "test")

    path_train_ham = path.join(train_data_path, "ham")
    path_train_spam = path.join(train_data_path, "spam")
    path_test_ham = path.join(test_data_path, "ham")
    path_test_spam = path.join(test_data_path, "spam")

    files_ham = get_txt_files(path_train_ham, shuffle=True)
    files_spam = get_txt_files(path_train_spam, shuffle=True)

    final_files_train = files_ham + files_spam

    words = get_words_list(final_files_train)

    vocabulary_final_train = ['weight-zero']
    for x in words:
        if x not in vocabulary_final_train:
            vocabulary_final_train.append(x)

    train_data = get_data_vocab_counts(vocabulary_final_train, final_files_train)

    labels_train = []
    for f in final_files_train:
        if f in files_ham:
            labels_train.append(1)
        else:
            labels_train.append(0)

    files_test = []
    files_test.extend(get_txt_files(path_test_ham))
    ham_labels_len = len(files_test)
    test_labels = [1] * ham_labels_len
    files_test.extend(get_txt_files(path_test_spam))
    spam_labels_len = len(files_test) - ham_labels_len
    test_labels.extend([0] * spam_labels_len)

    test_data = get_data_vocab_counts(vocabulary_final_train, files_test)

    parameter_grid = {'alpha': [0.01, 0.1, 0.3]}
    model = SGDClassifier(random_state=0, loss='log', penalty='l2', class_weight='balanced', max_iter=1000)
    model_grid = GridSearchCV(estimator=model, param_grid=parameter_grid, n_jobs=-1, scoring='roc_auc')
    model_grid.fit(train_data, labels_train)
    predictions = model_grid.predict(test_data)

    accuracy, precision, recall, f1_score = get_metrics(test_labels, predictions)
    print("Metrics for Bernoulli Model - SGDC -", dataset_name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- Model Run Time %s seconds ---" % (round(time.time() - start_time, 3)))
