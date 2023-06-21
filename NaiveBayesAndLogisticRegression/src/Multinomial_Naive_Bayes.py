import math
import os
import random
import re
import string
import sys
import time
from os import path
from string import punctuation

import numpy as np
import pandas as pd


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


def clean_words(words_list):
    x = string.printable
    waste = list(x)
    prepositions = ['and', 'the', 'or', 'are', 'in', 'to', 'be', 'is', 'as', 'by', 'if', 'will', 'as', 'for', 'on',
                    'it', 'we', 'than', 'this', 'an']
    words_list = [i for i in words_list if i not in waste]
    words_list = [i for i in words_list if i.isalnum() is True]
    words_list = [i for i in words_list if i not in prepositions]
    words_list = [i for i in words_list if len(i) != 2]

    words_arr = np.array(words_list)

    return words_arr


def clean_data(email_data):
    email_data = re.sub(r'http\S+', ' ', email_data)
    email_data = re.sub("\d+", " ", email_data)
    email_data = email_data.replace('\n', ' ')
    email_data = email_data.translate(str.maketrans("", "", punctuation))
    email_data = email_data.lower()
    return email_data


def process_train_data(train_ham_files, train_spam_files):
    words_list_total = []
    for train_ham_file in train_ham_files:
        with open(train_ham_file, 'r', encoding='latin-1') as file:
            data = file.read()
            data = data.replace('\n', ' ')
            words_list_total.extend(data.split())

    for train_spam_file in train_spam_files:
        with open(train_spam_file, 'r', encoding='latin-1') as file:
            data = file.read()
            data = data.replace('\n', ' ')
            words_list_total.extend(data.split())

    words_arr = clean_words(words_list_total)

    return words_arr


def append_labelled_data(label, train_files, words_columns, count_words_unique_total, df):
    itr = len(df.index)
    for train_ham_file in train_files:
        words_list_file = []
        with open(train_ham_file, 'r', encoding='latin-1') as file:
            data = file.read()
            data = clean_data(data)
            words_list_file.extend(data.split())

        words_arr_mail = clean_words(words_list_file)
        count_words_unique_file = [0] * len(count_words_unique_total)

        words_unique_file, count_words_unique_file_temp = np.unique(words_arr_mail, return_counts=True)
        common_words, common_indices_total, common_indices_file = np.intersect1d(words_columns, words_unique_file,
                                                                                 return_indices=True)
        for i in range(len(common_indices_total)):
            count_words_unique_file[common_indices_total[i]] = count_words_unique_file_temp[common_indices_file[i]]

        p = list(count_words_unique_file)
        p.append(label)
        df.loc[itr] = p
        itr += 1


def bag_of_words(train_ham_files, train_spam_files, words_array_train):
    words_columns, count_words_unique_total = np.unique(words_array_train, return_counts=True)
    p_columns = list(words_columns)
    p_columns.append('labels_Ans')
    df = pd.DataFrame(columns=p_columns)
    append_labelled_data(0, train_ham_files, words_columns, count_words_unique_total, df)
    append_labelled_data(1, train_spam_files, words_columns, count_words_unique_total, df)
    return df


def multinomial_naive_bayes(test_ham_files, test_spam_files, df):
    df_spam = df[df['labels_Ans'] == 1]
    df_ham = df[df['labels_Ans'] == 0]

    ham_mail_count = len(df_ham.index)
    spam_mail_count = len(df_spam.index)
    p_ham = math.log(ham_mail_count, 2) - math.log(ham_mail_count + spam_mail_count, 2)
    p_spam = math.log(spam_mail_count, 2) - math.log(ham_mail_count + spam_mail_count, 2)

    spam_words_count_arr = df_spam.sum()
    count_words_spam = spam_words_count_arr.sum() - df_spam['labels_Ans'].sum()

    ham_words_count_arr = df_ham.sum() - df_ham['labels_Ans'].sum()
    count_words_ham = ham_words_count_arr.sum()

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    test_data_files = test_ham_files + test_spam_files
    for test_data_file in test_data_files:
        words_list = []
        with open(test_data_file, 'r', encoding='latin-1') as file:
            data = file.read()
            data = clean_data(data)
            words_list.extend(data.split())

        words_arr_mail = clean_words(words_list)

        PHam_Mail = p_ham
        PSpam_Mail = p_spam

        for i in range(len(words_arr_mail)):
            if words_arr_mail[i] in df.columns:
                PoWgH = math.log(1 + df_ham[words_arr_mail[i]].sum(), 2) \
                        - math.log(len(df.columns) - 1 + count_words_ham, 2)
                PoWgS = math.log(1 + df_spam[words_arr_mail[i]].sum(), 2) - \
                        math.log(len(df.columns) - 1 + count_words_spam, 2)
            else:
                PoWgH = -math.log(len(df.columns) - 1 + count_words_ham, 2)
                PoWgS = -math.log(len(df.columns) - 1 + count_words_spam, 2)

            PHam_Mail += PoWgH
            PSpam_Mail += PoWgS

        if PHam_Mail >= PSpam_Mail:
            if test_data_file in test_ham_files:
                true_negative += 1
            else:
                false_negative += 1
        else:
            if test_data_file in test_spam_files:
                true_positive += 1
            else:
                false_positive += 1
    accuracy = float(true_positive + true_negative) / len(test_data_files)
    precision = float(true_positive) / (true_positive + false_positive)
    recall = float(true_positive) / (true_positive + false_negative)
    f1_score = 2 * float(precision * recall) / (precision + recall)

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

    files_train_ham = get_txt_files(path_train_ham)
    files_train_spam = get_txt_files(path_train_spam)

    words_array_train = process_train_data(files_train_ham, files_train_spam)
    data_BoW = bag_of_words(files_train_ham, files_train_spam, words_array_train)

    files_test_ham = get_txt_files(path_test_ham)
    files_test_spam = get_txt_files(path_test_spam)

    accuracy, precision, recall, f1_score = multinomial_naive_bayes(files_test_ham, files_test_spam, data_BoW)
    print("Metrics for Multinomial Naive Bayes - Bag of Words -", dataset_name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- Model Run Time %s seconds ---" % (round(time.time() - start_time, 3)))
