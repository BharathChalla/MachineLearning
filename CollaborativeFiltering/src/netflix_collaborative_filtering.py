import math
import pickle
import sys
import time
from os import path

import numpy as np
import pandas as pd


def load_netflix_ratings_data(data_path):
    train_data_path = path.join(data_path, "TrainingRatings.txt")
    test_data_path = path.join(data_path, "TestingRatings.txt")

    # Netflix Data Column Names
    column_names = ['MovieId', 'UserId', 'Rating']

    print("Loading Netflix Training Ratings Data")
    train_data = pd.read_csv(train_data_path, header=None)
    train_data.columns = column_names

    print("Loading Netflix Testing Ratings Data")
    test_data = pd.read_csv(test_data_path, header=None)
    test_data.columns = column_names

    print("Loaded Netflix Ratings Data")
    return train_data, test_data


def calculate_weights_numerator_part(mean_diff_matrix):
    print("Correlation Numerator Part PreComputation")

    # (num_of_users x num_of_movies) * (num_of_movies x num_of_users)
    # Multiplication cost = num_of_movies * num_of_users**2
    mean_diff_product_ai = np.dot(mean_diff_matrix, mean_diff_matrix.T)

    return mean_diff_product_ai


def calculate_weights_denominator_part(mean_diff_matrix):
    print("Correlation Denominator Part PreComputation")
    num_of_users = mean_diff_matrix.shape[0]

    mean_diff_square = np.square(mean_diff_matrix)
    mean_diff_square_sums = np.sum(mean_diff_square, axis=1)
    mean_diff_square_sums.resize([num_of_users, 1])

    # (num_of_users x 1) * (1 x num_of_users)
    # Multiplication cost = num_of_users**2
    mean_diff_square_sums_product_ai = np.dot(mean_diff_square_sums, mean_diff_square_sums.T)

    sqrt_of_mean_diff_square_sums_product_ai = np.sqrt(mean_diff_square_sums_product_ai)
    return sqrt_of_mean_diff_square_sums_product_ai


# Numpy dot product creates high memory allocations
# and won't get garbage collected until all references are removed
def free_memory(numerator, denominator, pkl_save=False):
    if pkl_save:
        with open('numerator.pkl', 'wb') as f:
            pickle.dump(numerator, f)
        with open('denominator.pkl', 'wb') as f:
            pickle.dump(denominator, f)
    # Delete the object with reference
    del numerator
    del denominator
    # Pass By Object-reference below won't work, references are passed by value
    # Variables contains reference pointing to memory
    # numerator = None
    # denominator = None
    # print("numerator: ", numerator, " denominator: ", denominator)
    return None, None


def calculate_kappa(weights_ai, num_of_users):
    abs_weights_ai = np.abs(weights_ai)
    abs_weights_sum_ai = np.sum(abs_weights_ai, axis=1)
    abs_weights_sum_ai.resize([num_of_users, 1])

    # Calculate inverse of absolute sums for kappa
    kappa = np.ones(abs_weights_sum_ai.shape)
    kappa = np.divide(kappa, abs_weights_sum_ai, where=abs_weights_sum_ai != 0)
    # # Kappa Absolute sum for a user with all other users = 1.0
    # kappa_sum = np.multiply(kappa, abs_weights_ai)
    # kappa_sum = np.sum(kappa_sum, axis=1)
    # print("Kappa Sum:", kappa_sum)
    return kappa


def calculate_predictions_aj(v_bar, weights_ai, mean_diff_matrix):
    print("Predictions of active users 'a' and movies 'j'")

    num_of_users = mean_diff_matrix.shape[0]
    kappa = calculate_kappa(weights_ai, num_of_users)

    # Weighted Mean Diff Matrix Multiplication
    # (num_of_users x num_of_users) * (num_of_users x num_of_movies)
    # Multiplication cost = num_of_movies * num_of_users**2
    weighted_mean_diff = np.dot(weights_ai, mean_diff_matrix)

    # v_bar -> users_mean | v_bar_a -> active user mean | v_bar_i -> other user mean
    v_bar.resize([num_of_users, 1])

    kappa_weighted_ai = np.multiply(kappa, weighted_mean_diff)
    predictions_aj = np.add(v_bar, kappa_weighted_ai)

    return predictions_aj


def calculate_errors(test_data_df, predicted_ratings_df):
    # Calculating Errors
    absolute_error = 0
    squared_error = 0
    test_data_len = len(test_data_df)
    for i in range(test_data_len):
        p_aj = predicted_ratings_df.at[test_data_df['UserId'][i], test_data_df['MovieId'][i]]
        y_true = test_data_df['Rating'][i]
        error = p_aj - y_true
        absolute_error = absolute_error + abs(error)
        squared_error = squared_error + error ** 2

    mean_absolute_error = float(absolute_error) / test_data_len
    root_mean_squared_error = math.sqrt(float(squared_error) / test_data_len)

    return mean_absolute_error, root_mean_squared_error


def main():
    arg_list = sys.argv
    if len(arg_list) != 2:
        print("Pass the Netflix data path as an argument")
        return

    data_path = arg_list[1]  #
    dataset_name = path.basename(path.normpath(data_path))
    print("DatasetName:", dataset_name)

    train_data, test_data = load_netflix_ratings_data(data_path)

    train = train_data.pivot(index="UserId", columns="MovieId", values="Rating")
    users_movies_matrix = train.values
    users = np.array(train.index)
    num_of_users = train.shape[0]
    num_of_movies = train.shape[1]
    print("Users count: ", num_of_users)
    print("Movies count: ", num_of_movies)

    v_bar = np.nanmean(users_movies_matrix, axis=1, dtype=float)

    # mean difference for each user
    mean_diff_matrix = np.zeros(users_movies_matrix.shape, dtype=float)
    for i in range(num_of_users):
        x = users_movies_matrix[i, :] - v_bar[i]
        mean_diff_matrix[i, :] = np.nan_to_num(x, nan=0.0)

    # Calculating weights between active user and other users
    s_time = time.time()
    # numerator -> mean_diff_product_ai
    numerator = calculate_weights_numerator_part(mean_diff_matrix)
    print("--- Computation Time %s seconds ---" % (round(time.time() - s_time, 3)))

    s_time = time.time()  # size of matrix out=np.ones_like(numerator)
    # denominator -> sqrt_of_mean_diff_square_sums_product_ai
    denominator = calculate_weights_denominator_part(mean_diff_matrix)
    print("--- Computation Time %s seconds ---" % (round(time.time() - s_time, 3)))

    # Weights_ai - w(a, i)
    weights_ai = np.divide(numerator, denominator, where=denominator != 0)
    numerator, denominator = free_memory(numerator, denominator)

    s_time = time.time()
    predictions_aj = calculate_predictions_aj(v_bar, weights_ai, mean_diff_matrix)
    print("--- Computation Time %s seconds ---" % (round(time.time() - s_time, 3)))

    predicted_ratings = pd.DataFrame(predictions_aj, index=train.index, columns=train.columns)
    mean_absolute_error, root_mean_squared_error = calculate_errors(test_data, predicted_ratings)
    print("Mean Absolute Error:", mean_absolute_error)
    print("Root Mean Squared Error:", root_mean_squared_error)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- Model Run Time %s seconds ---" % (round(time.time() - start_time, 3)))
