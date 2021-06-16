import os

import numpy as np
import pandas as pd
import scipy.stats

from src.analises.genre import compute_genre
from src.analises.popularity import compute_popularity
from src.config.labels import USER_LABEL, ITEM_LABEL, GENRES_LABEL, TRANSACTION_VALUE_LABEL, RAW_MAJORITY_GENRE, \
    RAW_MINORITY_GENRE, USER_MODEL_SIZE_LABEL, NUMBER_OF_GENRES, TOTAL_TIMES_LABEL, \
    NUMBER_OF_SHORT_TAIL_ITEMS_LABEL, NUMBER_OF_MEDIUM_TAIL_ITEMS_LABEL, PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL, \
    PERCENTAGE_OF_MEDIUM_TAIL_ITEMS_LABEL, TYPE_OF_POPULARITY
from src.config.language_strings import LANGUAGE_OMS_SELECTED, LANGUAGE_MOVIELENS_SELECTED, LANGUAGE_ANALYZING_GENRES, \
    LANGUAGE_ANALYZING_POPULARITY
from src.config.path_dir_files import data_analyze_path, DATA_ANALYSIS_FILE
from src.config.variables import OMS_DATASET, MOVIELENS_25M_DATASET, OMS_FULL_DATASET, MOVIELENS_1M_DATASET, \
    MOVIELENS_20M_DATASET
from src.preprocessing.load_database import oms_load_full_data, oms_load_raw_items, oms_load_raw_preferences, \
    oms_load_raw_genres, movielens_load_raw_items, movielens_load_raw_preferences, movielens_load_full_data, \
    oms_load_blocked_list, movielens_load_blocked_list


def create_df_with_dataset_numbers(transactions_df, item_df, index, genre_df=None):
    genres = []
    if genre_df is None:
        vec = item_df[GENRES_LABEL].tolist()
        for item_genre in vec:
            splitted = item_genre.split('|')
            genre_list = [genre for genre in splitted]
            genres = genres + genre_list
    else:
        genres = genre_df[RAW_MAJORITY_GENRE].tolist() + genre_df[RAW_MINORITY_GENRE].tolist()

    Users = transactions_df[USER_LABEL].nunique()
    Preferences = len(transactions_df)
    Itens = int(item_df[ITEM_LABEL].nunique())
    Genres = int(len(list(set(genres))))
    Bigger_User_Model = transactions_df[USER_LABEL].value_counts().max()
    Average_User_Model = transactions_df[USER_LABEL].value_counts().mean()
    Median_User_Model = transactions_df[USER_LABEL].value_counts().median()
    Std_User_Model = transactions_df[USER_LABEL].value_counts().std()
    Smaller_User_Model = transactions_df[USER_LABEL].value_counts().min()
    Max_Transaction = transactions_df[TRANSACTION_VALUE_LABEL].max()
    Average_Transaction = transactions_df[TRANSACTION_VALUE_LABEL].mean()
    Std_Transaction = transactions_df[TRANSACTION_VALUE_LABEL].std()
    Median_Transaction = transactions_df[TRANSACTION_VALUE_LABEL].median()
    Min_Transaction = transactions_df[TRANSACTION_VALUE_LABEL].min()
    return pd.DataFrame(data=[[
        Users,
        Preferences,
        Itens,
        Genres,
        Bigger_User_Model,
        Average_User_Model,
        Std_User_Model,
        Median_User_Model,
        Smaller_User_Model,
        Max_Transaction,
        Average_Transaction,
        Std_Transaction,
        Median_Transaction,
        Min_Transaction
    ]],
        columns=[
            'Users',
            'Preferences',
            'Itens',
            'Genres',
            'Bigger_User_Model',
            'Average_User_Model',
            'Std_User_Model',
            'Median_User_Model',
            'Smaller_User_Model',
            'Max_Transaction',
            'Average_Transaction',
            'Std_Transaction',
            'Median_Transaction',
            'Min_Transaction'
        ],
        index=[index]
    )


def print_dataset_numbers(df):
    print("Users: ", df['Users'].tolist())
    print("Preferences: ", df['Preferences'].tolist())
    print("Items: ", df['Itens'].tolist())
    print("Genres: ", df['Genres'].tolist())
    print("User - bigger model: ", df['Bigger_User_Model'].tolist())
    print("User - average model: ", df['Average_User_Model'].tolist())
    print("User - std model: ", df['Std_User_Model'].tolist())
    print("User - median model: ", df['Median_User_Model'].tolist())
    print("User - smaller model: ", df['Smaller_User_Model'].tolist())
    print("Big rating or listening: ", df['Max_Transaction'].tolist())
    print("Average rating or listening: ", df['Average_Transaction'].tolist())
    print("Std rating or listening: ", df['Std_Transaction'].tolist())
    print("Median rating or listening: ", df['Median_Transaction'].tolist())
    print("Small rating or listening: ", df['Min_Transaction'].tolist())


def describe_popularity(transactions_df):
    print('-' * 50)
    print(LANGUAGE_ANALYZING_POPULARITY)
    analysis_of_users_df, analysis_of_items_df = compute_popularity(transactions_df)
    item_popularity_list = analysis_of_items_df[TOTAL_TIMES_LABEL].tolist()
    n, min_max, mean, var, skew, kurt = scipy.stats.describe(item_popularity_list)
    median = np.median(item_popularity_list)
    std = scipy.std(item_popularity_list)

    print("Minimum: {0:8.3f} Maximum: {1:.6f}".format(
        analysis_of_items_df[TOTAL_TIMES_LABEL].min(), analysis_of_items_df[TOTAL_TIMES_LABEL].max()))
    print("Minimum: {0:8.3f} Maximum: {1:8.6f}".format(min_max[0], min_max[1]))
    print("Median: {0:8.3f}".format(median))
    print("Mean: {0:8.3f}".format(mean))
    print("Std. deviation : {0:8.3f}".format(std))
    print("Variance: {0:8.3f}".format(var))
    print("Skew : {0:8.3f}".format(skew))
    print("Kurtosis: {0:8.3f}".format(kurt))

    print('-' * 50)
    counted_values = analysis_of_items_df[TYPE_OF_POPULARITY].value_counts()
    short_tail_sum = 0
    medium_tail_sum = 99999
    cut_value = 0
    while short_tail_sum < medium_tail_sum:
        cut_value += 1
        short_tail_sum = (analysis_of_items_df.iloc[:cut_value])[TOTAL_TIMES_LABEL].sum()
        medium_tail_sum = (analysis_of_items_df.iloc[cut_value:])[TOTAL_TIMES_LABEL].sum()
    short_cut_value = (analysis_of_items_df.iloc[:cut_value])[TOTAL_TIMES_LABEL].sum()
    medium_cut_value = (analysis_of_items_df.iloc[cut_value:])[TOTAL_TIMES_LABEL].sum()
    print(counted_values)
    print("Medium Tail total transactions: {0:8.3f}".format(medium_cut_value))
    print("Short Tail total transactions: {0:8.3f}".format(short_cut_value))

    print('-' * 50)
    counted_values = analysis_of_users_df[TYPE_OF_POPULARITY].value_counts()
    print(counted_values)

    print('-' * 50)
    analysis_of_items_df.sort_values(by=[TOTAL_TIMES_LABEL], ascending=[False], inplace=True)
    x_data = [i + 1 for i in range(len(analysis_of_items_df))]
    y_data = analysis_of_items_df[TOTAL_TIMES_LABEL].tolist()
    corr = scipy.stats.spearmanr(x_data, y_data).correlation
    print('Correlation Between i) Item size and ii) Popularity: ', corr)

    analysis_of_users_df.sort_values(by=[PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL],
                                     ascending=[True], inplace=True)
    x_data = [i + 1 for i in range(len(analysis_of_users_df))]

    print('-' * 50)
    y_data = analysis_of_users_df[PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL].tolist()
    corr = scipy.stats.spearmanr(x_data, y_data).correlation
    print('Correlation Between i) Total os users and ii) % of popular itens: ', corr)

    print('-' * 50)
    x_data = analysis_of_users_df[USER_MODEL_SIZE_LABEL].tolist()
    y_data = analysis_of_users_df[NUMBER_OF_SHORT_TAIL_ITEMS_LABEL].tolist()
    corr = scipy.stats.spearmanr(x_data, y_data).correlation
    print('Correlation Between i) User model size and ii) Number of short tail itens: ', corr)

    y_data = analysis_of_users_df[NUMBER_OF_MEDIUM_TAIL_ITEMS_LABEL].tolist()
    corr = scipy.stats.spearmanr(x_data, y_data).correlation
    print('Correlation Between i) User model size and ii) Number of medium tail itens: ', corr)

    y_data = analysis_of_users_df[PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL].tolist()
    corr = scipy.stats.spearmanr(x_data, y_data).correlation
    print('Correlation Between i) User model size and ii) % of short tail itens: ', corr)

    y_data = analysis_of_users_df[PERCENTAGE_OF_MEDIUM_TAIL_ITEMS_LABEL].tolist()
    corr = scipy.stats.spearmanr(x_data, y_data).correlation
    print('Correlation Between i) User model size and ii) % of medium tail itens: ', corr)


def describe_genres(transactions_df):
    print('-' * 50)
    print(LANGUAGE_ANALYZING_GENRES)
    analysis_of_users_df = compute_genre(transactions_df)
    x_data = analysis_of_users_df[USER_MODEL_SIZE_LABEL].tolist()
    y_data = analysis_of_users_df[NUMBER_OF_GENRES].tolist()
    n, min_max, mean, var, skew, kurt = scipy.stats.describe(y_data)
    median = np.median(y_data)
    std = scipy.std(y_data)

    print("Minimum: {0:8.6f} Maximum: {1:8.6f}".format(min_max[0], min_max[1]))
    print("Median: {0:8.6f}".format(median))
    print("Mean: {0:8.6f}".format(mean))
    print("Std. deviation : {0:8.6f}".format(std))
    print("Variance: {0:8.6f}".format(var))
    print("Skew : {0:8.6f}".format(skew))
    print("Kurtosis: {0:8.6f}".format(kurt))
    #
    corr = scipy.stats.spearmanr(x_data, y_data).correlation
    print('Correlation Between i) User model size and ii) Number of Genres: ', corr)


def movielens_analysis(db=MOVIELENS_20M_DATASET):
    print("$" * 50)
    print(">>>>> Analyzing raw data: " + LANGUAGE_MOVIELENS_SELECTED)
    raw_items_df = movielens_load_raw_items(db=db)
    raw_preference_df = movielens_load_raw_preferences(db=db)
    df_raw = create_df_with_dataset_numbers(raw_preference_df, raw_items_df, 'raw_movielens')
    print_dataset_numbers(df_raw)

    print("$" * 50)

    print(">>>>> Analyzing clean data: " + LANGUAGE_MOVIELENS_SELECTED)
    transactions_df, items_df = movielens_load_full_data(db=db)
    df_clean = create_df_with_dataset_numbers(transactions_df, items_df, 'clean_movielens')
    print_dataset_numbers(df_clean)
    results_df = pd.concat([df_raw, df_clean])
    blocked_df = movielens_load_blocked_list(db)
    print("Blocked List lenght: ", len(blocked_df))

    describe_popularity(transactions_df)
    describe_genres(transactions_df)
    print("$" * 50)
    return results_df


def oms_analysis(db=OMS_FULL_DATASET):
    print("$" * 50)
    print(">>>>> Analyzing raw data: " + LANGUAGE_OMS_SELECTED)
    raw_items_df = oms_load_raw_items(db=db)
    raw_preference_df = oms_load_raw_preferences(db=db)
    genres_df = oms_load_raw_genres(db=db)
    df_raw = create_df_with_dataset_numbers(raw_preference_df, raw_items_df, 'raw_oms', genres_df)
    print_dataset_numbers(df_raw)

    print("$" * 50)

    print(">>>>> Analyzing clean data: " + LANGUAGE_OMS_SELECTED)
    transactions_df, items_df = oms_load_full_data(db=OMS_FULL_DATASET)
    df_clean = create_df_with_dataset_numbers(transactions_df, items_df, 'clean_oms')
    print_dataset_numbers(df_clean)

    results_df = pd.concat([df_raw, df_clean])

    blocked_df = oms_load_blocked_list(db)
    print("Blocked List lenght: ", len(blocked_df))

    describe_popularity(transactions_df)
    describe_genres(transactions_df)
    print("$" * 50)
    return results_df


def database_analysis():
    results_df = pd.DataFrame()
    results_df = pd.concat([results_df, movielens_analysis(db=MOVIELENS_20M_DATASET)])
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    results_df = pd.concat([results_df, oms_analysis(db=OMS_FULL_DATASET)])
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    results_path = data_analyze_path()
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    path_file = os.path.join(results_path, DATA_ANALYSIS_FILE)
    results_df.to_csv(path_file, index=True)
    print('Saved in: ', str(path_file))
