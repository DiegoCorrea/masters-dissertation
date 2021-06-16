import multiprocessing
from collections import Counter
from copy import deepcopy
from multiprocessing import Pool, Process

import numpy as np
import pandas as pd
# from dask.distributed import Client

from src.config.labels import USER_LABEL, GENRES_LABEL
from src.config.variables import N_CORES
from src.conversions.pandas_to_models import user_transactions_df_to_item_mapping

GLOBAL_TRANSACTIONS_DF = pd.DataFrame()
GLOBAL_ITEM_MAPPING = dict()


def compute_genre_distr_with_weigth(items):
    """Compute the genre distribution for a given list of Items."""
    genre_distr_dict = dict()
    weigth_sum_dict = dict()
    result = dict()
    for item in items:
        item_class = items[item]
        item_weigth = item_class.score
        for genre, score in item_class.genres.items():
            genre_sum = genre_distr_dict.get(genre, 0.)
            genre_distr_dict[genre] = genre_sum + score * item_weigth
            weigth_rating = weigth_sum_dict.get(genre, 0.)
            weigth_sum_dict[genre] = weigth_rating + item_weigth

    for genre, genre_score in genre_distr_dict.items():
        w = genre_score / weigth_sum_dict[genre]
        normed_genre_score = round(w, 3)
        result[genre] = normed_genre_score
    return result


def map_get_user_items(user_id):
    global GLOBAL_TRANSACTIONS_DF, GLOBAL_ITEM_MAPPING
    df = GLOBAL_TRANSACTIONS_DF[GLOBAL_TRANSACTIONS_DF[USER_LABEL] == user_id]
    items = user_transactions_df_to_item_mapping(df, GLOBAL_ITEM_MAPPING)
    interacted_distr = compute_genre_distr_with_weigth(items)
    return pd.DataFrame(interacted_distr, index=[user_id])


def gd_loop(id_list):
    return pd.concat([map_get_user_items(user_id) for user_id in id_list], sort=False)


def multiprocess_get_distribution(transactions_df, item_mapping):
    global GLOBAL_TRANSACTIONS_DF, GLOBAL_ITEM_MAPPING
    GLOBAL_ITEM_MAPPING = item_mapping
    GLOBAL_TRANSACTIONS_DF = transactions_df
    ids = transactions_df[USER_LABEL].unique().tolist()
    grous_split_id = np.array_split(ids, N_CORES)
    pool = Pool(N_CORES)
    map_results_df = pool.map(gd_loop, grous_split_id)
    pool.close()
    pool.join()
    result_df = pd.concat(map_results_df, sort=False)
    result_df.fillna(0.0, inplace=True)
    return result_df


def get_distribution(transactions_df, item_mapping):
    global GLOBAL_TRANSACTIONS_DF, GLOBAL_ITEM_MAPPING
    GLOBAL_ITEM_MAPPING = item_mapping
    GLOBAL_TRANSACTIONS_DF = transactions_df
    ids = transactions_df[USER_LABEL].unique().tolist()
    map_results_df = [map_get_user_items(user_id) for user_id in ids]
    result_df = pd.concat(map_results_df, sort=False)
    result_df.fillna(0.0, inplace=True)
    return result_df


def user_get_distribution(user_transations_df, item_mapping):
    items = user_transactions_df_to_item_mapping(user_transations_df, item_mapping)
    interacted_distr = compute_genre_distr_with_weigth(items)
    return interacted_distr


# #######################################################################################################
#
# #######################################################################################################
def split_genres(user_transactions_df):
    transactions_genres_list = user_transactions_df[GENRES_LABEL].tolist()
    genres_list = []
    for item_genre in transactions_genres_list:
        splitted = item_genre.split('|')
        splitted_genre_list = [genre for genre in splitted]
        genres_list = genres_list + splitted_genre_list
    count_dict = Counter(genres_list)
    values_list = list(count_dict.values())
    sum_values_list = sum(values_list)
    values_list = [v / sum_values_list for v in values_list]
    df = pd.DataFrame([values_list], columns=list(count_dict.keys()))
    return df


def genre_probability_distribution(transactions_df, label=USER_LABEL):
    id_list = transactions_df[label].unique().tolist()
    pool = Pool(N_CORES)
    list_df = pool.map(split_genres, [transactions_df[transactions_df[label] == uid] for uid in id_list])
    pool.close()
    pool.join()
    result_df = pd.concat(list_df, sort=False)
    result_df.fillna(0.0, inplace=True)
    return result_df


# #######################################################################################################
#
# #######################################################################################################


def get_user_items_multiprocessing(user_id, user_model_df, item_mapping):
    items = user_transactions_df_to_item_mapping(user_model_df, item_mapping)
    interacted_distr = compute_genre_distr_with_weigth(items)
    return pd.DataFrame(interacted_distr, index=[user_id])


def gd_loop_multiprocessing(users_model_df, item_mapping, shared_queue):
    id_list = users_model_df[USER_LABEL].unique().tolist()
    results = pd.concat(
        [get_user_items_multiprocessing(user_id, users_model_df[users_model_df[USER_LABEL] == user_id], item_mapping)
         for user_id in id_list], sort=False)
    shared_queue.put(deepcopy(results))


def big_genre_distribution_with_multiprocessing(df, item_mapping):
    # Preparing: users, results dataframe and shared queue over processes
    users_ids = df[USER_LABEL].unique().tolist()
    grous_split_id = np.array_split(users_ids, N_CORES)

    result_df = pd.DataFrame()

    manager = multiprocessing.Manager()
    shared_queue = manager.Queue()
    # While has users to process do in nbeans (ncores)
    cores_control = list(range(0, N_CORES))
    all_processes = list()

    # Print the state of the execution
    while grous_split_id and cores_control:
        # Allocate core and select the user to pos process
        cores_control.pop(0)
        user_id = grous_split_id.pop(0)
        # Prepare user data
        users_model_df = deepcopy(df[df[USER_LABEL].isin(user_id)])
        # Create the process
        p = Process(target=gd_loop_multiprocessing,
                    args=(users_model_df, item_mapping, shared_queue))
        all_processes.append(p)
    # Start all process
    for p in all_processes:
        p.start()
    # Wait and close the all processes
    results = list()
    for p in all_processes:
        p.join()
        # Get results from all processes
        results.append(shared_queue.get())
        p.close()

    # Concat and resume the results
    result_df = pd.concat([x for x in results], sort=False)
    result_df.fillna(0.0, inplace=True)
    return result_df


# class ClassDistribution:
#     def __init__(self):
#         self.distribution = None
#
#     def distributed_computation(self, df, item_mapping):
#         # Preparing users
#         client = Client()
#         print(client.cluster)
#         users_ids_list = df[USER_LABEL].unique().tolist()
#         # delayed_list = [delayed(get_user_items_multiprocessing)(user_id, df[df[USER_LABEL] == user_id], item_mapping)
#         #                 for user_id in users_ids_list]
#         # delayed_list.compute()
#         # future = client.map(get_user_items_multiprocessing, [(user_id, df[df[USER_LABEL] == user_id], item_mapping)
#         #                                                      for user_id in users_ids_list])
#         # print(future)
#         # results = client.gather(future)
#         # print(results)
#         client.shutdown()
