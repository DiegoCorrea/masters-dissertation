import datetime
import gc
import logging
import multiprocessing
import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd

from src.analises.popularity import count_popularity_item
from src.config.labels import TRANSACTION_VALUE_LABEL, POPULARITY_LABEL, USER_LABEL, ITEM_LABEL, TOTAL_TIMES_LABEL, \
    TRADITIONAL_RECOMMENDERS
from src.config.language_strings import LANGUAGE_RECOMMENDATION, LANGUAGE_CALCULATING_USER_MODEL_WGENRE_DISTRIBUTION, \
    LANGUAGE_LOAD_DATA_SET, LANGUAGE_RECOMMENDER_ALGORITHM_START, LANGUAGE_RECOMMENDER_ALGORITHM_STOP
from src.config.path_dir_files import data_results_path
from src.config.variables import K_FOLDS_VALUES, N_CORES, CANDIDATES_LIST_SIZE
from src.conversions.pandas_to_models import user_transactions_df_to_item_mapping
from src.models.item import create_item_mapping
from src.posprocessing.bias import calculating_item_bias
from src.posprocessing.distributions import get_distribution, multiprocess_get_distribution, \
    big_genre_distribution_with_multiprocessing
from src.posprocessing.step import pos_processing_calibration
from src.preprocessing.load_database import load_db_and_fold, load_blocked_list
from src.processing.merge_results import merge_recommender_results, k_fold_results_concat
from src.processing.recommender_algorithms import split_equally

logger = logging.getLogger(__name__)


def ncores_traditional_recommendation_process(user_model_df, user_model_genres_distr_df, user_expected_items_df,
                                              items_mapping_dict, user_blocked_items_df, recommender_label,
                                              popularity_df, transaction_mean, control_count=None, start_time=None):
    """
    A user by time
    Responsible for: the recommender algorithm prediction,
                    the models to be used in the pos process
                    and the pos processing
    :param start_time:
    :param control_count:
    :param user_blocked_items_df:
    :param transaction_mean: the users transactions mean
    :param popularity_df: DataFrame with items popularity
    :param user_model_df: All user transactions
    :param user_model_genres_distr_df: The user genres distribution
    :param user_expected_items_df: The user expected items in the final recommendation
    :param items_mapping_dict: A dict with all items in the system
    :param recommender_label: The recommender algorithm label
    :return: None
    """
    # Get known items ids by the user
    items_ids = items_mapping_dict.keys()
    know_items_ids = user_model_df[ITEM_LABEL].unique().tolist()
    blocked_items_ids = user_blocked_items_df[ITEM_LABEL].unique().tolist()
    items_ids = set(items_ids) - set(blocked_items_ids)
    # Get unknown items ids by the user
    unknowing_items_ids = list(set(items_ids) - set(know_items_ids))

    user_candidate_items_max_df = popularity_df[popularity_df[ITEM_LABEL].isin(unknowing_items_ids)]

    user_candidate_items_max_df.sort_values(by=[TRANSACTION_VALUE_LABEL], ascending=False)
    user_candidate_items_max_df = user_candidate_items_max_df[:CANDIDATES_LIST_SIZE]

    user_candidate_items_max_dict = user_transactions_df_to_item_mapping(user_candidate_items_max_df,
                                                                         items_mapping_dict)
    user_evaluation_results_df = pos_processing_calibration(user_model_genres_distr_df=user_model_genres_distr_df,
                                                            candidates_items_mapping=user_candidate_items_max_dict,
                                                            user_expected_items_ids=user_expected_items_df[
                                                                ITEM_LABEL].tolist(),
                                                            recommender_label=recommender_label,
                                                            transaction_mean=transaction_mean)
    if control_count is not None and control_count % 100 == 0:
        logger.info(' '.join(['PId:', str(os.getpid()), '->', 'Total of users done:', str(control_count),
                              '->', 'Total time:', str(datetime.timedelta(seconds=time.time() - start_time))]))
    return user_evaluation_results_df


def ncores_generate_recommendation(users_ids_list, users_model_df, users_model_genres_distr_df, users_expected_items_df,
                                   items_mapping_dict, users_blocked_items_df, recommender_label, popularity_df,
                                   transaction_mean, shared_queue):
    """
    A user by time
    Responsible for: the recommender algorithm prediction,
                    the models to be used in the pos process
                    and the post processing
    :param popularity_df:
    :param users_ids_list:
    :param users_blocked_items_df:
    :param transaction_mean: the users transactions mean
    :param users_ids_list: The group of unique users identification
    :param users_model_df: All user transactions
    :param users_model_genres_distr_df: The user genres distribution
    :param users_expected_items_df: The user expected items in the final recommendation
    :param items_mapping_dict: A dict with all items in the system
    :param recommender_label: The recommender algorithm label
    :param shared_queue: A shared memory to be used for all ncores process
    :return: The user results in the shared memory
    """
    start_time = time.time()
    results = pd.concat(
        [ncores_traditional_recommendation_process(users_model_df[users_model_df[USER_LABEL] == user_id],
                                                   users_model_genres_distr_df.loc[user_id],
                                                   users_expected_items_df[
                                                       users_expected_items_df[USER_LABEL] == user_id],
                                                   items_mapping_dict,
                                                   users_blocked_items_df[
                                                       users_blocked_items_df[USER_LABEL] == user_id],
                                                   recommender_label,
                                                   popularity_df,
                                                   transaction_mean,
                                                   control_count, start_time) for
         control_count, user_id in enumerate(users_ids_list)], sort=False)
    finish_time = time.time()
    logger.info(" ".join(['>', 'Time Execution:', str(datetime.timedelta(seconds=finish_time - start_time)),
                          'Total of users:', str(len(users_ids_list))]))
    shared_queue.put(deepcopy(results))


def ncores_traditional_recommendation(users_genres_distr_df, trainset_df, testset_df, items_mapping_dict,
                                      blocked_items_df, popularity_df, transaction_mean, recommender_label):
    """
    Multiprocessing recommendations to each user do the pos process
    :param blocked_items_df:
    :param popularity_df: DataFrame with items popularity
    :param transaction_mean: the users transactions mean
    :param users_genres_distr_df: A dataframe with the users genres distributions
    :param trainset_df: A dataframe with the train set transactions
    :param testset_df: A dataframe with the test set transactions
    :param items_mapping_dict: A dict with all items in the system
    :param recommender_label: The recommender algorithm label
    :return: A dataframe with the results of all used metrics
    """
    # Preparing: users, results dataframe and shared queue over processes
    users_ids_list = users_genres_distr_df.index.values.tolist()
    # chunk_users_ids = np.array_split(users_ids_list, N_CORES)
    chunk_users_ids = split_equally(trainset_df)

    # Print the state of the execution
    i = len(users_ids_list)
    logger.info(str(i) + ' users to finish')

    manager = multiprocessing.Manager()
    shared_queue = manager.Queue()
    all_processes = list()

    # As long as there are users on the list to process and cores to allocate, do
    while chunk_users_ids:
        # Allocate core and select the user to pos process
        users_ids_list = chunk_users_ids.pop(0)

        # Prepare user data
        users_model_df = deepcopy(trainset_df[trainset_df[USER_LABEL].isin(users_ids_list)])
        users_model_genres_distr_df = deepcopy(users_genres_distr_df.loc[list(users_ids_list)])
        users_expected_items_df = deepcopy(testset_df[testset_df[USER_LABEL].isin(users_ids_list)])
        users_blocked_items_df = deepcopy(blocked_items_df[blocked_items_df[USER_LABEL].isin(users_ids_list)])

        # Create the process
        p = multiprocessing.Process(target=ncores_generate_recommendation,
                                    args=(users_ids_list,
                                          users_model_df, users_model_genres_distr_df, users_expected_items_df,
                                          items_mapping_dict, users_blocked_items_df, recommender_label, popularity_df,
                                          transaction_mean, shared_queue,))
        all_processes.append(p)
    # Start all process
    for p in all_processes:
        p.start()
    # Wait and close the all processes
    user_evaluation_results = list()
    for p in all_processes:
        p.join()
        # Get results from all processes
        user_evaluation_results.append(shared_queue.get())
        p.close()

    # Concat and resume the results
    evaluation_results_df = pd.concat(user_evaluation_results)
    return evaluation_results_df


# #################################################################################################################### #
# #################################################################################################################### #
# #################################################################################################################### #
def traditional_generate_recommendation(user_model_df, user_model_genres_distr_df, user_expected_items_df,
                                        items_mapping_dict, user_blocked_items_df, recommender_label, popularity_df,
                                        transaction_mean,
                                        shared_queue):
    """
    A user by time
    Responsible for: the recommender algorithm prediction,
                    the models to be used in the pos process
                    and the pos processing
    :param user_blocked_items_df:
    :param transaction_mean: the users transactions mean
    :param popularity_df: DataFrame with items popularity
    :param user_model_df: All user transactions
    :param user_model_genres_distr_df: The user genres distribution
    :param user_expected_items_df: The user expected items in the final recommendation
    :param items_mapping_dict: A dict with all items in the system
    :param recommender_label: The recommender algorithm label
    :param shared_queue: A shared memory to be used for all ncores process
    :return: None
    """
    # Get known items ids by the user
    items_ids = items_mapping_dict.keys()
    know_items_ids = user_model_df[ITEM_LABEL].unique().tolist()
    blocked_items_ids = user_blocked_items_df[ITEM_LABEL].unique().tolist()
    items_ids = set(items_ids) - set(blocked_items_ids)
    # Get unknown items ids by the user
    unknowing_items_ids = list(set(items_ids) - set(know_items_ids))

    user_candidate_items_max_df = popularity_df[popularity_df[ITEM_LABEL].isin(unknowing_items_ids)]

    user_candidate_items_max_df.sort_values(by=[TRANSACTION_VALUE_LABEL], ascending=False)
    user_candidate_items_max_df = user_candidate_items_max_df[:CANDIDATES_LIST_SIZE]

    user_candidate_items_max_dict = user_transactions_df_to_item_mapping(user_candidate_items_max_df,
                                                                         items_mapping_dict)
    user_evaluation_results_df = pos_processing_calibration(user_model_genres_distr_df=user_model_genres_distr_df,
                                                            candidates_items_mapping=user_candidate_items_max_dict,
                                                            user_expected_items_ids=user_expected_items_df[
                                                                ITEM_LABEL].tolist(),
                                                            recommender_label=recommender_label,
                                                            transaction_mean=transaction_mean)
    shared_queue.put(deepcopy(user_evaluation_results_df))


def traditional_recommendation_process(users_genres_distr_df, trainset_df, testset_df, items_mapping_dict,
                                       blocked_items_df,
                                       popularity_df, transaction_mean, recommender_label):
    """
    Multiprocessing recommendations to each user do the pos process
    :param popularity_df: DataFrame with items popularity
    :param transaction_mean: the users transactions mean
    :param users_genres_distr_df: A dataframe with the users genres distributions
    :param trainset_df: A dataframe with the train set transactions
    :param testset_df: A dataframe with the test set transactions
    :param items_mapping_dict: A dict with all items in the system
    :param recommender_label: The recommender algorithm label
    :return: A dataframe with the results of all used metrics
    """
    # Preparing: users, results dataframe and shared queue over processes
    users_ids = users_genres_distr_df.index.values.tolist()
    evaluation_results_df = pd.DataFrame()
    manager = multiprocessing.Manager()
    shared_queue = manager.Queue()
    # While has users to process do in nbeans (ncores)
    while users_ids:
        cores_control = list(range(0, N_CORES))
        all_processes = list()

        # Print the state of the execution
        i = len(users_ids)
        print(f'\r{i} users to finish', end='', flush=True)
        # As long as there are users on the list to process and cores to allocate, do
        while users_ids and cores_control:
            # Allocate core and select the user to pos process
            cores_control.pop(0)
            user_id = users_ids.pop(0)
            # Prepare user data
            user_model_df = deepcopy(trainset_df[trainset_df[USER_LABEL] == user_id])
            user_model_genres_distr_df = deepcopy(users_genres_distr_df.loc[user_id])
            user_expected_items_df = deepcopy(testset_df[testset_df[USER_LABEL] == user_id])
            user_blocked_items_df = deepcopy(blocked_items_df[blocked_items_df[USER_LABEL] == user_id])
            # Create the process
            p = multiprocessing.Process(target=traditional_generate_recommendation,
                                        args=(
                                            user_model_df, user_model_genres_distr_df, user_expected_items_df,
                                            items_mapping_dict, user_blocked_items_df, recommender_label, popularity_df,
                                            transaction_mean, shared_queue,))
            all_processes.append(p)
        # Start all process
        for p in all_processes:
            p.start()
        # Wait and close the all processes
        user_evaluation_results = list()
        for p in all_processes:
            p.join()
            # Get results from all processes
            user_evaluation_results.append(shared_queue.get())
            p.close()

        # Concat and resume the results
        evaluation_results_df = pd.concat([evaluation_results_df, pd.concat(user_evaluation_results)])
    return evaluation_results_df


def start_traditional_recommenders(recommenders_to_use, db):
    """
    Start the traditional recommenders (process and post process)
    :param recommenders_to_use:
    :param db: 0 to movielens; 1 to oms;
    :return: None
    """

    for label in recommenders_to_use:
        logger.info('-' * 50)
        logger.info(" ".join([LANGUAGE_RECOMMENDER_ALGORITHM_START, "->", label]))

        # For each dataset fold do process and pos process
        for fold in range(1, K_FOLDS_VALUES + 1):
            gc.collect()

            # Load the fold of the dataset
            logger.info(" ".join(["+", LANGUAGE_LOAD_DATA_SET, "->", str(fold)]))
            trainset_df, testset_df, items_df = load_db_and_fold(db, fold)

            # calculating the user bias
            transaction_mean = trainset_df[TRANSACTION_VALUE_LABEL].mean()
            items_mapping_dict = create_item_mapping(items_df, calculating_item_bias(trainset_df, transaction_mean))

            # get the users genres distribution based on they models
            logger.info(" ".join(["+", LANGUAGE_CALCULATING_USER_MODEL_WGENRE_DISTRIBUTION]))
            users_prefs_distr_df = get_distribution(trainset_df, items_mapping_dict)

            # Start the recommendation process with the pos processing
            logger.info(" ".join(["+", LANGUAGE_RECOMMENDATION]))
            if label == POPULARITY_LABEL:
                popularity_df = count_popularity_item(trainset_df)
            else:
                popularity_df = pd.DataFrame(
                    data=trainset_df.groupby(by=ITEM_LABEL, as_index=False).mean())
            popularity_df.rename(inplace=True, columns={TOTAL_TIMES_LABEL: TRANSACTION_VALUE_LABEL})

            results_df = traditional_recommendation_process(users_prefs_distr_df, trainset_df, testset_df,
                                                            items_mapping_dict, popularity_df,
                                                            transaction_mean, label)
            merged_results_df = k_fold_results_concat(results_df)
            # Save the results
            path_to_save = "".join([data_results_path(db), label, "/"])
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            merged_results_df.to_csv(os.path.join(path_to_save, str(fold) + ".csv"),
                                     index=False)
        # Merge the recommender results into a average based in the configuration
        merge_recommender_results(label, db)
        logger.info(" ".join([LANGUAGE_RECOMMENDER_ALGORITHM_STOP, "->", label]))
        logger.info('-' * 50)


def all_traditional_recommenders(db):
    """
    Start all traditioinal recommenders
    :param db: 0 to movielens; 1 to oms;
    :return: None
    """
    # Start recommender algorithms that have params
    start_traditional_recommenders(TRADITIONAL_RECOMMENDERS, db)


# #################################################################################################################### #
# #################################################################################################################### #
# #################################################################################################################### #


def personalized_traditional_recommender(label, db, fold):
    # Load the fold of the dataset
    logger.info(" ".join(["+", LANGUAGE_LOAD_DATA_SET, "->", str(fold)]))
    trainset_df, testset_df, items_df = load_db_and_fold(db, fold)
    blocked_items_df = load_blocked_list(db)

    # calculating the user bias
    transaction_mean = trainset_df[TRANSACTION_VALUE_LABEL].mean()
    items_mapping_dict = create_item_mapping(items_df, calculating_item_bias(trainset_df, transaction_mean))

    # get the users genres distribution based on they models
    logger.info(" ".join(["+", LANGUAGE_CALCULATING_USER_MODEL_WGENRE_DISTRIBUTION]))
    # users_prefs_distr_df = get_distribution(trainset_df, items_mapping_dict)
    # users_prefs_distr_df = multiprocess_get_distribution(trainset_df, items_mapping_dict)
    users_prefs_distr_df = big_genre_distribution_with_multiprocessing(trainset_df, items_mapping_dict)

    # Start the recommendation process with the pos processing
    logger.info(" ".join(["+", LANGUAGE_RECOMMENDATION]))
    if label == POPULARITY_LABEL:
        popularity_df = count_popularity_item(trainset_df)
    else:
        popularity_df = pd.DataFrame(
            data=trainset_df.groupby(by=ITEM_LABEL, as_index=False).mean())
    popularity_df.rename(inplace=True, columns={TOTAL_TIMES_LABEL: TRANSACTION_VALUE_LABEL})

    results_df = ncores_traditional_recommendation(users_prefs_distr_df, trainset_df, testset_df,
                                                   items_mapping_dict, blocked_items_df, popularity_df,
                                                   transaction_mean, label)
    merged_results_df = k_fold_results_concat(results_df)
    # Save the results
    path_to_save = "".join([data_results_path(db), label, "/"])
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    merged_results_df.to_csv(os.path.join(path_to_save, str(fold) + ".csv"),
                             index=False)
    # Merge the recommender results into a average based in the configuration
    # merge_recommender_results(label, db)
    logger.info(" ".join([LANGUAGE_RECOMMENDER_ALGORITHM_STOP, "->", label]))
    logger.info('-' * 50)
