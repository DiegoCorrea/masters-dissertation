import gc
import json
import logging
import multiprocessing
import os
import time
from copy import deepcopy
from multiprocessing import Process
import datetime
import pandas as pd
import numpy as np
from surprise import SVD, NMF, KNNBasic
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.matrix_factorization import SVDpp
from surprise.prediction_algorithms.slope_one import SlopeOne

from src.config.labels import NMF_LABEL, SVD_LABEL, SVDpp_LABEL, ITEM_KNN_LABEL, USER_KNN_LABEL, \
    CO_CLUSTERING_LABEL, USER_LABEL, ITEM_LABEL, TRANSACTION_VALUE_LABEL, LEARNING_RECOMMENDERS, TEST_RECOMMENDERS, \
    SLOPE_LABEL
from src.config.language_strings import LANGUAGE_RECOMMENDER_ALGORITHM_START, LANGUAGE_RECOMMENDER_ALGORITHM_STOP, \
    LANGUAGE_LOAD_DATA_SET, LANGUAGE_TRANSFORMING_DATA, LANGUAGE_CALCULATING_USER_MODEL_WGENRE_DISTRIBUTION, \
    LANGUAGE_TRAINING_THE_RECOMMENDER, LANGUAGE_RECOMMENDATION
from src.config.path_dir_files import data_results_path, grid_search_path
from src.config.variables import CANDIDATES_LIST_SIZE, N_CORES, K_FOLDS_VALUES
from src.conversions.pandas_to_models import transform_testset, user_transactions_df_to_item_mapping, transform_trainset
from src.conversions.suprise_and_pandas import surprise_to_pandas_get_candidates_items
from src.models.item import create_item_mapping
from src.posprocessing.bias import calculating_item_bias
from src.posprocessing.distributions import get_distribution, big_genre_distribution_with_multiprocessing
from src.posprocessing.step import pos_processing_calibration
from src.preprocessing.load_database import load_db_and_fold, load_blocked_list
from src.processing.merge_results import merge_recommender_results, k_fold_results_concat

logger = logging.getLogger(__name__)


def get_candidate_items_model(user_id, user_model_df, items_mapping_dict, user_blocked_items_df):
    """
    Get all candidate items to be recommended to the user
    :param user_blocked_items_df:
    :param user_id: The user unique identification
    :param user_model_df: All user transactions
    :param items_mapping_dict: A dict with all items in the system
    :return: A dataframe with all unknown items by the user
    """
    # Get known items ids by the user
    items_ids = items_mapping_dict.keys()
    know_items_ids = user_model_df[ITEM_LABEL].unique().tolist()
    blocked_items_ids = user_blocked_items_df[ITEM_LABEL].unique().tolist()

    # Get unknown items ids by the user
    # items_ids = set(items_ids) - set(blocked_items_ids)
    data = {ITEM_LABEL: list(set(items_ids) - set(know_items_ids))}

    # Create dataframe with unknown items by the user
    user_candidate_items_model_df = pd.DataFrame(data)
    user_candidate_items_model_df[USER_LABEL] = user_id
    user_candidate_items_model_df[TRANSACTION_VALUE_LABEL] = 0.0

    return user_candidate_items_model_df


def recommender_prediction(reco_instance_worker, user_candidate_items_model_df):
    """
    The recommender algorithm prediction step
    :param reco_instance_worker: A trained instance of a recommender algorithm class
    :param user_candidate_items_model_df: A dataframe with all unknown items by the user
    :return: A dataframe with the candidates items maximized, the top items
    """
    test_items = transform_testset(user_candidate_items_model_df)
    candidate_items_max = reco_instance_worker.test(test_items)
    return surprise_to_pandas_get_candidates_items(candidate_items_max, n=CANDIDATES_LIST_SIZE)


def recommendation_proccess(user_id, user_model_df, user_model_genres_distr_df, user_expected_items_df,
                            items_mapping_dict, user_blocked_items_df, reco_instance_worker, reco_label,
                            transaction_mean, control_count=None, start_time=None):
    """
    A user by time
    Responsible for: the recommender algorithm prediction,
                    the models to be used in the pos process
                    and the post processing
    :param start_time:
    :param control_count:
    :param user_blocked_items_df:
    :param transaction_mean: the users transactions mean
    :param user_id: The user unique identification
    :param user_model_df: All user transactions
    :param user_model_genres_distr_df: The user genres distribution
    :param user_expected_items_df: The user expected items in the final recommendation
    :param items_mapping_dict: A dict with all items in the system
    :param reco_instance_worker: A trained instance of a recommender algorithm class
    :param reco_label: The recommender algorithm label
    :return: The user results
    """
    user_candidate_items_model_df = get_candidate_items_model(user_id, user_model_df, items_mapping_dict,
                                                              user_blocked_items_df)

    user_candidate_items_max_df = recommender_prediction(reco_instance_worker, user_candidate_items_model_df)

    user_candidate_items_max_dict = user_transactions_df_to_item_mapping(user_candidate_items_max_df,
                                                                         items_mapping_dict)
    user_evaluation_results_df = pos_processing_calibration(user_model_genres_distr_df=user_model_genres_distr_df,
                                                            candidates_items_mapping=user_candidate_items_max_dict,
                                                            user_expected_items_ids=user_expected_items_df[
                                                                ITEM_LABEL].tolist(),
                                                            recommender_label=reco_label,
                                                            transaction_mean=transaction_mean)
    if control_count is not None and control_count % 100 == 0:
        logger.info(' '.join(['PId:', str(os.getpid()), '->', 'Total of users done:', str(control_count),
                              '->', 'Total time:', str(datetime.timedelta(seconds=time.time() - start_time))]))
    return user_evaluation_results_df


def generate_recommendation(user_id, user_model_df, user_model_genres_distr_df, user_expected_items_df,
                            items_mapping_dict, user_blocked_items_df, reco_instance_worker, reco_label,
                            transaction_mean,
                            shared_queue):
    """
    A user by time
    Responsible for: the recommender algorithm prediction,
                    the models to be used in the pos process
                    and the post processing
    :param user_blocked_items_df:
    :param transaction_mean: the users transactions mean
    :param user_id: The user unique identification
    :param user_model_df: All user transactions
    :param user_model_genres_distr_df: The user genres distribution
    :param user_expected_items_df: The user expected items in the final recommendation
    :param items_mapping_dict: A dict with all items in the system
    :param reco_instance_worker: A trained instance of a recommender algorithm class
    :param reco_label: The recommender algorithm label
    :param shared_queue: A shared memory to be used for all ncores process
    :return: The user results in the shared memory
    """
    user_evaluation_results_df = recommendation_proccess(user_id, user_model_df, user_model_genres_distr_df,
                                                         user_expected_items_df,
                                                         items_mapping_dict, user_blocked_items_df,
                                                         reco_instance_worker, reco_label,
                                                         transaction_mean)
    shared_queue.put(deepcopy(user_evaluation_results_df))


# #################################################################################################################### #
# #################################################################################################################### #
# #################################################################################################################### #
def split_equally(trainset_df):
    values_counts = trainset_df[USER_LABEL].value_counts()
    values_dict = values_counts.to_dict()
    ids = list(values_dict.keys())
    groups_split_id = []
    for i in range(N_CORES):
        groups_split_id.append([])
    while ids:
        for i in range(N_CORES):
            if ids:
                groups_split_id[i].append(ids.pop(0))
            else:
                break
        for i in range(N_CORES-1, -1, -1):
            if ids:
                groups_split_id[i].append(ids.pop(0))
            else:
                break
    return groups_split_id


def ncores_generate_recommendation(group_user_id, users_model_df, users_model_genres_distr_df, user_expected_items_df,
                                   items_mapping_dict, users_blocked_items_df, reco_instance_worker, reco_label,
                                   transaction_mean,
                                   shared_queue):
    """
    A user by time
    Responsible for: the recommender algorithm prediction,
                    the models to be used in the pos process
                    and the post processing
    :param users_blocked_items_df:
    :param transaction_mean: the users transactions mean
    :param group_user_id: The group of unique users identification
    :param users_model_df: All user transactions
    :param users_model_genres_distr_df: The user genres distribution
    :param user_expected_items_df: The user expected items in the final recommendation
    :param items_mapping_dict: A dict with all items in the system
    :param reco_instance_worker: A trained instance of a recommender algorithm class
    :param reco_label: The recommender algorithm label
    :param shared_queue: A shared memory to be used for all ncores process
    :return: The user results in the shared memory
    """
    start_time = time.time()
    results = pd.concat([recommendation_proccess(user_id,
                                                 users_model_df[users_model_df[USER_LABEL] == user_id],
                                                 users_model_genres_distr_df.loc[user_id],
                                                 user_expected_items_df[user_expected_items_df[USER_LABEL] == user_id],
                                                 items_mapping_dict,
                                                 users_blocked_items_df[users_blocked_items_df[USER_LABEL] == user_id],
                                                 reco_instance_worker,
                                                 reco_label,
                                                 transaction_mean, count, start_time) for count, user_id in enumerate(group_user_id)], sort=False)
    finish_time = time.time()
    logger.info(" ".join(['>', 'Time Execution:', str(datetime.timedelta(seconds=finish_time - start_time)),
                          'Total of users:', str(len(group_user_id))]))
    shared_queue.put(deepcopy(results))


def n_cores_recommendations(reco_instance_worker, users_genres_distr_df, trainset_df, testset_df,
                            items_mapping_dict, blocked_items_df, recommender_label, transaction_mean):
    """
    Multiprocessing recommendations to each user do the pos process
    :param blocked_items_df:
    :param transaction_mean: the users transactions mean
    :param reco_instance_worker: A trained instance of a recommender algorithm class
    :param users_genres_distr_df: A dataframe with the users genres distributions
    :param trainset_df: A dataframe with the train set transactions
    :param testset_df: A dataframe with the test set transactions
    :param items_mapping_dict: A dict with all items in the system
    :param recommender_label: The recommender algorithm label
    :return: A dataframe with the results of all used metrics
    """
    # Preparing: users, results dataframe and shared queue over processes
    ids = users_genres_distr_df.index.values.tolist()
    # groups_split_id = np.array_split(ids, N_CORES)
    groups_split_id = split_equally(trainset_df)

    # Print the state of the execution
    i = len(ids)
    logger.info(str(i) + ' users to finish')

    manager = multiprocessing.Manager()
    shared_queue = manager.Queue()
    all_processes = list()

    # As long as there are users on the list to process and cores to allocate, do
    while groups_split_id:
        # Allocate core and select the user to pos process
        group_user_id = np.asarray(groups_split_id.pop(0))
        # #
        users_model_df = deepcopy(trainset_df[trainset_df[USER_LABEL].isin(group_user_id)])
        users_model_genres_distr_df = deepcopy(users_genres_distr_df.loc[list(group_user_id)])
        users_expected_items_df = deepcopy(testset_df[testset_df[USER_LABEL].isin(group_user_id)])
        users_blocked_items_df = deepcopy(blocked_items_df[blocked_items_df[USER_LABEL].isin(group_user_id)])

        # Create the process
        p = Process(target=ncores_generate_recommendation,
                    args=(group_user_id, users_model_df, users_model_genres_distr_df, users_expected_items_df,
                          items_mapping_dict, users_blocked_items_df, reco_instance_worker, recommender_label,
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
    return pd.concat([x for x in user_evaluation_results])


# #################################################################################################################### #
# #################################################################################################################### #
# #################################################################################################################### #
def multiprocessing_recommendations(reco_instance_worker, users_genres_distr_df, trainset_df, testset_df,
                                    items_mapping_dict, blocked_items_df, recommender_label, transaction_mean):
    """
    Multiprocessing recommendations to each user do the pos process
    :param blocked_items_df:
    :param transaction_mean: the users transactions mean
    :param reco_instance_worker: A trained instance of a recommender algorithm class
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
        logger.info(str(i) + ' users to finish')
        # print(f'\r{i} users to finish', end='', flush=True)
        # print(str(i) + " Faltam")
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
            p = Process(target=generate_recommendation,
                        args=(user_id, user_model_df, user_model_genres_distr_df, user_expected_items_df,
                              items_mapping_dict, user_blocked_items_df, reco_instance_worker, recommender_label,
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


# #################################################################################################################### #
# #################################################################################################################### #
# #################################################################################################################### #
def learning_recommender_instance(label, db):
    """
    Start learning recommenders (process and post process)
    :param label: The recommender to be instanced by the surprise framework
    :param db: 0 to movielens; 1 to oms;
    :return: None
    """
    if label == SLOPE_LABEL:
        return SlopeOne()
    else:
        path_to_save = grid_search_path(db)
        with open("".join([path_to_save, label, ".json"])) as json_file:
            params = json.load(json_file)
            if label == SVD_LABEL:
                return SVD(n_factors=params['n_factors'], n_epochs=params['n_epochs'],
                           lr_all=params['lr_all'], reg_all=params['reg_all'], biased=True,
                           random_state=42, verbose=True)
            elif label == NMF_LABEL:
                return NMF(n_factors=params['n_factors'], n_epochs=params['n_epochs'], reg_bi=params['reg_bi'],
                           reg_pu=params['reg_pu'], reg_qi=params['reg_qi'], reg_bu=params['reg_bu'],
                           lr_bu=params['lr_bu'], lr_bi=params['lr_bi'], biased=params['biased'],
                           random_state=42, verbose=True)
            elif label == CO_CLUSTERING_LABEL:
                return CoClustering(n_epochs=params['n_epochs'], n_cltr_u=params['n_cltr_u'],
                                    n_cltr_i=params['n_cltr_i'],
                                    verbose=True)
            elif label == ITEM_KNN_LABEL:
                return KNNBasic(k=params['k'], sim_options=params['sim_options'], verbose=True)
            elif label == USER_KNN_LABEL:
                return KNNBasic(k=params['k'], sim_options=params['sim_options'], verbose=True)
            elif label == SVDpp_LABEL:
                return SVDpp(n_factors=params['n_factors'], n_epochs=params['n_epochs'],
                             lr_all=params['lr_all'], reg_all=params['reg_all'],
                             random_state=42, verbose=True)


def start_learning_recommenders(recommenders_labels, db):
    """
    Start learning recommenders (process and post process)
    :param recommenders_labels: A list with all learning recommenders for processing
    :param db: 0 to movielens; 1 to oms;
    :return: None
    """
    # For each recommender algorithm do
    for label in recommenders_labels:
        logger.info('-' * 50)
        logger.info(" ".join([LANGUAGE_RECOMMENDER_ALGORITHM_START, "->", label]))

        # For each dataset fold do process and pos process
        for fold in range(1, K_FOLDS_VALUES + 1):
            gc.collect()
            # Copy the recommender instance to work with
            reco_instance_worker = learning_recommender_instance(label, db)

            # Load the fold of the dataset
            logger.info(" ".join(["+", LANGUAGE_LOAD_DATA_SET, "->", str(fold)]))
            trainset_df, testset_df, items_df = load_db_and_fold(db, fold)
            blocked_items_df = load_blocked_list(db)

            # Transform data in a manipulable structure
            logger.info(" ".join(["+", LANGUAGE_TRANSFORMING_DATA]))
            trainset = transform_trainset(trainset_df)
            # calculating the user bias
            transaction_mean = trainset_df[TRANSACTION_VALUE_LABEL].mean()
            items_mapping_dict = create_item_mapping(items_df, calculating_item_bias(trainset_df, transaction_mean))
            del items_df

            # get the users genres distribution based on they models
            logger.info(" ".join(["+", LANGUAGE_CALCULATING_USER_MODEL_WGENRE_DISTRIBUTION]))
            users_prefs_distr_df = get_distribution(trainset_df, items_mapping_dict)

            # Train the recommender algorithm
            logger.info(" ".join(["+", LANGUAGE_TRAINING_THE_RECOMMENDER]))
            reco_instance_worker.fit(trainset)

            # Start the recommendation process with the pos processing
            logger.info(" ".join(["+", LANGUAGE_RECOMMENDATION]))
            # results_df = multiprocessing_recommendations(reco_instance_worker, users_prefs_distr_df,
            #                                              trainset_df, testset_df, items_mapping_dict, blocked_items_df,
            #                                              label, transaction_mean)
            results_df = n_cores_recommendations(reco_instance_worker, users_prefs_distr_df,
                                                 trainset_df, testset_df, items_mapping_dict, blocked_items_df,
                                                 label, transaction_mean)
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


def all_learning_recommenders(db):
    """
    Start all recommenders based in machine learning
    :param db: 0 to movielens; 1 to oms;
    :return: None
    """
    # Start recommender algorithms that have params
    start_learning_recommenders(LEARNING_RECOMMENDERS, db)


def test_learning_recommender(db):
    """
    Start a limited recommenders based in machine learning to test the processing
    :param db: 0 to movielens; 1 to oms;
    :return: None
    """
    start_learning_recommenders(TEST_RECOMMENDERS, db)


# ################################################################################################ #
# ################################################################################################ #
# ################################################################################################ #


def personalized_learning_recommender(label, db, fold):
    # Copy the recommender instance to work with
    reco_instance_worker = learning_recommender_instance(label, db)

    # Load the fold of the dataset
    logger.info(" ".join(["+", LANGUAGE_LOAD_DATA_SET, "->", str(fold)]))
    trainset_df, testset_df, items_df = load_db_and_fold(db, fold)
    blocked_items_df = load_blocked_list(db)

    # Transform data in a manipulable structure
    logger.info(" ".join(["+", LANGUAGE_TRANSFORMING_DATA]))
    trainset = transform_trainset(trainset_df)
    # calculating the user bias
    transaction_mean = trainset_df[TRANSACTION_VALUE_LABEL].mean()
    items_mapping_dict = create_item_mapping(items_df, calculating_item_bias(trainset_df, transaction_mean))
    del items_df

    # get the users genres distribution based on they models
    logger.info(" ".join(["+", LANGUAGE_CALCULATING_USER_MODEL_WGENRE_DISTRIBUTION]))
    # users_prefs_distr_df = get_distribution(trainset_df, items_mapping_dict)
    # users_prefs_distr_df = multiprocess_get_distribution(trainset_df, items_mapping_dict)
    users_prefs_distr_df = big_genre_distribution_with_multiprocessing(trainset_df, items_mapping_dict)

    # Train the recommender algorithm
    logger.info(" ".join(["+", LANGUAGE_TRAINING_THE_RECOMMENDER]))
    reco_instance_worker.fit(trainset)

    # Start the recommendation process with the pos processing
    logger.info(" ".join(["+", LANGUAGE_RECOMMENDATION]))
    # results_df = multiprocessing_recommendations(reco_instance_worker=reco_instance_worker,
    #                                              users_genres_distr_df=users_prefs_distr_df,
    #                                              trainset_df=trainset_df, testset_df=testset_df,
    #                                              items_mapping_dict=items_mapping_dict,
    #                                              blocked_items_df=blocked_items_df,
    #                                              recommender_label=label, transaction_mean=transaction_mean)
    results_df = n_cores_recommendations(reco_instance_worker=reco_instance_worker,
                                         users_genres_distr_df=users_prefs_distr_df,
                                         trainset_df=trainset_df, testset_df=testset_df,
                                         items_mapping_dict=items_mapping_dict,
                                         blocked_items_df=blocked_items_df,
                                         recommender_label=label,
                                         transaction_mean=transaction_mean)
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
