import logging
import multiprocessing
from copy import deepcopy
from multiprocessing import Process
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.labels import USER_LABEL
from src.config.variables import TEST_SIZE, N_CORES

logger = logging.getLogger(__name__)


def map_user_split(df):
    train, test = train_test_split(df, test_size=TEST_SIZE)
    return (train, test)


def split_df(df):
    users_ids = df[USER_LABEL].unique().tolist()
    map_results_df = [map_user_split(df[df[USER_LABEL] == user_id].copy()) for user_id in users_ids]
    list1 = [x[0] for x in map_results_df]
    list2 = [x[1] for x in map_results_df]
    train_results_df = pd.concat(list1, sort=False)
    test_results_df = pd.concat(list2, sort=False)
    return train_results_df, test_results_df


def user_split(df, shared_queue):
    train, test = train_test_split(df, test_size=TEST_SIZE)
    shared_queue.put(deepcopy((train, test)))


def split_with_multiprocessing(df):
    # Preparing: users, results dataframe and shared queue over processes
    users_ids = df[USER_LABEL].unique().tolist()

    train_results_df = pd.DataFrame()
    test_results_df = pd.DataFrame()

    manager = multiprocessing.Manager()
    shared_queue = manager.Queue()
    # While has users to process do in nbeans (ncores)
    while users_ids:
        cores_control = list(range(0, N_CORES))
        all_processes = list()

        # Print the state of the execution
        i = len(users_ids)
        print(f'\r{i} users to finish', end='', flush=True)
        while users_ids and cores_control:
            # Allocate core and select the user to pos process
            cores_control.pop(0)
            user_id = users_ids.pop(0)
            # Prepare user data
            user_model_df = deepcopy(df[df[USER_LABEL] == user_id])
            # Create the process
            p = Process(target=user_split,
                        args=(user_model_df, shared_queue))
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
        train = pd.concat([x[0] for x in results], sort=False)
        test = pd.concat([x[1] for x in results], sort=False)
        train_results_df = pd.concat([train_results_df, train], sort=False)
        test_results_df = pd.concat([test_results_df, test], sort=False)
    return train_results_df, test_results_df


def big_user_split(df, shared_queue):
    train, test = split_df(df)
    shared_queue.put(deepcopy((train, test)))


def big_split_with_multiprocessing(df):
    # Preparing: users, results dataframe and shared queue over processes
    users_ids = df[USER_LABEL].unique().tolist()
    grous_split_id = np.array_split(users_ids, N_CORES)

    train_results_df = pd.DataFrame()
    test_results_df = pd.DataFrame()

    manager = multiprocessing.Manager()
    shared_queue = manager.Queue()
    # While has users to process do in nbeans (ncores)
    cores_control = list(range(0, N_CORES))
    all_processes = list()

    # Print the state of the execution
    while users_ids and cores_control:
        # Allocate core and select the user to pos process
        cores_control.pop(0)
        user_id = grous_split_id.pop(0)
        # Prepare user data
        user_model_df = deepcopy(df[df[USER_LABEL].isin(user_id)])
        # Create the process
        p = Process(target=big_user_split,
                    args=(user_model_df, shared_queue))
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
    train_results_df = pd.concat([x[0] for x in results], sort=False)
    test_results_df = pd.concat([x[1] for x in results], sort=False)
    # train_results_df = pd.concat([train_results_df, train], sort=False)
    # test_results_df = pd.concat([test_results_df, test], sort=False)
    return train_results_df, test_results_df