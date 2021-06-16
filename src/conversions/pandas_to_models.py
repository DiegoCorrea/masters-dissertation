from copy import deepcopy

import pandas as pd
from surprise import Reader, Dataset

from src.config.labels import USER_LABEL, ITEM_LABEL, TRANSACTION_VALUE_LABEL, ORDER_LABEL
from src.config.language_strings import LANGUAGE_PANDAS_TO_SURPRISE_DATA
from src.models.item import create_item_mapping


def user_transactions_df_to_item_mapping(user_transactions_df, item_mapping):
    user_items = {}
    for row in user_transactions_df.itertuples():
        item_id = getattr(row, ITEM_LABEL)
        user_items[item_id] = deepcopy(item_mapping[item_id])
        user_items[item_id].score = getattr(row, TRANSACTION_VALUE_LABEL)
    return user_items


def transform_all_transaction_df_to_item_mapping(transactions_df, item_mapping):
    print(LANGUAGE_PANDAS_TO_SURPRISE_DATA)
    transactions_item_mapping = {}
    for user_id in transactions_df[USER_LABEL].unique().tolist():
        user_transactions_df = transactions_df[transactions_df[USER_LABEL] == user_id]
        transactions_item_mapping[user_id] = user_transactions_df_to_item_mapping(user_transactions_df, item_mapping)
    return transactions_item_mapping


def items_to_pandas(users_items):
    results_df = pd.DataFrame()
    order = 0
    user_results = []
    for item_id, item in users_items.items():
        order += 1
        user_results += [pd.DataFrame(data=[[item_id, item.score, order]],
                                      columns=[ITEM_LABEL, TRANSACTION_VALUE_LABEL, ORDER_LABEL])]
    user_results = pd.concat(user_results, sort=False)
    results_df = pd.concat([results_df, user_results], sort=False)
    return results_df


def transform_trainset(trainset_df):
    min = trainset_df[TRANSACTION_VALUE_LABEL].min()
    max = trainset_df[TRANSACTION_VALUE_LABEL].max()
    reader_train = Reader(rating_scale=(min, max))
    data_train = Dataset.load_from_df(trainset_df[[USER_LABEL, ITEM_LABEL, TRANSACTION_VALUE_LABEL]], reader_train)
    return data_train.build_full_trainset()


def transform_testset(testset_df):
    min = testset_df[TRANSACTION_VALUE_LABEL].min()
    max = testset_df[TRANSACTION_VALUE_LABEL].max()
    reader_test = Reader(rating_scale=(min, max))
    data_test = Dataset.load_from_df(testset_df[[USER_LABEL, ITEM_LABEL, TRANSACTION_VALUE_LABEL]], reader_test)
    testset = data_test.build_full_trainset()
    return testset.build_testset()


def transform_all_dataset(dataset_df):
    min = dataset_df[TRANSACTION_VALUE_LABEL].min()
    max = dataset_df[TRANSACTION_VALUE_LABEL].max()
    reader_train = Reader(rating_scale=(min, max))
    return Dataset.load_from_df(dataset_df[[USER_LABEL, ITEM_LABEL, TRANSACTION_VALUE_LABEL]], reader_train)
