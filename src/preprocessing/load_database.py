import logging
import os

import pandas as pd

from src.config.labels import USER_LABEL, ITEM_LABEL, TITLE_LABEL, GENRES_LABEL, TRANSACTION_VALUE_LABEL, TIME_LABEL, \
    RAW_YEAR_LABEL, RAW_ALBUM_LABEL, RAW_ARTIST_LABEL, RAW_TRACK_LABEL, RAW_MAJORITY_GENRE, RAW_MINORITY_GENRE
from src.config.path_dir_files import TRAIN_FILE, TEST_FILE, ITEMS_FILE, TRANSACTIONS_FILE, \
    MOVIELENS_TRANSACTIONS_FILE, MOVIELENS_ITEMS_FILE, OMS_TRANSACTIONS_FILE, OMS_ITEMS_FILE, OMS_GENRES_FILE, \
    load_clean_dataset_path, load_raw_dataset_path, MOVIELENS_ITEMS_FILE_DAT, MOVIELENS_TRANSACTIONS_FILE_DAT, \
    BLOCKED_ITEMS
from src.config.variables import MOVIELENS_20M_DATASET, MOVIELENS_1M_DATASET, MOVIELENS_DATASET_LIST, \
    MOVIELENS_25M_DATASET

logger = logging.getLogger(__name__)


# ################################################################# #
# Load Movielens Clean Data
# ################################################################# #
def movielens_load_blocked_list(db):
    return pd.read_csv(os.path.join(load_clean_dataset_path(db), BLOCKED_ITEMS))


def movielens_load_full_dataset(db):
    return pd.read_csv(os.path.join(load_clean_dataset_path(db), TRANSACTIONS_FILE))


def movielens_load_preference_trainset(fold, db):
    return pd.read_csv(os.path.join(load_clean_dataset_path(db) + str(fold), TRAIN_FILE))


def movielens_load_preference_testset(fold, db):
    return pd.read_csv(os.path.join(load_clean_dataset_path(db) + str(fold), TEST_FILE))


def movielens_load_items(db):
    return pd.read_csv(os.path.join(load_clean_dataset_path(db), ITEMS_FILE))


def movielens_load_data(fold, db):
    trainset_df = movielens_load_preference_trainset(fold=fold, db=db)
    testset_df = movielens_load_preference_testset(fold=fold, db=db)
    item_df = movielens_load_items(db=db)
    trainset_df = trainset_df.merge(item_df, on=ITEM_LABEL)
    testset_df = testset_df.merge(item_df, on=ITEM_LABEL)
    for col in (USER_LABEL, ITEM_LABEL):
        trainset_df[col] = trainset_df[col].astype('category')
        testset_df[col] = testset_df[col].astype('category')
    return trainset_df, testset_df, item_df


def movielens_load_full_data(db):
    raw_transactions_df = movielens_load_full_dataset(db=db)
    items_df = movielens_load_items(db=db)
    transactions_df = raw_transactions_df.merge(items_df, on=ITEM_LABEL)
    for col in (USER_LABEL, ITEM_LABEL):
        transactions_df[col] = transactions_df[col].astype('category')
    return transactions_df, items_df


# ################################################################# #
# Load OMS Clean Data
# ################################################################# #
def oms_load_blocked_list(db):
    return pd.read_csv(os.path.join(load_clean_dataset_path(db), BLOCKED_ITEMS))


def oms_load_full_preference(db):
    return pd.read_csv(os.path.join(load_clean_dataset_path(db), TRANSACTIONS_FILE))


def oms_load_preference_trainset(fold, db):
    return pd.read_csv(os.path.join(load_clean_dataset_path(db) + str(fold), TRAIN_FILE))


def oms_load_preference_testset(fold, db):
    return pd.read_csv(os.path.join(load_clean_dataset_path(db) + str(fold), TEST_FILE))


def oms_load_items(db):
    return pd.read_csv(os.path.join(load_clean_dataset_path(db), ITEMS_FILE))


def oms_load_data(fold, db):
    trainset_df = oms_load_preference_trainset(fold=fold, db=db)
    testset_df = oms_load_preference_testset(fold=fold, db=db)
    item_df = oms_load_items(db=db)
    trainset_df = trainset_df.merge(item_df, on=ITEM_LABEL)
    testset_df = testset_df.merge(item_df, on=ITEM_LABEL)
    for col in (USER_LABEL, ITEM_LABEL):
        trainset_df[col] = trainset_df[col].astype('category')
        testset_df[col] = testset_df[col].astype('category')
    return trainset_df, testset_df, item_df


def oms_load_full_data(db):
    raw_transactions_df = oms_load_full_preference(db=db)
    items_df = oms_load_items(db=db)
    transactions_df = raw_transactions_df.merge(items_df, on=ITEM_LABEL)
    for col in (USER_LABEL, ITEM_LABEL):
        transactions_df[col] = transactions_df[col].astype('category')
    return transactions_df, items_df


# ################################################################# #
# Load Raw Data
# ################################################################# #
def movielens_load_raw_items(db):
    dataset_raw_dir = load_raw_dataset_path(db)
    if db == MOVIELENS_20M_DATASET or db == MOVIELENS_25M_DATASET:
        df = pd.read_csv(os.path.join(dataset_raw_dir, MOVIELENS_ITEMS_FILE), header=0,
                         names=[ITEM_LABEL, TITLE_LABEL, GENRES_LABEL])
        # df.set_axis([ITEM_LABEL, TITLE_LABEL, GENRES_LABEL], axis=1, inplace=False)
    else:
        df = pd.read_csv(os.path.join(dataset_raw_dir, MOVIELENS_ITEMS_FILE_DAT),
                         engine='python', sep='::', header=0,
                         names=[ITEM_LABEL, TITLE_LABEL, GENRES_LABEL])
    return df


def movielens_load_raw_preferences(db):
    dataset_raw_dir = load_raw_dataset_path(db)
    if db == MOVIELENS_20M_DATASET or db == MOVIELENS_25M_DATASET:
        df = pd.read_csv(os.path.join(dataset_raw_dir, MOVIELENS_TRANSACTIONS_FILE), header=0,
                         names=[USER_LABEL, ITEM_LABEL, TRANSACTION_VALUE_LABEL, TIME_LABEL])
        # df.set_axis([USER_LABEL, ITEM_LABEL, TRANSACTION_VALUE_LABEL, TIME_LABEL], axis=1, inplace=False)
    else:
        df = pd.read_csv(os.path.join(dataset_raw_dir, MOVIELENS_TRANSACTIONS_FILE_DAT),
                         names=[USER_LABEL, ITEM_LABEL, TRANSACTION_VALUE_LABEL, TIME_LABEL],
                         engine='python', sep='::')
    return df


def oms_load_raw_items(db):
    return pd.read_csv(os.path.join(load_raw_dataset_path(db), OMS_ITEMS_FILE),
                       names=[ITEM_LABEL, TITLE_LABEL, RAW_ARTIST_LABEL, RAW_ALBUM_LABEL, RAW_YEAR_LABEL])


def oms_load_raw_preferences(db):
    return pd.read_csv(os.path.join(load_raw_dataset_path(db), OMS_TRANSACTIONS_FILE),
                       names=[USER_LABEL, ITEM_LABEL, TRANSACTION_VALUE_LABEL], sep='\t')


def oms_load_raw_genres(db):
    return pd.read_csv(load_raw_dataset_path(db) + OMS_GENRES_FILE, sep='\t',
                       names=[RAW_TRACK_LABEL, RAW_MAJORITY_GENRE, RAW_MINORITY_GENRE],
                       na_values=' ')


# #####################################################################################################
def load_db_and_fold(db, fold):
    if db in MOVIELENS_DATASET_LIST:
        trainset_df, testset_df, items_df = movielens_load_data(fold, db=db)
    else:
        trainset_df, testset_df, items_df = oms_load_data(fold, db=db)
    return trainset_df, testset_df, items_df


def load_complete_transactions_db_and_fold(db):
    if db in MOVIELENS_DATASET_LIST:
        transactions_df = movielens_load_full_dataset(db=db)
    else:
        transactions_df = oms_load_full_preference(db=db)
    return transactions_df


def load_blocked_list(db):
    if db in MOVIELENS_DATASET_LIST:
        blocked_df = movielens_load_blocked_list(db=db)
    else:
        blocked_df = oms_load_blocked_list(db=db)
    return blocked_df
