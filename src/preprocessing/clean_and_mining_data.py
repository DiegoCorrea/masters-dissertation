import logging
import os

import numpy as np
import pandas as pd

from src.config.labels import RAW_TRACK_LABEL, RAW_ARTIST_LABEL, RAW_MAJORITY_GENRE, RAW_MINORITY_GENRE, \
    RAW_YEAR_LABEL, RAW_ALBUM_LABEL, RAW_MOVIE_ID_LABEL, ITEM_LABEL, RAW_RATING_LABEL, TRANSACTION_VALUE_LABEL, \
    RAW_USER_LABEL, USER_LABEL, GENRES_LABEL, RAW_TIMESTAMP_LABEL, TITLE_LABEL, RAW_TITLE_LABEL, RAW_GENRES_LABEL, \
    TIME_LABEL
from src.config.language_strings import LANGUAGE_CREATING_FOLDS, LANGUAGE_MINING_PREFERENCES, LANGUAGE_SAVE_TEST, \
    LANGUAGE_SAVE_TRAIN, LANGUAGE_SPLIT_DATA, LANGUAGE_FOLD, LANGUAGE_MINING_ITEMS, LANGUAGE_OMS_SELECTED, \
    LANGUAGE_MOVIELENS_SELECTED
from src.config.path_dir_files import TRANSACTIONS_FILE, ITEMS_FILE, TEST_FILE, TRAIN_FILE, OMS_ITEMS_FILE, \
    MOVIELENS_ITEMS_FILE, MOVIELENS_TRANSACTIONS_FILE, OMS_TRANSACTIONS_FILE, OMS_GENRES_FILE, \
    OMS_TRACK_FILE, load_clean_dataset_path, load_raw_dataset_path, MOVIELENS_TRANSACTIONS_FILE_DAT, \
    MOVIELENS_ITEMS_FILE_DAT, OMS_ITEM_TRACK_FILE, BLOCKED_ITEMS
from src.config.variables import MOVIELENS_PROFILE_LEN_CUT_VALUE, MOVIELENS_ITEM_TRANSACTION_CUT_VALUE, K_FOLDS_VALUES, \
    RATING_CUT_VALUE, MOVIELENS_20M_DATASET, OMS_DATASET, OMS_ITEM_TRANSACTION_CUT_VALUE, OMS_PROFILE_LEN_CUT_VALUE, \
    LISTEN_CUT_VALUE, MOVIELENS_25M_DATASET, OMS_FULL_DATASET
from src.preprocessing.split import big_split_with_multiprocessing

logger = logging.getLogger(__name__)


def create_folds(transactions_df, db):
    for k in range(1, K_FOLDS_VALUES + 1):
        logger.info('+' * 50)
        logger.info("".join([LANGUAGE_FOLD, ': ', str(k)]))
        logger.info(LANGUAGE_SPLIT_DATA)
        # trainset_df, testset_df = split_df(transactions_df)
        trainset_df, testset_df = big_split_with_multiprocessing(transactions_df)
        fold_dir = "".join([load_clean_dataset_path(db), str(k)])
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        logger.info(LANGUAGE_SAVE_TRAIN)
        rating_path = os.path.join(fold_dir, TRAIN_FILE)
        trainset_df.to_csv(rating_path, index=False)
        logger.info(LANGUAGE_SAVE_TEST)
        rating_path = os.path.join(fold_dir, TEST_FILE)
        testset_df.to_csv(rating_path, index=False)


def cut_users(df, cut_value=MOVIELENS_PROFILE_LEN_CUT_VALUE):
    user_counts = df[USER_LABEL].value_counts()
    return df[df[USER_LABEL].isin([k for k, v in user_counts.items() if v >= cut_value])].copy()


def cut_items(df, cut_value=MOVIELENS_ITEM_TRANSACTION_CUT_VALUE):
    item_counts = df[ITEM_LABEL].value_counts()
    return df[df[ITEM_LABEL].isin([k for k, v in item_counts.items() if v >= cut_value])].copy()


def transactions_set_numbers(transactions_df):
    logger.info("".join(["Total of Users: ", str(int(transactions_df[USER_LABEL].nunique()))]))
    logger.info("".join(["Total of Transactions: ", str(len(transactions_df))]))


def items_set_numbers(item_df):
    logger.info("".join(["Total of Items: ", str(int(item_df[ITEM_LABEL].nunique()))]))
    vec = item_df[GENRES_LABEL].tolist()
    genres = []
    for item_genre in vec:
        splitted = item_genre.split('|')
        genre_list = [genre for genre in splitted]
        genres = genres + genre_list
    logger.info("".join(["Total of Genres: ", str(int(len(list(set(genres)))))]))


# ################################################################# #
# Movielens Data
# ################################################################# #
def mining_movielens_transaction_set(db=MOVIELENS_20M_DATASET):
    dataset_raw_dir = load_raw_dataset_path(db)
    dataset_clean_dir = load_clean_dataset_path(db)
    # Load, drop and fill transactions
    if db == MOVIELENS_20M_DATASET or db == MOVIELENS_25M_DATASET:
        raw_transactions_df = pd.read_csv(os.path.join(dataset_raw_dir, MOVIELENS_TRANSACTIONS_FILE))
        # Rename Columns to follow a pattern
        raw_transactions_df.rename(columns={RAW_MOVIE_ID_LABEL: ITEM_LABEL,
                                            RAW_RATING_LABEL: TRANSACTION_VALUE_LABEL,
                                            RAW_USER_LABEL: USER_LABEL,
                                            RAW_TIMESTAMP_LABEL: TIME_LABEL}, inplace=True)
    else:
        raw_transactions_df = pd.read_csv(os.path.join(dataset_raw_dir, MOVIELENS_TRANSACTIONS_FILE_DAT),
                                          names=[USER_LABEL, ITEM_LABEL, TRANSACTION_VALUE_LABEL, TIME_LABEL],
                                          engine='python', sep='::')
    raw_transactions_df.drop([TIME_LABEL], inplace=True, axis=1)
    # Cut
    cut_transactions_df = raw_transactions_df[raw_transactions_df[TRANSACTION_VALUE_LABEL] >= RATING_CUT_VALUE].copy()

    blocked_df = raw_transactions_df[raw_transactions_df[TRANSACTION_VALUE_LABEL] < RATING_CUT_VALUE].copy()
    blocked_df.drop([TRANSACTION_VALUE_LABEL], inplace=True, axis=1)

    cut_transactions_df = cut_users(cut_transactions_df, cut_value=MOVIELENS_PROFILE_LEN_CUT_VALUE)
    final_transactions_df = cut_items(cut_transactions_df, cut_value=MOVIELENS_ITEM_TRANSACTION_CUT_VALUE)
    # Save
    if not os.path.exists(dataset_clean_dir):
        os.makedirs(dataset_clean_dir)
    final_transactions_df.to_csv(os.path.join(dataset_clean_dir, TRANSACTIONS_FILE),
                                 index=False)
    blocked_df.to_csv(os.path.join(dataset_clean_dir, BLOCKED_ITEMS),
                      index=False)
    return final_transactions_df


def mining_movielens_items(items_ids, db=MOVIELENS_20M_DATASET):
    dataset_raw_dir = load_raw_dataset_path(db)
    dataset_clean_dir = load_clean_dataset_path(db)
    # Load transactions set from csv
    if db == MOVIELENS_20M_DATASET or db == MOVIELENS_25M_DATASET:
        item_df = pd.read_csv(os.path.join(dataset_raw_dir, MOVIELENS_ITEMS_FILE))
    else:
        item_df = pd.read_csv(os.path.join(dataset_raw_dir, MOVIELENS_ITEMS_FILE_DAT), engine='python', sep='::',
                              names=[ITEM_LABEL, TITLE_LABEL, GENRES_LABEL])
    item_df.rename(columns={RAW_MOVIE_ID_LABEL: ITEM_LABEL,
                            RAW_TITLE_LABEL: TITLE_LABEL,
                            RAW_GENRES_LABEL: GENRES_LABEL}, inplace=True)
    # Cut
    item_df = item_df[item_df[ITEM_LABEL].isin(items_ids)]
    item_df = item_df[item_df[GENRES_LABEL] != '(no genres listed)']
    # Save
    if not os.path.exists(dataset_clean_dir):
        os.makedirs(dataset_clean_dir)
    item_df.to_csv(os.path.join(dataset_clean_dir, ITEMS_FILE), index=False)
    return item_df


def movielens_mining_data_and_create_fold(db=MOVIELENS_20M_DATASET):
    logger.info(">>>>> " + LANGUAGE_MOVIELENS_SELECTED)
    # Extract the transactions
    logger.info(LANGUAGE_MINING_PREFERENCES)
    transactions_df = mining_movielens_transaction_set(db=db)
    # Extract the items
    logger.info(LANGUAGE_MINING_ITEMS)
    item_df = mining_movielens_items(items_ids=transactions_df[ITEM_LABEL].unique().tolist(), db=db)
    # Creating Folds
    logger.info('-' * 50)
    logger.info(LANGUAGE_CREATING_FOLDS)
    create_folds(transactions_df=transactions_df, db=db)


# ################################################################# #
# One Million Songs Data
# ################################################################# #
def mining_oms_transactions_set(item_df, db):
    dataset_raw_dir = load_raw_dataset_path(db)
    dataset_clean_dir = load_clean_dataset_path(db)
    # Load and filter transactions set from csv
    raw_transactions_df = pd.read_csv(os.path.join(dataset_raw_dir, OMS_TRANSACTIONS_FILE),
                                      names=[USER_LABEL, ITEM_LABEL, TRANSACTION_VALUE_LABEL], sep='\t')
    transactions_df = raw_transactions_df[
        raw_transactions_df[ITEM_LABEL].isin(item_df[ITEM_LABEL].tolist())
    ]
    blocked_df = transactions_df[transactions_df[TRANSACTION_VALUE_LABEL] < LISTEN_CUT_VALUE].copy()
    blocked_df.drop([TRANSACTION_VALUE_LABEL], inplace=True, axis=1)
    transactions_df = transactions_df[transactions_df[TRANSACTION_VALUE_LABEL] >= LISTEN_CUT_VALUE].copy()

    #
    transactions_df = cut_users(transactions_df, OMS_PROFILE_LEN_CUT_VALUE)
    transactions_df = cut_items(transactions_df, OMS_ITEM_TRANSACTION_CUT_VALUE)
    # Save transactions
    if not os.path.exists(dataset_clean_dir):
        os.makedirs(dataset_clean_dir)
    transactions_df.to_csv(os.path.join(dataset_clean_dir, TRANSACTIONS_FILE), index=False)
    blocked_df.to_csv(os.path.join(dataset_clean_dir, BLOCKED_ITEMS),
                      index=False)
    # Find and save new items set
    new_item_df = item_df[
        item_df[ITEM_LABEL].isin(transactions_df[ITEM_LABEL].unique().tolist())
    ]
    new_item_df.to_csv(os.path.join(dataset_clean_dir, ITEMS_FILE), index=False)
    return transactions_df, new_item_df


def load_raw_track(db):
    dataset_raw_dir = load_raw_dataset_path(db)
    song_by_track_df = pd.read_csv(dataset_raw_dir + OMS_TRACK_FILE, engine='python',
                                   sep='<SEP>', names=[RAW_TRACK_LABEL, ITEM_LABEL, RAW_TITLE_LABEL, RAW_ARTIST_LABEL])
    return song_by_track_df.drop([RAW_TITLE_LABEL, RAW_ARTIST_LABEL], axis=1)


def load_raw_gender(db):
    dataset_raw_dir = load_raw_dataset_path(db)
    return pd.read_csv(dataset_raw_dir + OMS_GENRES_FILE,
                       sep='\t', names=[RAW_TRACK_LABEL, RAW_MAJORITY_GENRE, RAW_MINORITY_GENRE], na_values=' ')


def oms_filter_columns(item_df):
    item_df = item_df.replace(np.nan, '', regex=True)
    item_df[GENRES_LABEL] = item_df.apply(
        lambda r: r[RAW_MAJORITY_GENRE] + '|' + r[RAW_MINORITY_GENRE] if r[RAW_MINORITY_GENRE] != '' else r[
            RAW_MAJORITY_GENRE], axis=1)
    item_df.drop([RAW_MAJORITY_GENRE, RAW_MINORITY_GENRE], inplace=True, axis=1)
    return item_df


def mining_oms_items(db):
    dataset_raw_dir = load_raw_dataset_path(db)
    dataset_clean_dir = load_clean_dataset_path(db)
    # Load and drop
    if db == OMS_DATASET or db == OMS_FULL_DATASET:
        raw_items_df = pd.read_csv(os.path.join(dataset_raw_dir, OMS_ITEMS_FILE),
                                   names=[ITEM_LABEL, TITLE_LABEL, RAW_ARTIST_LABEL, RAW_ALBUM_LABEL, RAW_YEAR_LABEL])
        raw_items_df.drop([RAW_ARTIST_LABEL, RAW_ALBUM_LABEL, RAW_YEAR_LABEL], inplace=True, axis=1)
        merged_items_df = pd.merge(
            pd.merge(raw_items_df, load_raw_track(db=db),
                     how='left', left_on=ITEM_LABEL, right_on=ITEM_LABEL
                     ),
            load_raw_gender(db=db), how='inner', left_on=RAW_TRACK_LABEL, right_on=RAW_TRACK_LABEL
        )
    else:
        raw_items_df = pd.read_csv(dataset_raw_dir + OMS_ITEM_TRACK_FILE, engine='python', sep='<SEP>',
                                   names=[RAW_TRACK_LABEL, ITEM_LABEL, RAW_TITLE_LABEL, RAW_ARTIST_LABEL])
        raw_items_df.drop([RAW_ARTIST_LABEL], inplace=True, axis=1)
        merged_items_df = pd.merge(
            raw_items_df,
            load_raw_gender(db=db), how='inner', left_on=RAW_TRACK_LABEL, right_on=RAW_TRACK_LABEL
        )
    # Merge
    merged_items_df.drop_duplicates([ITEM_LABEL], inplace=True)
    merged_items_df.set_index(RAW_TRACK_LABEL, inplace=True, drop=True)
    items_df = oms_filter_columns(merged_items_df)
    if not os.path.exists(dataset_clean_dir):
        os.makedirs(dataset_clean_dir)
    items_df.to_csv(os.path.join(dataset_clean_dir, ITEMS_FILE), index=False)
    return items_df


def oms_mining_data_and_create_folds(db=OMS_DATASET):
    logger.info(">>>>> " + LANGUAGE_OMS_SELECTED)
    logger.info(LANGUAGE_MINING_ITEMS)
    items_df = mining_oms_items(db=db)
    logger.info(LANGUAGE_MINING_PREFERENCES)
    transactions_df, new_item_df = mining_oms_transactions_set(item_df=items_df, db=db)
    logger.info('-' * 50)
    logger.info(LANGUAGE_CREATING_FOLDS)
    create_folds(transactions_df, db=db)
