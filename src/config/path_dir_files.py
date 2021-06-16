from src.config.language_strings import lang

# Datasets files name
from src.config.variables import MOVIELENS_25M_DATASET, OMS_10K_DATASET, MOVIELENS_1M_DATASET, OMS_DATASET, \
    MOVIELENS_20M_DATASET

MOVIELENS_TRANSACTIONS_FILE = 'rating.csv'
MOVIELENS_ITEMS_FILE = 'movie.csv'
MOVIELENS_TRANSACTIONS_FILE_DAT = 'ratings.dat'
MOVIELENS_ITEMS_FILE_DAT = 'movies.dat'

OMS_TRANSACTIONS_FILE = 'train_triplets.txt'
OMS_GENRES_FILE = 'msd_tagtraum_cd2.cls'
OMS_ITEMS_FILE = 'songs.csv'
OMS_TRACK_FILE = 'unique_tracks.txt'
OMS_ITEM_TRACK_FILE = 'subset_unique_tracks.txt'

# Datasets directory names
MOVIELENS_25M_DIR = "ml-25m/"
MOVIELENS_20M_DIR = "movielens-20m/"
MOVIELENS_1M_DIR = "ml-1m/"

OMS_DIR = "oms/"
OMS_10K_DIR = "oms-10k/"
OMS_FULL_DIR = "oms-full/"

# Files names
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
TRANSACTIONS_FILE = 'transactions.csv'
ITEMS_FILE = 'item.csv'
DATA_ANALYSIS_FILE = 'data_analysis.csv'
ALL_FOLDS_FILE = 'all_folds.csv'
ALL_RECOMMENDERS_RESULTS_FILE = 'all_recommenders_results.csv'
BLOCKED_ITEMS = 'blocked_items.csv'


# Adaptive path functions
def used_db(db):
    if db == MOVIELENS_20M_DATASET:
        return MOVIELENS_20M_DIR
    elif db == OMS_DATASET:
        return OMS_DIR
    elif db == MOVIELENS_1M_DATASET:
        return MOVIELENS_1M_DIR
    elif db == OMS_10K_DATASET:
        return OMS_10K_DIR
    elif db == MOVIELENS_25M_DATASET:
        return MOVIELENS_25M_DIR
    else:
        return OMS_FULL_DIR


def load_raw_dataset_path(db):
    return "/".join(['datasets/raw', used_db(db)])


def load_clean_dataset_path(db):
    return "/".join(['datasets/clean', used_db(db)])


def pre_processing_to_use_path(db):
    return "/".join(['results/database', lang, used_db(db)])


def pos_processing_results_path(db):
    return "/".join(['results/pos_processing', lang, used_db(db)])


def data_results_path(db):
    return "/".join(['results/data', used_db(db)])


def post_graphics_results_path(db):
    return "/".join(['results/graphics/post_processing', lang, used_db(db)])


def coefficient_results_path(db):
    return "/".join(['results/coefficient', lang, used_db(db)])


def data_analyze_path():
    return 'results/data/dataset/'


def grid_search_path(db):
    return "/".join(['src/config/recommenders_params', used_db(db)])
