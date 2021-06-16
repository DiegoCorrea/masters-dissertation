from src.config.variables import MOVIELENS_20M_DATASET, OMS_DATASET, MOVIELENS_1M_DATASET, OMS_10K_DATASET, \
    MOVIELENS_25M_DATASET, OMS_FULL_DATASET, MOVIELENS_DATASET_LIST
from src.preprocessing.clean_and_mining_data import movielens_mining_data_and_create_fold, \
    oms_mining_data_and_create_folds


def link_and_clean_datasets():
    # movielens_mining_data_and_create_fold(db=MOVIELENS_25M_DATASET)
    movielens_mining_data_and_create_fold(db=MOVIELENS_20M_DATASET)
    # movielens_mining_data_and_create_fold(db=MOVIELENS_1M_DATASET)
    oms_mining_data_and_create_folds(db=OMS_FULL_DATASET)
    # oms_mining_data_and_create_folds(db=OMS_DATASET)
    # oms_mining_data_and_create_folds(db=OMS_10K_DATASET)


def mining_and_clean_dataset(db):
    if db in MOVIELENS_DATASET_LIST:
        movielens_mining_data_and_create_fold(db=db)
    else:
        oms_mining_data_and_create_folds(db=db)
