from src.config.labels import GRID_SEARCH_RECOMMENDERS
from src.config.language_strings import LANGUAGE_MOVIELENS_SELECTED, LANGUAGE_OMS_SELECTED
from src.config.variables import MOVIELENS_20M_DATASET, OMS_DATASET, MOVIELENS_1M_DATASET
from src.conversions.pandas_to_models import transform_all_dataset
from src.processing.grid_search import grid_search_recommenders, grid_search_fold, grid_search_cv
from src.preprocessing.load_database import movielens_load_full_data, oms_load_full_data
import logging
logger = logging.getLogger(__name__)


def grid_search_movielens():
    """"
    Start the grid search to all recommenders algorithms using the Movielens dataset
    """
    logger.info("$" * 50)
    logger.info(" ".join(['>', LANGUAGE_MOVIELENS_SELECTED]))
    # Load the Movielens dataset
    transactions_df, items_df = movielens_load_full_data(db=MOVIELENS_1M_DATASET)
    # Transform the dataset by the Surprise Reader class
    dataset = transform_all_dataset(transactions_df)
    # Start the grid search
    grid_search_recommenders(GRID_SEARCH_RECOMMENDERS, dataset, db=MOVIELENS_1M_DATASET)


def grid_search_oms():
    """"
    Start the grid search to all recommenders algorithms using the OMS dataset
    """
    logger.info("$" * 50)
    logger.info(" ".join(['>', LANGUAGE_OMS_SELECTED]))
    # Load the OMS dataset
    transactions_df, items_df = oms_load_full_data(db=OMS_DATASET)
    # Transform the dataset by the Surprise Reader class
    dataset = transform_all_dataset(transactions_df)
    # Start the grid search
    grid_search_recommenders(GRID_SEARCH_RECOMMENDERS, dataset, db=OMS_DATASET)


def grid_search():
    """
    Grid Search start function
    """
    grid_search_movielens()
    grid_search_oms()


def customized_grid_search(recommender, dataset, cv):
    grid_search_cv(recommender_label=recommender, db=dataset, cv=cv)
