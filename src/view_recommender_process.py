import logging
from src.config.labels import FULL_RECOMMENDERS, TEST_RECOMMENDERS, LEARNING_RECOMMENDERS
from src.config.language_strings import LANGUAGE_MOVIELENS_SELECTED, LANGUAGE_OMS_SELECTED
from src.config.variables import MOVIELENS_20M_DATASET, OMS_DATASET
from src.preprocessing.load_database import load_db_and_fold
from src.processing.merge_results import merge_all_results
from src.processing.recommender_algorithms import all_learning_recommenders, test_learning_recommender, \
    personalized_learning_recommender
from src.processing.traditional_recommenders import all_traditional_recommenders, personalized_traditional_recommender

logger = logging.getLogger(__name__)


def recommender_process():
    """
    The recommendation system process execute all recommenders algorithms to all datasets
    """
    # Start recommenders with movielens dataset
    logger.info("$" * 50)
    logger.info(" ".join(['>', LANGUAGE_MOVIELENS_SELECTED]))

    all_learning_recommenders(db=MOVIELENS_20M_DATASET)
    all_traditional_recommenders(db=MOVIELENS_20M_DATASET)

    merge_all_results(FULL_RECOMMENDERS, db=MOVIELENS_20M_DATASET)
    # Start recommenders with OMS dataset
    logger.info("$" * 50)
    logger.info(" ".join(['>', LANGUAGE_OMS_SELECTED]))

    all_learning_recommenders(db=OMS_DATASET)
    all_traditional_recommenders(db=OMS_DATASET)

    merge_all_results(FULL_RECOMMENDERS, db=OMS_DATASET)


def one_recommender_process():
    """
    Test the recommendation process
    """
    # Start recommenders with movielens dataset
    logger.info("$" * 50)
    logger.info(" ".join(['>', LANGUAGE_MOVIELENS_SELECTED]))

    # all_traditional_recommenders(db=MOVIELENS_20M_DATASET)
    test_learning_recommender(db=MOVIELENS_20M_DATASET)

    merge_all_results(TEST_RECOMMENDERS, db=MOVIELENS_20M_DATASET)
    # Start recommenders with OMS dataset
    logger.info("$" * 50)
    logger.info(" ".join(['>', LANGUAGE_OMS_SELECTED]))

    # all_traditional_recommenders(db=OMS_DATASET)
    test_learning_recommender(db=OMS_DATASET)

    merge_all_results(TEST_RECOMMENDERS, db=OMS_DATASET)


def customized_start_recommender(recommender, dataset, fold):
    if recommender in LEARNING_RECOMMENDERS:
        personalized_learning_recommender(label=recommender, db=dataset, fold=fold)
    else:
        personalized_traditional_recommender(label=recommender, db=dataset, fold=fold)
