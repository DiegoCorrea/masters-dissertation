import json
import logging
import os

from surprise import SVD, KNNBasic
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.matrix_factorization import SVDpp, NMF

from src.config.labels import NMF_LABEL, SVD_LABEL, SVDpp_LABEL, ITEM_KNN_LABEL, USER_KNN_LABEL, CO_CLUSTERING_LABEL
from src.config.language_strings import LANGUAGE_GRID_SEARCH
from src.config.path_dir_files import grid_search_path
from src.config.recommenders_params.grid_params import USERKNN_GRID_PARAMS, ITEMKNN_GRID_PARAMS, SVD_GRID_PARAMS, \
    SVDpp_GRID_PARAMS, NMF_GRID_PARAMS, CLUSTERING_GRID_PARAMS
from src.config.variables import N_CORES, K_FOLDS_VALUES
from src.conversions.pandas_to_models import transform_all_dataset
from src.preprocessing.load_database import load_complete_transactions_db_and_fold

logger = logging.getLogger(__name__)


def grid_search_instance(instance, params, dataset, measures, folds, label, n_jobs=N_CORES):
    """
    Grid Search Cross Validation to get the best params to the recommender algorithm
    :param label:
    :param instance: Recommender algorithm instance
    :param params: Recommender algorithm params set
    :param dataset: A dataset modeled by the surprise Reader class
    :param measures: A string with the measure name
    :param folds: Number of folds to cross validation
    :param n_jobs: Number of CPU/GPU to be used
    :return: A Grid Search instance
    """
    if label == SVDpp_LABEL:
        n = N_CORES//2
        gs = GridSearchCV(instance, params, measures=measures, cv=folds, n_jobs=n, joblib_verbose=100)
    else:
        gs = GridSearchCV(instance, params, measures=measures, cv=folds, n_jobs=n_jobs, joblib_verbose=100)
    # gs = GridSearchCV(instance, params, measures=measures, cv=folds, n_jobs=n_jobs, joblib_verbose=100)
    gs.fit(dataset)
    return gs


def grid_search_one(dataset, db):
    """
    A helper function to be used like the grid_search_all_recommenders
    :param db: 0 to Movielens; 1 to OMS;
    :param dataset: A dataset modeled by the surprise Reader class
    :return: No return
    """
    logger.info(" ".join(['> >', LANGUAGE_GRID_SEARCH]))
    recommenders_instance_list = [SVD]
    recommenders_label_list = [SVD_LABEL]
    recommenders_params_list = [SVD_GRID_PARAMS]
    for instance, label, params in zip(recommenders_instance_list,
                                       recommenders_label_list,
                                       recommenders_params_list):
        logger.info(" ".join(['> > >', label]))
        gs = grid_search_instance(instance=instance, params=params,
                                  dataset=dataset, measures=['mae'],
                                  folds=K_FOLDS_VALUES,
                                  label=label)
        print("mae score", gs.best_score['mae'])
        path_to_save = grid_search_path(db)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        with open("".join([path_to_save, label, ".json"]), 'w') as fp:
            json.dump(gs.best_params['mae'], fp)


def callable_recommender_and_params(label):
    if label == SVD_LABEL:
        return SVD, SVD_GRID_PARAMS
    elif label == NMF_LABEL:
        return NMF, NMF_GRID_PARAMS
    elif label == CO_CLUSTERING_LABEL:
        return CoClustering, CLUSTERING_GRID_PARAMS
    elif label == ITEM_KNN_LABEL:
        return KNNBasic, ITEMKNN_GRID_PARAMS
    elif label == USER_KNN_LABEL:
        return KNNBasic, USERKNN_GRID_PARAMS
    else:
        return SVDpp, SVDpp_GRID_PARAMS


def grid_search_recommenders(recommenders_label_list, dataset, db):
    """
    Preparing all recommender algorithms to run in grid search
    :param recommenders_label_list:
    :param db: 0 to Movielens; 1 to OMS;
    :param dataset: A dataset modeled by the surprise Reader class
    :return: None
    """
    logger.info(" ".join(['> >', LANGUAGE_GRID_SEARCH]))
    for label in recommenders_label_list:
        logger.info(" ".join(['> > >', label]))
        instance, params = callable_recommender_and_params(label)
        gs = grid_search_instance(instance=instance, params=params,
                                  dataset=dataset, measures=['mae'],
                                  folds=K_FOLDS_VALUES,
                                  label=label)
        path_to_save = grid_search_path(db)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        with open("".join([path_to_save, label, ".json"]), 'w') as fp:
            json.dump(gs.best_params['mae'], fp)

# ################################################################################################ #
# ################################################################################################ #
# ################################################################################################ #


def grid_search_fold(recommender_label, db, fold):
    """
    Preparing all recommender algorithms to run in grid search
    :param recommender_label:
    :param db: 0 to Movielens; 1 to OMS;
    :param fold:
    :return: None
    """
    transactions_df = load_complete_transactions_db_and_fold(db)

    # Transform the dataset by the Surprise Reader class
    dataset = transform_all_dataset(transactions_df)

    instance, params = callable_recommender_and_params(recommender_label)

    gs = grid_search_instance(instance=instance, params=params,
                              dataset=dataset, measures=['mae'],
                              folds=1,
                              label=recommender_label)

    path_to_save = grid_search_path(db)
    path_to_save = path_to_save + recommender_label + '/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    with open("".join([path_to_save, str(fold), ".json"]), 'w') as fp:
        json.dump(gs.best_params['mae'], fp)


def grid_search_cv(recommender_label, db, cv):
    """
    Preparing all recommender algorithms to run in grid search
    :param recommender_label:
    :param db: 0 to Movielens; 1 to OMS;
    :param cv:
    :return: None
    """
    transactions_df = load_complete_transactions_db_and_fold(db)

    # Transform the dataset by the Surprise Reader class
    dataset = transform_all_dataset(transactions_df)

    instance, params = callable_recommender_and_params(recommender_label)

    gs = grid_search_instance(instance=instance, params=params,
                              dataset=dataset, measures=['mae'],
                              folds=cv,
                              label=recommender_label)

    path_to_save = grid_search_path(db)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    with open("".join([path_to_save, recommender_label, ".json"]), 'w') as fp:
        json.dump(gs.best_params['mae'], fp)
