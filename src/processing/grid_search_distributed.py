import json
import logging
import os

import joblib
import dask.distributed
from dask_jobqueue import SLURMCluster
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
    cluster = SLURMCluster(cores=24,
                           processes=2,
                           memory='64GB',
                           queue="nvidia_dev",
                           project="SVD",
                           name=label,
                           log_directory='logs/slurm',
                           walltime='00:15:00')
    # cluster.scale(2)
    cluster.adapt(minimum=1, maximum=360)
    client = dask.distributed.Client(cluster)
    print(client)
    print(cluster.job_script())
    gs = GridSearchCV(instance, params, measures=measures, cv=folds, joblib_verbose=100)
    with joblib.parallel_backend("dask"):
        print(client)
        gs.fit(dataset)
    return gs


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


def grid_search_recommenders_parallel(recommenders_label_list, dataset, db):
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
