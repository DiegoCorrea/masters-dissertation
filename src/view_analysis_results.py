import logging
from os import listdir
from os.path import isfile, join

from src.config import path_dir_files
from src.config.path_dir_files import data_results_path, ALL_RECOMMENDERS_RESULTS_FILE
from src.config.variables import MOVIELENS_20M_DATASET, OMS_FULL_DATASET
from src.evaluation.coefficient import coefficient
from src.processing.merge_results import merge_all_results, merge_recommender_results
import pandas as pd
import os

logger = logging.getLogger(__name__)


def analysis_results(dbs=None):
    if dbs is None:
        dbs = [MOVIELENS_20M_DATASET, OMS_FULL_DATASET]
    for db in dbs:
        path = path_dir_files.data_results_path(db)
        recommenders = [f for f in listdir(path) if not isfile(join(path, f))]
        logger.info(" ".join(["+", db]))
        for recommender in recommenders:
            merge_recommender_results(recommender, db=db)
        merge_all_results(recommenders, db=db)


def coefficient_for_calibration(dbs=None):
    if dbs is None:
        dbs = [MOVIELENS_20M_DATASET, OMS_FULL_DATASET]
    for db in dbs:
        logger.info(" ".join(["+", db]))
        evaluation_results_df = pd.read_csv(os.path.join(data_results_path(db), ALL_RECOMMENDERS_RESULTS_FILE))
        coefficient(evaluation_results_df, db)
