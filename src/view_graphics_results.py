import logging
import pandas as pd
import os

from src.config.path_dir_files import data_results_path, ALL_RECOMMENDERS_RESULTS_FILE
from src.config.variables import OMS_DATASET, MOVIELENS_20M_DATASET, MOVIELENS_1M_DATASET, OMS_FULL_DATASET
from src.graphics.experimental_evaluation import evaluation_linear_fairness_by_algo_over_lambda, evaluation_map_by_mrmc, \
    evaluation_map_by_mace
logger = logging.getLogger(__name__)


def generating_results_graphics(dbs=None):
    if dbs is None:
        dbs = [MOVIELENS_20M_DATASET, OMS_FULL_DATASET]
    for db in dbs:
        logger.info(" ".join(["+", db]))
        evaluation_results_df = pd.read_csv(os.path.join(data_results_path(db), ALL_RECOMMENDERS_RESULTS_FILE))
        evaluation_linear_fairness_by_algo_over_lambda(evaluation_results_df, 0, db)
        evaluation_map_by_mrmc(evaluation_results_df, 0, db)
        evaluation_map_by_mace(evaluation_results_df, 0, db)
