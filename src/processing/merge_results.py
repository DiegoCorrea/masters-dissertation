import os

import pandas as pd

from src.config.labels import ALGORITHM_LABEL, CALIBRATION_LABEL, FAIRNESS_METRIC_LABEL, LAMBDA_LABEL, \
    LAMBDA_VALUE_LABEL, EVALUATION_METRIC_LABEL, EVALUATION_VALUE_LABEL, EVALUATION_LIST_LABELS
from src.config.path_dir_files import data_results_path, ALL_FOLDS_FILE, ALL_RECOMMENDERS_RESULTS_FILE
from src.config.variables import K_FOLDS_VALUES


def k_fold_results_concat(evaluation_results_df):
    k_results_df = pd.DataFrame()
    for recommender in evaluation_results_df[ALGORITHM_LABEL].unique().tolist():
        recommender_subset_df = evaluation_results_df[evaluation_results_df[ALGORITHM_LABEL] == recommender]
        for calib_method in recommender_subset_df[CALIBRATION_LABEL].unique().tolist():
            calib_subset_df = recommender_subset_df[recommender_subset_df[CALIBRATION_LABEL] == calib_method]
            for distance_metric in calib_subset_df[FAIRNESS_METRIC_LABEL].unique().tolist():
                fairness_subset_df = calib_subset_df[calib_subset_df[FAIRNESS_METRIC_LABEL] == distance_metric]
                for lambda_type in fairness_subset_df[LAMBDA_LABEL].unique().tolist():
                    lambda_subset_df = fairness_subset_df[fairness_subset_df[LAMBDA_LABEL] == lambda_type]
                    for lambda_value in lambda_subset_df[LAMBDA_VALUE_LABEL].unique().tolist():
                        lambda_value_subset_df = lambda_subset_df[lambda_subset_df[LAMBDA_VALUE_LABEL] == lambda_value]
                        for evaluation_metric in lambda_value_subset_df[EVALUATION_METRIC_LABEL].unique().tolist():
                            evaluation_subset_df = lambda_value_subset_df[
                                lambda_value_subset_df[EVALUATION_METRIC_LABEL] == evaluation_metric]
                            result = evaluation_subset_df[EVALUATION_VALUE_LABEL].mean()
                            k_results_df = pd.concat([k_results_df,
                                                      pd.DataFrame(
                                                          [[recommender,
                                                            calib_method,
                                                            distance_metric,
                                                            lambda_type,
                                                            lambda_value,
                                                            evaluation_metric,
                                                            result]],
                                                          columns=EVALUATION_LIST_LABELS
                                                      )
                                                      ])
    return k_results_df


def merge_recommender_results(label, db):
    evaluation_results_df = pd.DataFrame()
    for fold in range(1, K_FOLDS_VALUES + 1):
        tmp = pd.read_csv(os.path.join("/".join([data_results_path(db), label]) + "/", str(fold) + '.csv'))
        evaluation_results_df = pd.concat([evaluation_results_df, tmp])
    path_to_save = "".join([data_results_path(db), label, "/"])
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    evaluation_concat_df = k_fold_results_concat(evaluation_results_df)
    evaluation_concat_df.to_csv(os.path.join(path_to_save, ALL_FOLDS_FILE),
                                index=False)


def merge_all_results(recommenders_labels, db):
    evaluation_results_df = pd.DataFrame()
    for label in recommenders_labels:
        tmp = pd.read_csv(os.path.join("/".join([data_results_path(db), label]) + "/", ALL_FOLDS_FILE))
        evaluation_results_df = pd.concat([evaluation_results_df, tmp])
    path_to_save = data_results_path(db)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    evaluation_results_df.to_csv(os.path.join(path_to_save, ALL_RECOMMENDERS_RESULTS_FILE),
                                 index=False)
