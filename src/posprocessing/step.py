import numpy as np
import pandas as pd

import logging
import logging
from src.config.labels import KL_LABEL, HE_LABEL, CHI_LABEL, FAIRNESS_METRIC_LABEL, VARIANCE_TRADE_OFF_LABEL, \
    COUNT_GENRES_TRADE_OFF_LABEL, TRADE_OFF_LABEL, EVALUATION_LIST_LABELS, MACE_LABEL, FIXED_LABEL, MAP_LABEL, \
    MRR_LABEL, ORDER_LABEL, MC_LABEL, CALIBRATION_LABEL, LINEAR_CALIBRATION_LABEL, LOGARITHMIC_CALIBRATION_LABEL
from src.conversions.pandas_to_models import items_to_pandas
from src.evaluation.mace import ace
from src.evaluation.map import average_precision
from src.evaluation.misscalibration import mc
from src.evaluation.mrr import mrr
from src.posprocessing.greedy_algorithms import surrogate_submodular
from src.posprocessing.trade_off import personalized_trade_off


logger = logging.getLogger(__name__)


def pos_processing_calibration(user_model_genres_distr_df, candidates_items_mapping,
                               user_expected_items_ids, recommender_label, transaction_mean):
    """
    The pos process step to calibrate the final recommendation list based on the user model

    :param transaction_mean:
    :param user_model_genres_distr_df: The user genres distribution
    :param candidates_items_mapping: A dict with the candidates items maximized to the user
    :param user_expected_items_ids: The id of thea user expected items in the final recommendation
    :param recommender_label: The recommender algorithm label
    :return: A dataframe with the user results of all used metrics
    """
    # Start the config execution and create the dataframe to return the results
    config = dict()
    evaluation_results_df = pd.DataFrame()
    for calib_method in LINEAR_CALIBRATION_LABEL, LOGARITHMIC_CALIBRATION_LABEL:
        # logger.info(str('Calib: ' + calib_method))
        config[CALIBRATION_LABEL] = calib_method
        # For each divergence measure do
        for distance_metric in KL_LABEL, HE_LABEL, CHI_LABEL:
            config[FAIRNESS_METRIC_LABEL] = distance_metric

            # For each fixed lambda value
            for lambda_value in np.arange(0, 1.1, 0.1):
                lmbda = round(lambda_value, 1)

                # Execute the surrogate, measure and rank sum
                final_reco_df = items_to_pandas(
                    dict(
                        surrogate_submodular(user_model_genres_distr_df, candidates_items_mapping, transaction_mean,
                                             config, lmbda=lmbda)
                    )
                )
                final_reco_df.sort_values(by=[ORDER_LABEL], ascending=True, inplace=True)

                # Evaluation metrics applied to the final recommendation list
                ace_result = ace(user_model_genres_distr_df, final_reco_df, candidates_items_mapping)
                ap_result = average_precision(final_reco_df, user_expected_items_ids)
                rr_result = mrr(final_reco_df, user_expected_items_ids)
                mc_value = mc(user_model_genres_distr_df, final_reco_df, candidates_items_mapping, config)

                # Concat all metrics results
                evaluation_results_df = pd.concat([evaluation_results_df,
                                                   pd.DataFrame([
                                                       [recommender_label,
                                                        calib_method,
                                                        distance_metric,
                                                        FIXED_LABEL,
                                                        lmbda,
                                                        MACE_LABEL,
                                                        ace_result],
                                                       [recommender_label,
                                                        calib_method,
                                                        distance_metric,
                                                        FIXED_LABEL,
                                                        lmbda,
                                                        MAP_LABEL,
                                                        ap_result],
                                                       [recommender_label,
                                                        calib_method,
                                                        distance_metric,
                                                        FIXED_LABEL,
                                                        lmbda,
                                                        MRR_LABEL,
                                                        rr_result],
                                                       [recommender_label,
                                                        calib_method,
                                                        distance_metric,
                                                        FIXED_LABEL,
                                                        lmbda,
                                                        MC_LABEL,
                                                        mc_value]
                                                   ],
                                                       columns=EVALUATION_LIST_LABELS
                                                   )])
            # Personalized lambda
            for trade_off, lambda_value in zip([COUNT_GENRES_TRADE_OFF_LABEL, VARIANCE_TRADE_OFF_LABEL],
                                               ['CGR', 'VAR']):
                config[TRADE_OFF_LABEL] = trade_off

                # Execute the surrogate, trade off, measure and rank sum
                lmbda = personalized_trade_off(user_model_genres_distr_df, config)
                final_reco_df = items_to_pandas(
                    dict(
                        surrogate_submodular(user_model_genres_distr_df, candidates_items_mapping, transaction_mean,
                                             config, lmbda=lmbda)
                    )
                )

                # Evaluation metrics applied to the final recommendation list
                ace_result = ace(user_model_genres_distr_df, final_reco_df, candidates_items_mapping)
                map_result = average_precision(final_reco_df, user_expected_items_ids)
                mrr_result = mrr(final_reco_df, user_expected_items_ids)
                mc_value = mc(user_model_genres_distr_df, final_reco_df, candidates_items_mapping, config)

                # Concat all metrics results
                evaluation_results_df = pd.concat([evaluation_results_df,
                                                   pd.DataFrame([
                                                       [recommender_label,
                                                        calib_method,
                                                        distance_metric,
                                                        FIXED_LABEL,
                                                        lambda_value,
                                                        MACE_LABEL,
                                                        ace_result],
                                                       [recommender_label,
                                                        calib_method,
                                                        distance_metric,
                                                        FIXED_LABEL,
                                                        lambda_value,
                                                        MAP_LABEL,
                                                        map_result],
                                                       [recommender_label,
                                                        calib_method,
                                                        distance_metric,
                                                        FIXED_LABEL,
                                                        lambda_value,
                                                        MRR_LABEL,
                                                        mrr_result],
                                                       [recommender_label,
                                                        calib_method,
                                                        distance_metric,
                                                        FIXED_LABEL,
                                                        lambda_value,
                                                        MC_LABEL,
                                                        mc_value]
                                                   ],
                                                       columns=EVALUATION_LIST_LABELS
                                                   )])
    return evaluation_results_df
