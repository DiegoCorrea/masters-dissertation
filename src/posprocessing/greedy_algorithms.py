from copy import deepcopy

import numpy as np

from src.config.labels import CALIBRATION_LABEL, LINEAR_CALIBRATION_LABEL
from src.config.variables import RECOMMENDATION_LIST_SIZE
from src.posprocessing.calibrated_methods import linear_calibration, log_calibration


# ################################################################# #
# ############### Surrogate Submodular Optimization ############### #
# ################################################################# #
def surrogate_submodular(user_model_genres_distr_df, candidates_items_mapping, transaction_mean, config,
                         n=RECOMMENDATION_LIST_SIZE, lmbda=0.5):
    """
    start with an empty recommendation list,
    loop over the topn cardinality, during each iteration
    update the list with the item that maximizes the utility function.
    """
    calib_reco_dict = dict()
    for i in range(n):
        max_utility = -999999999999999999999999
        best_item = None
        best_id = None
        best_bias_list = list()
        for i_id, item in candidates_items_mapping.items():
            if i_id not in calib_reco_dict.keys() and i_id is not None:
                utility = -999999999999999999999999
                temp_reco_items = deepcopy(calib_reco_dict)
                temp_reco_items[i_id] = item
                bias_list = deepcopy(best_bias_list)

                if config[CALIBRATION_LABEL] == LINEAR_CALIBRATION_LABEL:
                    utility = linear_calibration(temp_reco_items, user_model_genres_distr_df, config, lmbda)
                else:
                    utility, bias_list = log_calibration(temp_reco_items, user_model_genres_distr_df, transaction_mean,
                                                         bias_list, i_id, config, lmbda)

                if utility > max_utility:
                    max_utility = deepcopy(utility)
                    best_item = deepcopy(item)
                    best_id = deepcopy(i_id)
                    best_bias_list = deepcopy(bias_list)
        calib_reco_dict[best_id] = best_item
    return calib_reco_dict
