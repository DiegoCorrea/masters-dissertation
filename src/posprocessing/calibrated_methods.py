import math

import numpy as np

from src.config.labels import HE_LABEL, CHI_LABEL, FAIRNESS_METRIC_LABEL
from src.posprocessing.bias import calculating_user_bias
from src.posprocessing.distance_measures import compute_kullback_leibler, compute_person_chi_square, compute_hellinger
from src.posprocessing.distributions import compute_genre_distr_with_weigth


# ################################################################# #
# ###################### Linear Calibration ####################### #
# ################################################################# #
def linear_calibration(temp_reco_items, user_model_genres_distr_df, config, lmbda=0.5):
    """
    Our objective function for computing the utility score for
    the list of recommended items.

    lmbda : float, 0.0 ~ 1.0, default 0.5
        Lambda term controls the score and calibration tradeoff,
        the higher the lambda the higher the resulting recommendation
        will be calibrated. Lambda is keyword in Python, so it's
        lmbda instead ^^
    """
    reco_distr = compute_genre_distr_with_weigth(temp_reco_items)

    div_value = 0.0
    if config[FAIRNESS_METRIC_LABEL] == CHI_LABEL:
        div_value = compute_person_chi_square(user_model_genres_distr_df, reco_distr)
    elif config[FAIRNESS_METRIC_LABEL] == HE_LABEL:
        div_value = compute_hellinger(user_model_genres_distr_df, reco_distr)
    else:
        div_value = compute_kullback_leibler(user_model_genres_distr_df, reco_distr)

    total_score = 0.0
    for i_id, item in temp_reco_items.items():
        total_score += item.score

    # the higher the better so remember to negate it in the calculation
    utility = (1 - lmbda) * total_score - lmbda * div_value
    return utility


def log_calibration(temp_reco_items, user_model_genres_distr_df, transaction_mean,
                    bias_list, i_id, config, lmbda=0.5):
    """
    Our objective function for computing the utility score for
    the list of recommended items.

    lmbda : float, 0.0 ~ 1.0, default 0.5
        Lambda term controls the score and calibration tradeoff,
        the higher the lambda the higher the resulting recommendation
        will be calibrated. Lambda is keyword in Python, so it's
        lmbda instead ^^
    """
    reco_distr = compute_genre_distr_with_weigth(temp_reco_items)

    div_value = 0.0
    if config[FAIRNESS_METRIC_LABEL] == CHI_LABEL:
        div_value = compute_person_chi_square(user_model_genres_distr_df, reco_distr)
    elif config[FAIRNESS_METRIC_LABEL] == HE_LABEL:
        div_value = compute_hellinger(user_model_genres_distr_df, reco_distr)
    else:
        div_value = compute_kullback_leibler(user_model_genres_distr_df, reco_distr)

    total_score = 0.0
    for i_id, item in temp_reco_items.items():
        total_score += item.score

    # the higher the better so remember to negate it in the calculation
    fina_value = (1 - lmbda) * total_score - lmbda * div_value
    user_bias, bias_list = calculating_user_bias(temp_reco_items, transaction_mean, bias_list, i_id)
    utility = np.sign(fina_value) * math.log2(abs(fina_value) + 1) + user_bias
    return utility, bias_list
