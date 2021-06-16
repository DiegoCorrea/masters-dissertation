import math

import numpy as np
import pandas as pd

from src.config.variables import RECOMMENDATION_LIST_SIZE, ALPHA_VALUE
from src.config.labels import AT_LABEL
from src.posprocessing.distributions import get_distribution

LOCAL_PREFERENCE_DISTRIBUTION_DF = pd.DataFrame()
LOCAL_RECOMMENDATION_DISTRIBUTION_DF = pd.DataFrame()


# ####################################################### #
# # # # # # # # # #  Kullback-Leibler # # # # # # # # # # #
# ####################################################### #
def compute_kullback_leibler(interacted_distr, reco_distr, alpha=ALPHA_VALUE):
    """
    Kullback-Leibler (p || q), the lower the better.

    alpha is not really a tuning parameter, it's just there to make the
    computation more numerically stable.
    """
    kl_div = 0.
    for genre, p in interacted_distr.items():
        q = reco_distr.get(genre, 0.)
        til_q = (1 - alpha) * q + alpha * p

        if p == 0.0 or til_q == 0.0:
            kl_div = kl_div
        else:
            kl_div = kl_div + (p * np.log2(p / til_q))
    return kl_div


def map_compute_kullback_leibler(user_id, n):
    interacted_distr = LOCAL_PREFERENCE_DISTRIBUTION_DF.loc[user_id]
    reco_distr = LOCAL_RECOMMENDATION_DISTRIBUTION_DF.loc[user_id]
    user_value = compute_kullback_leibler(interacted_distr.to_dict(), reco_distr.to_dict())
    return pd.DataFrame(data=[[user_value]], columns=[AT_LABEL + str(n)], index=[user_id])


def get_kullback_leibler_values(preference_distribution_df, recommendation_distribution_df, n=RECOMMENDATION_LIST_SIZE):
    """
    Kullback-Leibler (p || q)
        :return A Pandas DataFrame with m lines (distribution lines) and n columns
    """
    global LOCAL_PREFERENCE_DISTRIBUTION_DF
    global LOCAL_RECOMMENDATION_DISTRIBUTION_DF
    LOCAL_PREFERENCE_DISTRIBUTION_DF = preference_distribution_df
    LOCAL_RECOMMENDATION_DISTRIBUTION_DF = recommendation_distribution_df
    users_ids = preference_distribution_df.index.values.tolist()
    map_results_df = [map_compute_kullback_leibler(uid, n) for uid in users_ids]
    distance_results_df = pd.concat(map_results_df, sort=False)
    return distance_results_df


# ###################################################### #
# # # # # # # # # #  Hellinger # # # # # # # # # # # # # #
# ###################################################### #
def compute_hellinger(interacted_distr, reco_distr):
    """
    Hellinger (p || q), the lower the better.
    """
    sum_value = 0.
    for genre, p in interacted_distr.items():
        q = reco_distr.get(genre, 0.)
        sum_value += np.power(np.sqrt(p) - np.sqrt(q), 2.0)
    hl_value = np.sqrt(sum_value) / np.sqrt(2)
    return hl_value


def map_compute_hellinger(user_id, n):
    interacted_distr = LOCAL_PREFERENCE_DISTRIBUTION_DF.loc[user_id]
    reco_distr = LOCAL_RECOMMENDATION_DISTRIBUTION_DF.loc[user_id]
    user_value = compute_hellinger(interacted_distr.to_dict(), reco_distr.to_dict())
    return pd.DataFrame(data=[[user_value]], columns=[AT_LABEL + str(n)], index=[user_id])


def get_hellinger_values(preference_distribution_df, recommendation_distribution_df, n=RECOMMENDATION_LIST_SIZE):
    """
    Hellinger (p,q)
        :return A Pandas DataFrame with m lines (distribution lines) and n columns
    """
    global LOCAL_PREFERENCE_DISTRIBUTION_DF
    global LOCAL_RECOMMENDATION_DISTRIBUTION_DF
    LOCAL_PREFERENCE_DISTRIBUTION_DF = preference_distribution_df
    LOCAL_RECOMMENDATION_DISTRIBUTION_DF = recommendation_distribution_df
    users_ids = preference_distribution_df.index.values.tolist()
    map_results_df = [map_compute_hellinger(uid, n) for uid in users_ids]
    distance_results_df = pd.concat(map_results_df, sort=False)
    return distance_results_df


# ####################################################### #
# # # # # # # # #  Pearson Chi-Square # # # # # # # # # # #
# ####################################################### #
def compute_person_chi_square(interacted_distr, reco_distr, alpha=ALPHA_VALUE):
    """
    Pearson Chi-Square (p,q), the lower the better.
    """
    sum_value = 0.
    for genre, p in interacted_distr.items():
        q = reco_distr.get(genre, 0.)
        til_q = (1 - alpha) * q + alpha * p
        if math.isnan(til_q) or til_q == 0.0:
            temp_value = 0.0
        else:
            temp_value = np.power((p - til_q), 2.0) / til_q
        if math.isnan(temp_value):
            temp_value = 0.0
        sum_value += temp_value
    return sum_value


def map_compute_person_chi_square(user_id, n):
    interacted_distr = LOCAL_PREFERENCE_DISTRIBUTION_DF.loc[user_id]
    reco_distr = LOCAL_RECOMMENDATION_DISTRIBUTION_DF.loc[user_id]
    user_value = compute_person_chi_square(interacted_distr.to_dict(), reco_distr.to_dict())
    return pd.DataFrame(data=[[user_value]], columns=[AT_LABEL + str(n)], index=[user_id])


def get_person_chi_square_values(preference_distribution_df, recommendation_distribution_df,
                                 n=RECOMMENDATION_LIST_SIZE):
    """
    Pearson Chi-Square (p,q)
        :return A Pandas DataFrame with m lines (distribution lines) and n columns
    """
    global LOCAL_PREFERENCE_DISTRIBUTION_DF
    global LOCAL_RECOMMENDATION_DISTRIBUTION_DF
    LOCAL_PREFERENCE_DISTRIBUTION_DF = preference_distribution_df
    LOCAL_RECOMMENDATION_DISTRIBUTION_DF = recommendation_distribution_df
    users_ids = preference_distribution_df.index.values.tolist()
    map_results_df = [map_compute_person_chi_square(uid, n) for uid in users_ids]
    distance_results_df = pd.concat(map_results_df, sort=False)
    return distance_results_df


# ####################################################### #
# # # # # #  Processing all distance measures # # # # # # #
# ####################################################### #
def distance_measures_processing(trainset_df, top_n_df, item_mapping, n):
    preference_distribution_df = get_distribution(trainset_df, item_mapping)
    recommendation_distribution_df = get_distribution(top_n_df, item_mapping)
    kl_values_df = get_kullback_leibler_values(preference_distribution_df, recommendation_distribution_df, n)
    hellinger_values_df = get_hellinger_values(preference_distribution_df, recommendation_distribution_df, n)
    chi_square_values_df = get_person_chi_square_values(preference_distribution_df, recommendation_distribution_df, n)
    return kl_values_df, hellinger_values_df, chi_square_values_df
