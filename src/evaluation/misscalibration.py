import pandas as pd

from src.config.labels import ORDER_LABEL, FAIRNESS_METRIC_LABEL, CHI_LABEL, HE_LABEL
from src.posprocessing.distance_measures import compute_person_chi_square, compute_hellinger, compute_kullback_leibler
from src.posprocessing.distributions import user_get_distribution


def mc(interacted_distr, reco_items_df, item_mapping, config):
    order_list = reco_items_df[ORDER_LABEL].tolist()
    mc_value = 0.0
    for i in order_list:
        reco_distr = user_get_distribution(reco_items_df[:i], item_mapping)
        norm_dist = user_get_distribution(pd.DataFrame(), item_mapping)
        if config[FAIRNESS_METRIC_LABEL] == CHI_LABEL:
            mc_value += compute_person_chi_square(interacted_distr, reco_distr) / compute_person_chi_square(
                interacted_distr, norm_dist)
        elif config[FAIRNESS_METRIC_LABEL] == HE_LABEL:
            mc_value += compute_hellinger(interacted_distr, reco_distr) / compute_hellinger(interacted_distr, norm_dist)
        else:
            mc_value += compute_kullback_leibler(interacted_distr, reco_distr) / compute_kullback_leibler(
                interacted_distr, norm_dist)
    mc_value = mc_value / len(order_list)
    return mc_value
