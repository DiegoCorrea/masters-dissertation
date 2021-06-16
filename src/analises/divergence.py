import numpy as np
import pandas as pd

from src.config.variables import RECOMMENDATION_LIST_SIZE, CHI_LABEL, HE_LABEL
from src.config.language_strings import LANGUAGE_AT_LIST
from src.posprocessing.distance_measures import get_person_chi_square_values, get_hellinger_values, \
    get_kullback_leibler_values
from src.posprocessing.distributions import get_distribution
from src.posprocessing.filter import get_top_n


def calibrated_divergence_compute(trainset_df, calibrated_reco_structure, item_mapping, algorithm_label):
    print("Inicando ....")
    preference_distribution_df = get_distribution(trainset_df, item_mapping)
    results_df = pd.DataFrame()
    for metric_label in calibrated_reco_structure:
        for trade_off_df in calibrated_reco_structure[metric_label]:
            at_all_df = pd.DataFrame()
            for i in range(1, RECOMMENDATION_LIST_SIZE + 1):
                print(LANGUAGE_AT_LIST + str(i))
                recommendation_distribution_df = get_distribution(get_top_n(trade_off_df, n=i), item_mapping)
                if metric_label == CHI_LABEL:
                    chi_square_values_df = get_person_chi_square_values(preference_distribution_df,
                                                                        recommendation_distribution_df, i)
                    at_all_df = pd.concat([at_all_df, chi_square_values_df], axis=1)
                elif metric_label == HE_LABEL:
                    hellinger_values_df = get_hellinger_values(preference_distribution_df,
                                                               recommendation_distribution_df, i)
                    at_all_df = pd.concat([at_all_df, hellinger_values_df], axis=1)
                else:
                    kl_values_df = get_kullback_leibler_values(preference_distribution_df,
                                                               recommendation_distribution_df, i)
                    at_all_df = pd.concat([at_all_df, kl_values_df], axis=1)
            data_list = [at_all_df[col].mean() for col in at_all_df.columns]
            a = pd.DataFrame(data=[data_list], columns=list(range(1, RECOMMENDATION_LIST_SIZE + 1)),
                             index=[np.array([algorithm_label]), np.array([metric_label]),
                                    np.array([trade_off_df.name])])
            results_df = pd.concat([results_df, a], sort=False)
            print(results_df)
            print("Finalizando...")
    return results_df
