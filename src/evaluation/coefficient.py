import os

from src.config.labels import CALIBRATION_LABEL, FAIRNESS_METRIC_LABEL, EVALUATION_METRIC_LABEL, MAP_LABEL, MACE_LABEL, \
    ALGORITHM_LABEL, EVALUATION_VALUE_LABEL, MC_LABEL
from src.config.path_dir_files import coefficient_results_path
import pandas as pd


def coefficient(results_df, db):
    save_dir = coefficient_results_path(db) + 'all/'
    for divergence in results_df[FAIRNESS_METRIC_LABEL].unique().tolist():
        divergence_df = results_df[results_df[FAIRNESS_METRIC_LABEL] == divergence]
        for tradeoff in divergence_df[CALIBRATION_LABEL].unique().tolist():
            tradeoff_divergence_df = divergence_df[divergence_df[CALIBRATION_LABEL] == tradeoff]
            for metric in [MACE_LABEL, MC_LABEL]:
                metric_dict = {}
                for algorithm in tradeoff_divergence_df[ALGORITHM_LABEL].unique().tolist():
                    algorithm_tradeoff_divergence_df = tradeoff_divergence_df[
                        tradeoff_divergence_df[ALGORITHM_LABEL] == algorithm]

                    map_df = algorithm_tradeoff_divergence_df[
                        algorithm_tradeoff_divergence_df[EVALUATION_METRIC_LABEL] == MAP_LABEL]
                    metric_df = algorithm_tradeoff_divergence_df[
                        algorithm_tradeoff_divergence_df[EVALUATION_METRIC_LABEL] == metric]
                    map_value = map_df[EVALUATION_VALUE_LABEL].mean()
                    metric_value = metric_df[EVALUATION_VALUE_LABEL].mean()

                    metric_dict[algorithm] = [round(metric_value/map_value, 2)]
                metric_sorted_dict = dict(sorted(metric_dict.items()))
                coe_df = pd.DataFrame.from_dict(metric_sorted_dict)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                file_name = "_".join([metric, MAP_LABEL, divergence, tradeoff, '.csv'])

                coe_df.to_csv(os.path.join(save_dir, file_name), index=False)

