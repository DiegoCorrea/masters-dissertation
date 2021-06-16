import os
from copy import deepcopy

import matplotlib

from src.config.labels import EVALUATION_METRIC_LABEL, FAIRNESS_METRIC_LABEL, ALGORITHM_LABEL, \
    LAMBDA_VALUE_LABEL, EVALUATION_VALUE_LABEL, \
    MAP_LABEL, MC_LABEL, MACE_LABEL, CALIBRATION_LABEL, LINEAR_CALIBRATION_LABEL, LOGARITHMIC_CALIBRATION_LABEL, \
    POPULARITY_LABEL, BEST_SCORE_LABEL
from src.config.variables import DPI_VALUE, QUALITY_VALUE, markers_list, line_style_list, FONT_SIZE_VALUE, \
    linear_line_style_list, log_line_style_list
from src.config.path_dir_files import post_graphics_results_path

import matplotlib.pyplot as plt
matplotlib.style.use('default')


def evaluation_linear_fairness_by_algo_over_lambda(evaluation_results_df, k, db):
    if k == 0:
        save_dir = post_graphics_results_path(db) + 'all/'
    else:
        save_dir = post_graphics_results_path(db) + '/' + str(k) + '/'
    for metric in evaluation_results_df[EVALUATION_METRIC_LABEL].unique().tolist():
        evaluation_subset_df = evaluation_results_df[evaluation_results_df[EVALUATION_METRIC_LABEL] == metric]
        for recommender in evaluation_subset_df[ALGORITHM_LABEL].unique().tolist():
            recommender_subset_df = evaluation_subset_df[evaluation_subset_df[ALGORITHM_LABEL] == recommender]
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.grid(True)
            plt.grid(True)
            plt.rc('xtick', labelsize=16)
            plt.rc('ytick', labelsize=16)
            plt.xlabel("Peso", fontsize=18)
            lambda_values = [str(x) for x in recommender_subset_df[LAMBDA_VALUE_LABEL].unique().tolist()]
            plt.xticks(range(0, len(lambda_values)), lambda_values)
            plt.ylabel(metric, fontsize=18)
            fairness_measures = recommender_subset_df[FAIRNESS_METRIC_LABEL].unique().tolist()
            n = len(fairness_measures)
            for distance_metric, m, l in zip(fairness_measures, markers_list[:n], linear_line_style_list[:n]):
                distance_subset_df = recommender_subset_df[
                    (recommender_subset_df[FAIRNESS_METRIC_LABEL] == distance_metric) &
                    (recommender_subset_df[CALIBRATION_LABEL] == LINEAR_CALIBRATION_LABEL)]
                plt.plot([str(x) for x in distance_subset_df[LAMBDA_VALUE_LABEL].tolist()],
                         distance_subset_df[EVALUATION_VALUE_LABEL].tolist(), alpha=0.5, linestyle=l, marker=m,
                         label=str(LINEAR_CALIBRATION_LABEL + '_' + distance_metric), linewidth=4)
            for distance_metric, m, l in zip(fairness_measures, markers_list[:n], log_line_style_list[:n]):
                distance_subset_df = recommender_subset_df[
                    (recommender_subset_df[FAIRNESS_METRIC_LABEL] == distance_metric) &
                    (recommender_subset_df[CALIBRATION_LABEL] == LOGARITHMIC_CALIBRATION_LABEL)]
                plt.plot([str(x) for x in distance_subset_df[LAMBDA_VALUE_LABEL].tolist()],
                         distance_subset_df[EVALUATION_VALUE_LABEL].tolist(), alpha=0.5, linestyle=l, marker=m,
                         label=str(LOGARITHMIC_CALIBRATION_LABEL + '_' + distance_metric), linewidth=4)
            lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3, prop={'size': 18})
            plt.xticks(rotation=30)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(
                save_dir
                + metric
                + '_'
                + recommender
                + '.png',
                format='png',
                dpi=DPI_VALUE,
                quality=QUALITY_VALUE,
                bbox_extra_artists=(lgd,),
                bbox_inches='tight'
            )
            fig.clear()
            plt.close('all')


def evaluation_map_by_mrmc(results_df, k, db):
    if k == 0:
        save_dir = post_graphics_results_path(db) + 'all/'
    else:
        save_dir = post_graphics_results_path(db) + '/' + str(k) + '/'
    for calib_method in results_df[CALIBRATION_LABEL].unique().tolist():
        evaluation_results_df = results_df[results_df[CALIBRATION_LABEL] == calib_method]
        for distance_metric in evaluation_results_df[FAIRNESS_METRIC_LABEL].unique().tolist():
            map_subset_df = evaluation_results_df[
                (evaluation_results_df[FAIRNESS_METRIC_LABEL] == distance_metric) & (evaluation_results_df[
                                                                                         EVALUATION_METRIC_LABEL] == MAP_LABEL)]
            mc_subset_df = evaluation_results_df[
                (evaluation_results_df[FAIRNESS_METRIC_LABEL] == distance_metric) & (
                        evaluation_results_df[EVALUATION_METRIC_LABEL] == MC_LABEL)]
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.grid(True)
            plt.grid(True)
            plt.rc('xtick', labelsize=16)
            plt.rc('ytick', labelsize=16)
            plt.xlabel(MAP_LABEL, fontsize=18)
            plt.ylabel(MC_LABEL, fontsize=18)
            algorithm_list = evaluation_results_df[ALGORITHM_LABEL].unique().tolist()
            n = len(algorithm_list)
            for algorithm, m, l in zip(algorithm_list, markers_list[:n], line_style_list[:n]):
                if algorithm == POPULARITY_LABEL or algorithm == BEST_SCORE_LABEL:
                    continue
                algorithm_map_subset_df = deepcopy(map_subset_df[
                                                       map_subset_df[ALGORITHM_LABEL] == algorithm])
                algorihm_mc_subset_df = deepcopy(mc_subset_df[
                                                     mc_subset_df[ALGORITHM_LABEL] == algorithm])
                algorithm_map_subset_df[LAMBDA_VALUE_LABEL] = algorithm_map_subset_df[LAMBDA_VALUE_LABEL].astype('category')
                algorithm_map_subset_df.sort_values(by=[LAMBDA_VALUE_LABEL], inplace=True)
                algorihm_mc_subset_df[LAMBDA_VALUE_LABEL] = algorihm_mc_subset_df[LAMBDA_VALUE_LABEL].astype('category')
                algorihm_mc_subset_df.sort_values(by=[LAMBDA_VALUE_LABEL], inplace=True)
                plt.plot(algorithm_map_subset_df[EVALUATION_VALUE_LABEL].tolist(),
                         algorihm_mc_subset_df[EVALUATION_VALUE_LABEL].tolist(), alpha=0.5, linestyle=l, marker=m,
                         label=algorithm, linewidth=4)
            lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=4, prop={'size': 18})
            plt.xticks(rotation=30)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = "_".join([calib_method, MAP_LABEL, MC_LABEL, distance_metric])
            plt.savefig(
                save_dir
                + file_name
                + '.png',
                format='png',
                dpi=DPI_VALUE,
                quality=QUALITY_VALUE,
                bbox_extra_artists=(lgd,),
                bbox_inches='tight'
            )
            plt.close('all')


def evaluation_map_by_mace(results_df, k, db):
    if k == 0:
        save_dir = post_graphics_results_path(db) + 'all/'
    else:
        save_dir = post_graphics_results_path(db) + '/' + str(k) + '/'
    for calib_method in results_df[CALIBRATION_LABEL].unique().tolist():
        evaluation_results_df = results_df[results_df[CALIBRATION_LABEL] == calib_method]
        for distance_metric in evaluation_results_df[FAIRNESS_METRIC_LABEL].unique().tolist():
            map_subset_df = evaluation_results_df[
                (evaluation_results_df[FAIRNESS_METRIC_LABEL] == distance_metric) & (evaluation_results_df[
                                                                                         EVALUATION_METRIC_LABEL] == MAP_LABEL)]
            mc_subset_df = evaluation_results_df[
                (evaluation_results_df[FAIRNESS_METRIC_LABEL] == distance_metric) & (
                        evaluation_results_df[EVALUATION_METRIC_LABEL] == MACE_LABEL)]
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.grid(True)
            plt.grid(True)
            plt.rc('xtick', labelsize=16)
            plt.rc('ytick', labelsize=16)
            plt.xlabel(MAP_LABEL, fontsize=18)
            plt.ylabel(MACE_LABEL, fontsize=18)
            algorithm_list = evaluation_results_df[ALGORITHM_LABEL].unique().tolist()
            n = len(algorithm_list)
            for algorithm, m, l in zip(algorithm_list, markers_list[:n], line_style_list[:n]):
                if algorithm == POPULARITY_LABEL or algorithm == BEST_SCORE_LABEL:
                    continue
                algorithm_map_subset_df = deepcopy(map_subset_df[
                                                       map_subset_df[ALGORITHM_LABEL] == algorithm])
                algorihm_mc_subset_df = deepcopy(mc_subset_df[
                                                     mc_subset_df[ALGORITHM_LABEL] == algorithm])
                algorithm_map_subset_df[LAMBDA_VALUE_LABEL] = algorithm_map_subset_df[LAMBDA_VALUE_LABEL].astype('category')
                algorithm_map_subset_df.sort_values(by=[LAMBDA_VALUE_LABEL], inplace=True)
                algorihm_mc_subset_df[LAMBDA_VALUE_LABEL] = algorihm_mc_subset_df[LAMBDA_VALUE_LABEL].astype('category')
                algorihm_mc_subset_df.sort_values(by=[LAMBDA_VALUE_LABEL], inplace=True)
                plt.plot(algorithm_map_subset_df[EVALUATION_VALUE_LABEL].tolist(),
                         algorihm_mc_subset_df[EVALUATION_VALUE_LABEL].tolist(), alpha=0.5, linestyle=l, marker=m,
                         label=algorithm, linewidth=4)
            lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=4, prop={'size': 18})
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = "_".join([calib_method, MAP_LABEL, MACE_LABEL, distance_metric])
            plt.savefig(
                save_dir
                + file_name
                + '.png',
                format='png',
                dpi=DPI_VALUE,
                quality=QUALITY_VALUE,
                bbox_extra_artists=(lgd,),
                bbox_inches='tight'
            )
            plt.close('all')
