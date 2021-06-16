import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from src.config.variables import FONT_SIZE_VALUE, baselines_results_path, DPI_VALUE, QUALITY_VALUE, RECOMMENDATION_LIST_SIZE, \
    markers_list, line_style_list, postprocessing_results_path
from src.config.language_strings import LANGUAGE_DISTANCE_VALUE, LANGUAGE_TOP_N_ITEMS


# ########################## #
#     Baselines graphics     #
# ########################## #


def processing_distance(df_list, algorithm):
    list_range = range(1, RECOMMENDATION_LIST_SIZE + 1)
    plt.figure()
    plt.grid(True)
    plt.xlabel(LANGUAGE_TOP_N_ITEMS, fontsize=FONT_SIZE_VALUE)
    plt.xticks(list_range)
    plt.ylabel(LANGUAGE_DISTANCE_VALUE, fontsize=FONT_SIZE_VALUE)
    for df, m, l in zip(df_list, markers_list[:RECOMMENDATION_LIST_SIZE], line_style_list[:RECOMMENDATION_LIST_SIZE]):
        plt.plot(list_range, [df[col].mean() for col in df.columns], alpha=0.5, linestyle=l, marker=m, label=df.name)
    plt.legend(loc='best', borderaxespad=0.)
    if not os.path.exists(baselines_results_path):
        os.makedirs(baselines_results_path)
    plt.savefig(
        baselines_results_path
        + algorithm
        + '_'
        + 'processing_distance'
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE,
        bbox_inches='tight'
    )
    plt.close('all')


def algorithms_distance(df_list, measure):
    list_range = range(1, RECOMMENDATION_LIST_SIZE + 1)
    plt.figure()
    plt.grid(True)
    plt.xlabel(LANGUAGE_TOP_N_ITEMS, fontsize=FONT_SIZE_VALUE)
    plt.xticks(list_range)
    plt.ylabel(LANGUAGE_DISTANCE_VALUE, fontsize=FONT_SIZE_VALUE)
    for df, m, l in zip(df_list, markers_list[:RECOMMENDATION_LIST_SIZE], line_style_list[:RECOMMENDATION_LIST_SIZE]):
        plt.plot(list_range, [df[col].mean() for col in df.columns], alpha=0.5, linestyle=l, marker=m, label=df.name)
    plt.legend(loc='best', borderaxespad=0.)
    if not os.path.exists(baselines_results_path):
        os.makedirs(baselines_results_path)
    plt.savefig(
        baselines_results_path
        + 'algorithms_distance_'
        + measure
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE,
        bbox_inches='tight'
    )
    plt.close('all')


def calibrated_algorithms_distance(calib_divergence_df):
    postprocessing_path = postprocessing_results_path + 'distance/'
    list_range = range(1, RECOMMENDATION_LIST_SIZE + 1)
    if not os.path.exists(postprocessing_path):
        os.makedirs(postprocessing_path)
    for lvl_0 in set(calib_divergence_df.index.get_level_values(0)):
        for lvl_1 in set(calib_divergence_df.index.get_level_values(1)):
            plt.figure()
            plt.grid(True)
            plt.xlabel(LANGUAGE_TOP_N_ITEMS, fontsize=FONT_SIZE_VALUE)
            plt.xticks(list_range)
            plt.ylabel(LANGUAGE_DISTANCE_VALUE, fontsize=FONT_SIZE_VALUE)
            plt.title(lvl_0 + " " + lvl_1)
            set_lvl_2 = set(calib_divergence_df.index.get_level_values(2))
            length_lvl_2 = len(set_lvl_2)
            for lvl_2, m, l in zip(set_lvl_2, markers_list[:length_lvl_2], line_style_list[:length_lvl_2]):
                line = calib_divergence_df.loc[(lvl_0, lvl_1, lvl_2)]
                plt.plot(list_range, line.tolist(), alpha=0.5, linestyle=l, marker=m,
                         label=lvl_2)
            plt.legend(loc='best', borderaxespad=0.)
            plt.savefig(
                postprocessing_path
                + lvl_0
                + '_'
                + lvl_1
                + '.png',
                format='png',
                dpi=DPI_VALUE,
                quality=QUALITY_VALUE,
                bbox_inches='tight'
            )
            plt.close('all')
    for lvl_2 in set(calib_divergence_df.index.get_level_values(2)):
        for lvl_1 in set(calib_divergence_df.index.get_level_values(1)):
            plt.figure()
            plt.grid(True)
            plt.xlabel(LANGUAGE_TOP_N_ITEMS, fontsize=FONT_SIZE_VALUE)
            plt.xticks(list_range)
            plt.ylabel(LANGUAGE_DISTANCE_VALUE, fontsize=FONT_SIZE_VALUE)
            plt.title(lvl_1 + " " + lvl_2)
            set_lvl_0 = set(calib_divergence_df.index.get_level_values(0))
            length_lvl_0 = len(set_lvl_0)
            for lvl_0, m, l in zip(set_lvl_0, markers_list[:length_lvl_0], line_style_list[:length_lvl_0]):
                line = calib_divergence_df.loc[(lvl_0, lvl_1, lvl_2)]
                plt.plot(list_range, line.tolist(), alpha=0.5, linestyle=l, marker=m,
                         label=lvl_0)
            plt.legend(loc='best', borderaxespad=0.)
            plt.savefig(
                postprocessing_path
                + lvl_1
                + '_'
                + lvl_2
                + '.png',
                format='png',
                dpi=DPI_VALUE,
                quality=QUALITY_VALUE,
                bbox_inches='tight'
            )
            plt.close('all')
