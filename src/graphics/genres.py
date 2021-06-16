import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
# matplotlib.style.use('default')

from src.config.variables import FONT_SIZE_VALUE, \
    QUALITY_VALUE, DPI_VALUE, scatter_bubble_color
from src.config.path_dir_files import pos_processing_results_path, pre_processing_to_use_path
from src.config.labels import NUMBER_OF_GENRES, USER_MODEL_SIZE_LABEL
from src.config.language_strings import LANGUAGE_NUMBER_GENRES, \
    LANGUAGE_USER_PROFILE_SIZE


# ######################### #
#     Data set graphics     #
# ######################### #
def user_model_size_by_number_of_genres(user_profile_df, db=0):
    x_data = user_profile_df[USER_MODEL_SIZE_LABEL].tolist()
    y_data = user_profile_df[NUMBER_OF_GENRES].tolist()
    plt.figure()
    plt.grid(True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xlabel(LANGUAGE_USER_PROFILE_SIZE, fontsize=18)
    plt.ylabel(LANGUAGE_NUMBER_GENRES, fontsize=18)
    plt.scatter(x_data,
                y_data,
                alpha=0.5, c=scatter_bubble_color)
    list_number = list(set(y_data))
    plt.yticks(range(min(list_number), max(list_number) + 1))
    plt.xticks(rotation=30)
    # plt.legend()
    path_to_save = pre_processing_to_use_path(db)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    plt.savefig(
        path_to_save
        + 'user_model_size_by_number_of_genres'
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE
    )
    plt.close('all')


def genre_distribution_bar(genre_distr_df, db, distr_type):
    # genre_distr_df = genre_distr_df.reindex(sorted(genre_distr_df.columns), axis=1)
    x = genre_distr_df.columns.tolist()
    y = genre_distr_df.mean(axis=0).tolist()
    std = genre_distr_df.sem(axis=0).tolist()
    plt.figure()
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.bar(x, y, color=scatter_bubble_color, yerr=std)
    # Turn on the grid
    plt.xticks(rotation=90)
    path_to_save = pre_processing_to_use_path(db)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    plt.savefig(
        path_to_save
        + 'genre_distribution_bar_' + str(distr_type)
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE,
        bbox_inches='tight'
    )
    plt.close('all')


def compare_genre_distribution_bar(users_genre_distr_df, items_genre_distr_df, db=0):
    users_genre_distr_df = users_genre_distr_df.reindex(sorted(users_genre_distr_df.columns), axis=1)
    items_genre_distr_df = items_genre_distr_df.reindex(sorted(items_genre_distr_df.columns), axis=1)
    women_means = users_genre_distr_df.mean(axis=0).tolist()
    men_means = items_genre_distr_df.mean(axis=0).tolist()
    users_std = users_genre_distr_df.sem(axis=0).tolist()
    items_std = items_genre_distr_df.sem(axis=0).tolist()
    labels = users_genre_distr_df.columns.tolist()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    rects1 = ax.bar(x - width / 2, men_means, width, label='Itens', yerr=items_std)
    rects2 = ax.bar(x + width / 2, women_means, width, label='Usuários', yerr=users_std)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Probabilidade', fontsize=FONT_SIZE_VALUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.xticks(rotation=90)
    path_to_save = pre_processing_to_use_path(db)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    plt.savefig(
        path_to_save
        + 'compare_genre_distribution_bar'
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE,
        bbox_inches='tight'
    )
    plt.close('all')


# ############################### #
#     Postprocessing graphics     #
# ############################### #
def compare_distributions(distr_list, file_name, title, db=0):
    concated_distr = pd.DataFrame()
    for df in distr_list:
        genre_distr_df = df.reindex(sorted(df.columns), axis=1)
        x = genre_distr_df.columns.tolist()
        y = genre_distr_df.mean(axis=0).tolist()
        std = genre_distr_df.std(axis=0).tolist()
    distribution_path = pos_processing_results_path(db) + "genres/"
    plt.figure()
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    concated_distr.plot(kind='bar')
    # Turn on the grid
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title(title)
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
                     ncol=3, fancybox=True, shadow=True)
    plt.xticks(rotation=30)
    if not os.path.exists(distribution_path):
        os.makedirs(distribution_path)
    plt.savefig(
        distribution_path
        + file_name
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE,
        bbox_inches='tight'
    )
    plt.close('all')
