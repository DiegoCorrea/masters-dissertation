import os

import matplotlib

from src.config.path_dir_files import pre_processing_to_use_path

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

from src.config.variables import FONT_SIZE_VALUE, DPI_VALUE, QUALITY_VALUE, scatter_bubble_color, cmap_color_scale
from src.config.labels import NUMBER_OF_MEDIUM_TAIL_ITEMS_LABEL, PERCENTAGE_OF_MEDIUM_TAIL_ITEMS_LABEL, \
    PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL, TOTAL_TIMES_LABEL, NUMBER_OF_SHORT_TAIL_ITEMS_LABEL, USER_MODEL_SIZE_LABEL
from src.config.language_strings import LANGUAGE_IN_USERS_PROFILES, \
    LANGUAGE_NUMBER_ITEMS, LANGUAGE_NUMBER_POPULAR_ITEMS, \
    LANGUAGE_USER_PROFILE_SIZE, LANGUAGE_PERCENTAGE_POPULAR_ITEMS, \
    LANGUAGE_USER, LANGUAGE_NUMBER_UNPOPULAR_ITEMS, LANGUAGE_PERCENTAGE_UNPOPULAR_ITEMS


def user_model_size_by_short_tail_items(user_profile_df, db=None):
    # Ordenação dos dados pelo eixo x
    df = user_profile_df.sort_values(by=[USER_MODEL_SIZE_LABEL], ascending=[True])
    # Obtenção dos dados do eixo x e do eixo y
    x_data = df[USER_MODEL_SIZE_LABEL].tolist()
    y_data = df[NUMBER_OF_SHORT_TAIL_ITEMS_LABEL].tolist()
    z_data = df[PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL].tolist()
    # Figura iniciada e atribuição das labels dos eixos x e y
    matplotlib.use('Agg')
    matplotlib.style.use('ggplot')
    plt.figure()
    plt.grid(True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xlabel(LANGUAGE_USER_PROFILE_SIZE, fontsize=18)
    plt.ylabel(LANGUAGE_NUMBER_POPULAR_ITEMS, fontsize=18)
    # Dados desenhados na figura
    plt.scatter(x=x_data, y=y_data, c=z_data,
                alpha=0.5, label=LANGUAGE_USER, cmap=cmap_color_scale)
    plt.colorbar(label=LANGUAGE_PERCENTAGE_POPULAR_ITEMS)
    # Pasta para salvar a figura
    path_to_save = pre_processing_to_use_path(db)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # Salvar figura no disco
    plt.savefig(
        path_to_save
        + 'user_model_size_by_short_tail_items'
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE,
        bbox_inches='tight'
    )
    # Figura fechada
    plt.close('all')


def user_model_size_by_medium_tail_items(user_profile_df, db=None):
    # Ordenação dos dados pelo eixo x
    df = user_profile_df.sort_values(by=[USER_MODEL_SIZE_LABEL], ascending=[True])
    # Obtenção dos dados do eixo x e do eixo y
    x_data = df[USER_MODEL_SIZE_LABEL].tolist()
    y_data = df[NUMBER_OF_MEDIUM_TAIL_ITEMS_LABEL].tolist()
    z_data = df[PERCENTAGE_OF_MEDIUM_TAIL_ITEMS_LABEL].tolist()
    # Figura iniciada e atribuição das labels dos eixos x e y
    matplotlib.use('Agg')
    matplotlib.style.use('ggplot')
    plt.figure()
    plt.grid(True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xlabel(LANGUAGE_USER_PROFILE_SIZE, fontsize=FONT_SIZE_VALUE)
    plt.ylabel(LANGUAGE_NUMBER_UNPOPULAR_ITEMS, fontsize=FONT_SIZE_VALUE)
    # Dados desenhados na figura
    plt.scatter(x=x_data, y=y_data, c=z_data,
                alpha=0.5, label=LANGUAGE_USER, cmap=cmap_color_scale)
    plt.colorbar(label=LANGUAGE_PERCENTAGE_UNPOPULAR_ITEMS)
    # Pasta para salvar a figura
    path_to_save = pre_processing_to_use_path(db)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # Salvar figura no disco
    plt.savefig(
        path_to_save
        + 'user_model_size_by_medium_tail_items'
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE,
        bbox_inches='tight'
    )
    # Figura fechada
    plt.close('all')


def user_model_size_by_percentage_short_tail_items(user_profile_df, db=None):
    # Ordenação dos dados pelo eixo x
    df = user_profile_df.sort_values(by=[USER_MODEL_SIZE_LABEL], ascending=[True])
    # Obtenção dos dados do eixo x e do eixo y
    x_data = df[USER_MODEL_SIZE_LABEL].tolist()
    y_data = df[PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL].tolist()
    # Figura iniciada e atribuição das labels dos eixos x e y
    matplotlib.use('Agg')
    matplotlib.style.use('ggplot')
    plt.figure()
    plt.grid(True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xlabel(LANGUAGE_USER_PROFILE_SIZE, fontsize=FONT_SIZE_VALUE)
    plt.ylabel(LANGUAGE_PERCENTAGE_POPULAR_ITEMS, fontsize=FONT_SIZE_VALUE)
    # Dados desenhados na figura
    plt.scatter(x_data, y_data, alpha=0.5,
                c=scatter_bubble_color, label=LANGUAGE_USER)
    # Pasta para salvar a figura
    path_to_save = pre_processing_to_use_path(db)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # Salvar figura no disco
    plt.savefig(
        path_to_save
        + 'user_model_size_by_percentage_short_tail_items'
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE,
        bbox_inches='tight'
    )
    # Figura fechada
    plt.close('all')


def user_model_size_by_percentage_medium_tail_items(user_profile_df, db=None):
    # Ordenação dos dados pelo eixo x
    df = user_profile_df.sort_values(by=[USER_MODEL_SIZE_LABEL], ascending=[True])
    # Obtenção dos dados do eixo x e do eixo y
    x_data = df[USER_MODEL_SIZE_LABEL].tolist()
    y_data = df[PERCENTAGE_OF_MEDIUM_TAIL_ITEMS_LABEL].tolist()
    # Figura iniciada e atribuição das labels dos eixos x e y
    matplotlib.use('Agg')
    matplotlib.style.use('ggplot')
    plt.figure()
    plt.grid(True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xlabel(LANGUAGE_USER_PROFILE_SIZE, fontsize=FONT_SIZE_VALUE)
    plt.ylabel(LANGUAGE_PERCENTAGE_UNPOPULAR_ITEMS, fontsize=FONT_SIZE_VALUE)
    # Dados desenhados na figura
    plt.scatter(x_data, y_data, alpha=0.5,
                c=scatter_bubble_color, label=LANGUAGE_USER)
    # Pasta para salvar a figura
    path_to_save = pre_processing_to_use_path(db)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # Salvar figura no disco
    plt.savefig(
        path_to_save
        + 'user_model_size_by_percentage_medium_tail_items'
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE,
        bbox_inches='tight'
    )
    # Figura fechada
    plt.close('all')


def long_tail_graphic(item_popularity_df, db=None):
    # Ordenação dos dados pelo eixo y invertido
    df = item_popularity_df.sort_values(by=[TOTAL_TIMES_LABEL], ascending=[False])
    # Obtenção dos dados do eixo x e do eixo y
    x_data = [i + 1 for i in range(len(df))]
    y_data = df[TOTAL_TIMES_LABEL].tolist()
    # Figura iniciada e atribuição das labels dos eixos x e y
    matplotlib.use('Agg')
    matplotlib.style.use('ggplot')
    plt.figure()
    plt.grid(True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xlabel(LANGUAGE_NUMBER_ITEMS, fontsize=FONT_SIZE_VALUE)
    plt.ylabel(LANGUAGE_IN_USERS_PROFILES, fontsize=FONT_SIZE_VALUE)
    # Dados desenhados na figura
    plt.scatter(x_data, y_data, alpha=0.5,
                c=scatter_bubble_color)
    # Linha na horizontal
    short_tail_sum = 0
    medium_tail_sum = 99999
    cut_value = 0
    while short_tail_sum < medium_tail_sum:
        cut_value += 1
        short_tail_sum = (df.iloc[:cut_value])[TOTAL_TIMES_LABEL].sum()
        medium_tail_sum = (df.iloc[cut_value:])[TOTAL_TIMES_LABEL].sum()
    cuted_df = df.iloc[:cut_value]
    cut_value = cuted_df[TOTAL_TIMES_LABEL].min()
    plt.axhline(y=cut_value)
    # Pasta para salvar a figura
    path_to_save = pre_processing_to_use_path(db)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # Salvar figura no disco
    plt.savefig(
        path_to_save
        + 'long_tail'
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE,
        bbox_inches='tight'
    )
    # Figura fechada
    plt.close('all')


def user_tail_graphic(analysis_of_users_df, db=None):
    # Ordenação dos dados pelo eixo y
    df = analysis_of_users_df.sort_values(by=[PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL], ascending=[True])
    # Obtenção dos dados do eixo x e do eixo y
    x_data = [i + 1 for i in range(len(df))]
    y_data = df[PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL].tolist()
    # Figura iniciada e atribuição das labels dos eixos x e y
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.grid(True)
    plt.xlabel(LANGUAGE_USER, fontsize=FONT_SIZE_VALUE)
    plt.ylabel(LANGUAGE_PERCENTAGE_POPULAR_ITEMS, fontsize=FONT_SIZE_VALUE)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.xticks(rotation=30)
    # Dados desenhados na figura
    plt.scatter(x_data, y_data, alpha=0.5,
                c=scatter_bubble_color)
    # Linhas na horizontal
    plt.axhline(y=0.75)
    plt.axhline(y=0.25)
    # Pasta para salvar a figura
    path_to_save = pre_processing_to_use_path(db)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # Salvar figura no disco
    plt.savefig(
        path_to_save
        + 'user_tail_graphic'
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE,
        bbox_inches='tight'
    )
    fig.clear()
    # Figura fechada
    plt.close('all')
