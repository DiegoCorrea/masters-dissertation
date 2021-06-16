from src.analises.genre import compute_genre
from src.analises.popularity import compute_popularity
from src.config.language_strings import LANGUAGE_ANALYZING_POPULARITY, LANGUAGE_ANALYZING_GENRES, \
    LANGUAGE_MOVIELENS_SELECTED, LANGUAGE_OMS_SELECTED, LANGUAGE_CREATE_GRAPHICS
from src.config.labels import USER_LABEL, ITEM_LABEL, USER_MODEL_SIZE_LABEL
from src.config.variables import MOVIELENS_1M_DATASET, OMS_DATASET, MOVIELENS_20M_DATASET, OMS_FULL_DATASET
from src.graphics.genres import user_model_size_by_number_of_genres, compare_genre_distribution_bar
from src.graphics.popularity import user_model_size_by_percentage_short_tail_items, \
    long_tail_graphic, user_model_size_by_short_tail_items, \
    user_model_size_by_percentage_medium_tail_items, user_model_size_by_medium_tail_items, user_tail_graphic
from src.posprocessing.distributions import genre_probability_distribution
from src.preprocessing.load_database import movielens_load_full_data, oms_load_full_data
import logging
logger = logging.getLogger(__name__)


def popularity_graphics(transactions_df, db):
    analysis_of_users_df, analysis_of_items_df = compute_popularity(transactions_df)
    long_tail_graphic(analysis_of_items_df, db)
    user_tail_graphic(analysis_of_users_df, db)
    # user_model_size_by_short_tail_items(analysis_of_users_df, db)
    # user_model_size_by_medium_tail_items(analysis_of_users_df, db)
    # user_model_size_by_percentage_short_tail_items(analysis_of_users_df, db)
    # user_model_size_by_percentage_medium_tail_items(analysis_of_users_df, db)


def genres_graphics(transactions_df, items_df, db):
    analysis_of_users_df = compute_genre(transactions_df)
    if db == 1:
        analysis_of_users_df = (analysis_of_users_df.sort_values(by=[USER_MODEL_SIZE_LABEL], ascending=[False])).iloc[1:]
    user_model_size_by_number_of_genres(analysis_of_users_df, db)
    users_genre_distr_df = genre_probability_distribution(transactions_df, label=USER_LABEL)
    items_genre_distr_df = genre_probability_distribution(items_df, label=ITEM_LABEL)
    compare_genre_distribution_bar(users_genre_distr_df, items_genre_distr_df, db)


def diversity_graphics(transactions_df, items_df, db):
    pass


def database_graphics():
    print("$" * 50)
    logger.info(" ".join(['>', LANGUAGE_CREATE_GRAPHICS, ":", LANGUAGE_MOVIELENS_SELECTED]))
    transactions_df, items_df = movielens_load_full_data(db=MOVIELENS_20M_DATASET)
    print(LANGUAGE_ANALYZING_POPULARITY)
    popularity_graphics(transactions_df, db=MOVIELENS_20M_DATASET)
    # print(LANGUAGE_ANALYZING_GENRES)
    # genres_graphics(transactions_df, items_df, db=MOVIELENS_20M_DATASET)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # print("$" * 50)
    # print(">>>>> Create graphics: " + LANGUAGE_OMS_SELECTED)
    # transactions_df, items_df = oms_load_full_data(db=OMS_FULL_DATASET)
    # print(LANGUAGE_ANALYZING_POPULARITY)
    # popularity_graphics(transactions_df, db=OMS_FULL_DATASET)
    # print(LANGUAGE_ANALYZING_GENRES)
    # genres_graphics(transactions_df, items_df, db=OMS_FULL_DATASET)
