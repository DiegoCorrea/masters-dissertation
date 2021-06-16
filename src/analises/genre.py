import numpy as np
import pandas as pd

from src.config.labels import GENRES_LABEL, USER_LABEL, USER_MODEL_SIZE_LABEL, NUMBER_OF_GENRES


def total_genre_map(user, user_df):
    genres_list = []
    for row in user_df.itertuples():
        item_genre = getattr(row, GENRES_LABEL)
        splitted = item_genre.split('|')
        genres_list = genres_list + [genre for genre in splitted]
    return pd.DataFrame([[user, len(user_df), len(np.unique(genres_list))]],
                        columns=[USER_LABEL, USER_MODEL_SIZE_LABEL, NUMBER_OF_GENRES])


def compute_genre(transactions_df):
    users_list = transactions_df[USER_LABEL].unique().tolist()
    user_df = [total_genre_map(user, transactions_df[transactions_df[USER_LABEL] == user]) for user in users_list]
    analysis_of_users_df = pd.concat(user_df, sort=False)
    analysis_of_users_df.set_index(USER_LABEL, inplace=True, drop=True)
    return analysis_of_users_df
