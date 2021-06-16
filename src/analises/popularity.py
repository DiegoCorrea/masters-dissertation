from multiprocessing import Pool

import pandas as pd

from src.config.labels import (ITEM_LABEL, USER_LABEL, USER_MODEL_SIZE_LABEL, NUMBER_OF_SHORT_TAIL_ITEMS_LABEL,
                               PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL,
                               TOTAL_TIMES_LABEL, TYPE_OF_POPULARITY,
                               MEDIUM_TAIL_TYPE, SHORT_TAIL_TYPE, NUMBER_OF_MEDIUM_TAIL_ITEMS_LABEL,
                               PERCENTAGE_OF_MEDIUM_TAIL_ITEMS_LABEL, FOCUSED_TYPE,
                               NICHE_TYPE, DIVERSE_TYPE)
from src.config.variables import N_CORES


# ######################### #
#      Data set Compute     #
# ######################### #
def map_get_users_popularity(user, u_profile_df, item_short_tail_id_list, item_medium_tail_id_list):
    users_models_size = len(u_profile_df)
    short_tail_size = len(u_profile_df[u_profile_df[ITEM_LABEL].isin(item_short_tail_id_list)])
    medium_tail_size = len(u_profile_df[u_profile_df[ITEM_LABEL].isin(item_medium_tail_id_list)])

    return pd.DataFrame(data=[[user, users_models_size,
                               short_tail_size, medium_tail_size,
                               short_tail_size / users_models_size, medium_tail_size / users_models_size]],
                        columns=[USER_LABEL, USER_MODEL_SIZE_LABEL,
                                 NUMBER_OF_SHORT_TAIL_ITEMS_LABEL, NUMBER_OF_MEDIUM_TAIL_ITEMS_LABEL,
                                 PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL, PERCENTAGE_OF_MEDIUM_TAIL_ITEMS_LABEL])


def count_popularity_item(transactions_df):
    item_popularity_count = transactions_df[ITEM_LABEL].value_counts()
    popularity_dict = item_popularity_count.to_dict()
    analysis_of_items_df = pd.DataFrame(columns=[ITEM_LABEL, TOTAL_TIMES_LABEL])
    for item in popularity_dict:
        analysis_of_items_df = pd.concat([analysis_of_items_df, pd.DataFrame(data=[[item, popularity_dict[item]]],
                                                                             columns=[ITEM_LABEL, TOTAL_TIMES_LABEL])])
    return analysis_of_items_df


def alternative_get_short_tail_items(analysis_of_items_df):
    df = analysis_of_items_df.sort_values(by=[TOTAL_TIMES_LABEL], ascending=[False])
    short_tail_sum = 0
    medium_tail_sum = 99999
    cut_value = 0
    while short_tail_sum < medium_tail_sum:
        cut_value += 1
        short_tail_sum = (df.iloc[:cut_value])[TOTAL_TIMES_LABEL].sum()
        medium_tail_sum = (df.iloc[cut_value:])[TOTAL_TIMES_LABEL].sum()
    cuted_df = df.iloc[:cut_value]
    cut_value = cuted_df[TOTAL_TIMES_LABEL].min()
    return (cuted_df[cuted_df[TOTAL_TIMES_LABEL] >= cut_value])[ITEM_LABEL].tolist()


def get_focused_users(analysis_of_users_df):
    df = analysis_of_users_df[analysis_of_users_df[PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL] >= 0.75]
    return df[USER_LABEL].tolist()


def get_niche_users(analysis_of_users_df):
    df = analysis_of_users_df[analysis_of_users_df[PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL] <= 0.25]
    return df[USER_LABEL].tolist()


def compute_popularity(transactions_df):
    # Item Popularity
    analysis_of_items_df = count_popularity_item(transactions_df)

    item_short_tail_id_list = alternative_get_short_tail_items(analysis_of_items_df)

    analysis_of_items_df[TYPE_OF_POPULARITY] = MEDIUM_TAIL_TYPE

    analysis_of_items_df.loc[
        analysis_of_items_df[ITEM_LABEL].isin(item_short_tail_id_list), TYPE_OF_POPULARITY] = SHORT_TAIL_TYPE
    item_medium_tail_id_list = (analysis_of_items_df[analysis_of_items_df[TYPE_OF_POPULARITY] == MEDIUM_TAIL_TYPE])[
        ITEM_LABEL].unique().tolist()

    # User and Item
    users_id_list = transactions_df[USER_LABEL].unique().tolist()
    query = [(user_id,
              transactions_df[transactions_df[USER_LABEL] == user_id],
              item_short_tail_id_list,
              item_medium_tail_id_list) for user_id in users_id_list]

    pool = Pool(N_CORES)
    list_df = pool.starmap(map_get_users_popularity, query)
    pool.close()
    pool.join()
    analysis_of_users_df = pd.concat(list_df, sort=False)

    niche_id_list = get_niche_users(analysis_of_users_df)
    focused_id_list = get_focused_users(analysis_of_users_df)

    analysis_of_users_df[TYPE_OF_POPULARITY] = DIVERSE_TYPE

    analysis_of_users_df.loc[
        analysis_of_users_df[USER_LABEL].isin(niche_id_list), TYPE_OF_POPULARITY] = NICHE_TYPE
    analysis_of_users_df.loc[
        analysis_of_users_df[USER_LABEL].isin(focused_id_list), TYPE_OF_POPULARITY] = FOCUSED_TYPE

    return analysis_of_users_df, analysis_of_items_df
