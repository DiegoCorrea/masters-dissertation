from copy import deepcopy

import pandas as pd

from src.config.labels import ITEM_LABEL, TRANSACTION_VALUE_LABEL, BIAS_VALUE_LABEL
from src.config.variables import BIAS_SIGMA, BIAS_ALPHA


def calculating_item_bias(trainset_df, transaction_mean):
    in_bias_trainset_df = deepcopy(trainset_df)
    items_ids = in_bias_trainset_df[ITEM_LABEL].unique().tolist()
    in_bias_trainset_df[TRANSACTION_VALUE_LABEL] -= transaction_mean
    item_bias_df = pd.DataFrame()
    for item in items_ids:
        item_subset_df = in_bias_trainset_df[in_bias_trainset_df[ITEM_LABEL] == item]
        up = item_subset_df[TRANSACTION_VALUE_LABEL].sum()
        down = BIAS_ALPHA + len(item_subset_df)
        item_bias_df = pd.concat(
            [item_bias_df, pd.DataFrame(data=[[item, up / down]], columns=[ITEM_LABEL, BIAS_VALUE_LABEL])])
    return item_bias_df


def calculating_user_bias(users_items, transaction_mean, bias_list, i_id):
    up = users_items[i_id].score - transaction_mean - users_items[i_id].bias
    bias_list.append(up)
    return sum(bias_list) / (BIAS_SIGMA + len(bias_list)), bias_list
