from collections import defaultdict

import pandas as pd

from src.config.labels import USER_LABEL, ITEM_LABEL, \
    TRANSACTION_VALUE_LABEL
from src.config.variables import CANDIDATES_LIST_SIZE


def paralleling_convert(user, n=CANDIDATES_LIST_SIZE):
    top_n_df = pd.DataFrame()
    uid, user_ratings = user
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    for iid, est, true_r in user_ratings[:n]:
        top_n_df = pd.concat([top_n_df,
                              pd.DataFrame(data=[[uid, iid, est]],
                                           columns=[USER_LABEL, ITEM_LABEL, TRANSACTION_VALUE_LABEL])
                              ])
    return top_n_df


def surprise_to_pandas_get_candidates_items(predictions, n=CANDIDATES_LIST_SIZE):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A pandas dataframe with the top n items.
    """
    # print(LANGUAGE_TRANSFORM_DATA)
    # First, map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est, true_r))
    map_results_df = [paralleling_convert(user) for user in top_n.items()]
    return pd.concat(map_results_df, sort=False)
