from src.config.labels import TRADE_OFF_LABEL, COUNT_GENRES_TRADE_OFF_LABEL


def count_genres(row_df):
    """
    A trade off degree based in the user genre count

    :param row_df: A user dataframe row based in the genres distribution
    :return: A float that is the trade off degree
    """
    count = 0
    for i, number in row_df.iteritems():
        if number > 0.0:
            count += 1
    return count / len(row_df)


def variance(row_df):
    """
    A trade off degree based in the user genre variance

    :param row_df: A user dataframe row based in the genres distribution
    :return: A float that is the trade off degree
    """
    return 1 - row_df.var()


def personalized_trade_off(user_model_genres_distr_df, config):
    """
    Use the personalized trade off to do the pos processing

    :param user_model_genres_distr_df: Distribution based on the user model genres distribution
    :param config: A dict with the pos process combination configuration
    :return: A recommendation list with n items
    """
    lmbda = 0.0
    if config[TRADE_OFF_LABEL] == COUNT_GENRES_TRADE_OFF_LABEL:
        lmbda = count_genres(user_model_genres_distr_df)
    else:
        lmbda = variance(user_model_genres_distr_df)
    return lmbda
