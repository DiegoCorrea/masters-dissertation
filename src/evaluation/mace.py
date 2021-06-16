from src.config.labels import USER_LABEL, ORDER_LABEL
from src.posprocessing.distributions import user_get_distribution


def calibration_error(user_distribution, reco_distribution_df):
    result = [abs(float(user_distribution[column]) - float(reco_distribution_df[column])) for column in
              reco_distribution_df]
    a = sum(result)
    b = len(user_distribution)
    return a / b


def ace(user_distribution, final_reco_df, item_mapping):
    result = [calibration_error(user_distribution, user_get_distribution(final_reco_df[:k], item_mapping)) for k in
              final_reco_df[ORDER_LABEL].tolist()]
    ace_value = sum(result) / len(result)
    return ace_value


def mace(preference_distribution_df, final_reco_df, item_mapping):
    result = [ace(user_distribution, final_reco_df[final_reco_df[USER_LABEL] == user_id], item_mapping) for
              user_id, user_distribution in preference_distribution_df.iterrows()]
    return sum(result) / len(preference_distribution_df.index)
