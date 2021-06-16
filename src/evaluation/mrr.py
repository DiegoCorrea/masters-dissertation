from src.config.labels import ITEM_LABEL


def get_rr_from_list(relevance_array):
    relevance_list_size = len(relevance_array)
    if relevance_list_size == 0:
        return 0.0
    for i in range(relevance_list_size):
        if relevance_array[i]:
            return 1 / (i + 1)
    return 0.0


def mrr(reco_items_df, test_items_ids):
    precision = [True if x in test_items_ids else False for x in reco_items_df[ITEM_LABEL].tolist()]
    # if set(reco_items_df[ITEM_LABEL].tolist()) & set(test_items_ids):
    #     print('MMMMRRRRRRRRRRRRRRRRRRRR')
    #     print(set(reco_items_df[ITEM_LABEL].tolist()) & set(test_items_ids))
    return get_rr_from_list(precision)
