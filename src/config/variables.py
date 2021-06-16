import multiprocessing

from psutil import virtual_memory


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


MEM_RAM = get_size(virtual_memory().total, suffix="B")
# Machine config #
N_CORES = multiprocessing.cpu_count()

# Dataset config
MOVIELENS_20M_DATASET = 'Movielens-20M'
OMS_DATASET = 'OMS'

MOVIELENS_1M_DATASET = 'Movielens-1M'
OMS_10K_DATASET = 'OMS-10k'

MOVIELENS_25M_DATASET = 'Movielens-25M'
OMS_FULL_DATASET = 'OMS-Full'

MOVIELENS_DATASET_LIST = [MOVIELENS_1M_DATASET, MOVIELENS_20M_DATASET, MOVIELENS_25M_DATASET]
OMS_DATASET_LIST = [OMS_10K_DATASET, OMS_DATASET, OMS_FULL_DATASET]

DATASET_LIST = OMS_DATASET_LIST + MOVIELENS_DATASET_LIST

MOVIELENS_PROFILE_LEN_CUT_VALUE = 30
MOVIELENS_ITEM_TRANSACTION_CUT_VALUE = 3
OMS_PROFILE_LEN_CUT_VALUE = 30
OMS_ITEM_TRANSACTION_CUT_VALUE = 3

RATING_CUT_VALUE = 4.0
LISTEN_CUT_VALUE = 3
TEST_SIZE = 0.33
TRAIN_SIZE = 0.67
K_FOLDS_VALUES = 3

# Algorithm hyper param
ALPHA_VALUE = 0.01

# Bias
BIAS_ALPHA = 0.01
BIAS_SIGMA = 0.01

# Data Model config
RECOMMENDATION_LIST_SIZE = 10
CANDIDATES_LIST_SIZE = 100

# Colors, Hatch
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', '#0F0F0F0F']
niche_graph_color = 'tab:blue'
focused_graph_color = 'tab:cyan'
diverse_graph_color = 'tab:olive'
popularity_graph_color = 'tab:purple'

scatter_bubble_color = '#0066ff'
scatter_average_line_color = '#AA0000'
scatter_median_line_color = 'goldenrod'

# Markers
markers_list = ['o', '^', 's', 'D', 'x', 'p', '.', '1', '|', '*', '2']

# Line Style
line_style_list = [':', '--', ':', '-', '-', '-', '--', ':', '--', '-.', '-.']
linear_line_style_list = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
log_line_style_list = [':', ':', ':', ':', ':', ':', ':', ':', ':']
special_markers_list = ['o', '^', 's']

cmap_color_scale = 'viridis'

# Constant Values
FONT_SIZE_VALUE = 18
BAR_WIDTH_VALUE = 0.25
DPI_VALUE = 300
QUALITY_VALUE = 100
