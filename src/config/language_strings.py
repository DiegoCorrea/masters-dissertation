import gettext

lang = "pt"
lang_translations = gettext.translation('base', localedir='locales', languages=[lang])
lang_translations.install()
_ = lang_translations.gettext

LANGUAGE_LOAD_DATA_SET = _("Load the Data Set")
LANGUAGE_MOVIELENS_SELECTED = _("Movielens Selected")
LANGUAGE_OMS_SELECTED = _("OMS Selected")
LANGUAGE_DATA_SET_MEMORY = _("*** Data Set in the Memory ***")
LANGUAGE_PARSING_DATA = _("Parsing Data Set")
LANGUAGE_PROCESSING_STEP_START = _("+ + + Recommendation Process - Start + + +")
LANGUAGE_PROCESSING_STEP_STOP = _("+ + + Recommendation Process - Finish + + +")
LANGUAGE_ANALYZING_POPULARITY = _("- - Analyzing the Popularity")
LANGUAGE_ANALYZING_GENRES = _("- - Analyzing the Genres")
LANGUAGE_USER_KNN_START = _('UserKNN - Start')

LANGUAGE_USER_PROFILE_SIZE = _("User profile size")
LANGUAGE_NUMBER_POPULAR_ITEMS = _("Number of popular items")
LANGUAGE_NUMBER_UNPOPULAR_ITEMS = _("Number of unpopular items")
LANGUAGE_NUMBER_USERS = _("Number of users")
LANGUAGE_PERCENTAGE_POPULAR_ITEMS = _("Percentage of popular items")
LANGUAGE_PERCENTAGE_UNPOPULAR_ITEMS = _("Percentage of unpopular items")
LANGUAGE_NUMBER_ITEMS = _("Number of items")
LANGUAGE_IN_USERS_PROFILES = _("In users profiles")
LANGUAGE_NICHE = _('Niche')
LANGUAGE_FOCUSED = _('Focused')
LANGUAGE_DIVERSE = _('Diverse')
LANGUAGE_EXPLORER = _('Explorer')
LANGUAGE_COMMON = _('Common')
LANGUAGE_LOYAL = _('Loyal')
LANGUAGE_POPULARITY = _('Popularity')
LANGUAGE_NUMBER_GENRES = _("Number of genres")
LANGUAGE_GROUPS_USERS = _("Groups of users")

LANGUAGE_POPULARITY_RATIO = _('Popularity Ratio')
LANGUAGE_RECOMMENDATION_FREQUENCY = _('Recommendation Frequency')

LANGUAGE_ALGORITHM = _('Algorithm')
LANGUAGE_ALGORITHMS = _('Algorithms')

LANGUAGE_MAE = 'MAE'
LANGUAGE_MSE = 'MSE'
LANGUAGE_RMSE = 'RMSE'
error_list_print = [LANGUAGE_MAE, LANGUAGE_MSE, LANGUAGE_RMSE]

LANGUAGE_PANDAS_TO_SURPRISE_DATA = _("-> Convert Pandas DataFrame to Surprise Data Structure")
LANGUAGE_RECOMMENDER_ALGORITHM_START = _("Recommender Algorithm Start")
LANGUAGE_RECOMMENDER_ALGORITHM_STOP = _("Recommender Algorithm Stop")
LANGUAGE_USER_KNN_TRANSFORM_DATA = _('UserKNN - Transform data')
LANGUAGE_USER_KNN_EVALUATE = _('UserKNN - Evaluate')
LANGUAGE_USER_KNN_STOP = _('UserKNN - Stop')

LANGUAGE_ITEM_KNN_START = _('ItemKNN - Start')
LANGUAGE_ITEM_KNN_TRANSFORM_DATA = _('ItemKNN - Transform data')
LANGUAGE_ITEM_KNN_EVALUATE = _('ItemKNN - Evaluate')
LANGUAGE_ITEM_KNN_STOP = _('ItemKNN - Stop')

LANGUAGE_SVD_START = _('SVD - Start')
LANGUAGE_SVD_TRANSFORM_DATA = _('SVD - Transform data')
LANGUAGE_SVD_EVALUATE = _('SVD - Evaluate')
LANGUAGE_SVD_STOP = _('SVD - Stop')

LANGUAGE_SVDPP_START = _('SVD++ - Start')
LANGUAGE_SVDPP_TRANSFORM_DATA = _('SVD++ - Transform data')
LANGUAGE_SVDPP_EVALUATE = _('SVD++ - Evaluate')
LANGUAGE_SVDPP_STOP = _('SVD++ - Stop')

LANGUAGE_NMF_START = _('NMF - Start')
LANGUAGE_NMF_TRANSFORM_DATA = _('NMF - Transform data')
LANGUAGE_NMF_EVALUATE = _('NMF - Evaluate')
LANGUAGE_NMF_STOP = _('NMF - Stop')

LANGUAGE_TOP_N_ITEMS = _('Top N items')
LANGUAGE_DISTANCE_VALUE = _('Distance value')

LANGUAGE_TRANSFORMING_DATA = _('Transforming data')
LANGUAGE_EVALUATE = _('+ Evaluate')

LANGUAGE_MINING_PREFERENCES = _('Mining preferences')
LANGUAGE_CREATING_FOLDS = _('Creating Fold')
LANGUAGE_SAVE_TRAIN = _('Saving train set')
LANGUAGE_SAVE_TEST = _('Saving test set')
LANGUAGE_SPLIT_DATA = _('Split into train and test')
LANGUAGE_FOLD = _('Fold')
LANGUAGE_MINING_ITEMS = _('Mining Items')
LANGUAGE_TOTAL_USERS = _('Total of users')
LANGUAGE_MERGE_PREFERENCES_INTO_DF = _('Merge preferences into df')
LANGUAGE_DONE = _('Done')

LANGUAGE_POSTPROCESSING_START = _('Pos Processing - Start')
LANGUAGE_POSTPROCESSING_STOP = _('Pos Processing - Stop')
LANGUAGE_AT_LIST = _('Make list at: ')

LANGUAGE_SLOPE_ONE_START = _('Slope One - Start')
LANGUAGE_SLOPE_ONE_TRANSFORM_DATA = _('Slope One - Transform Data')
LANGUAGE_SLOPE_ONE_EVALUATE = _('Slope One - Evalluate')
LANGUAGE_SLOPE_ONE_STOP = _('Slope One - Stop')

LANGUAGE_CALIBRATION = _('Calibration')
LANGUAGE_PREFERENCE = _('Preference')
LANGUAGE_RECOMMENDATION = _('Recommendation')

LANGUAGE_LINEAR_CALIBRATION = _('Linear Calibration')
LANGUAGE_SURROGATE = _('Surrogate')
LANGUAGE_KL = _('Kullback-Leibler')
LANGUAGE_HE = _('Hellinger')
LANGUAGE_CHI = _('Person Chi Square')

LANGUAGE_COUNT_GENRES = _('Count Genres')
LANGUAGE_PROBABILITY_MEAN = _('Probability Mean')
LANGUAGE_VARIANCE = _('Variance')
LANGUAGE_MANUAL_VALUE = _('Manual Value')

LANGUAGE_MEAN = _('Mean')
LANGUAGE_MEDIAN = _('Median')
LANGUAGE_USER = _('User')

LANGUAGE_SHORT_TAIL = _('Short Tail')
LANGUAGE_MEDIUM_TAIL = _('Medium Tail')

LANGUAGE_MENU = _('Menu')
LANGUAGE_MENU_EXTRACT_AND_CLEAN = _('Extract and clean the raw data')
LANGUAGE_MENU_STATISTICAL_RAW_AND_CLEAN_DATASET = _('Statistical describe the Raw and Clean dataset')
LANGUAGE_GRAPHICS_FROM_DATASET = _('Create charts from clean dataset')
LANGUAGE_GRID_SEARCH = _('Grid Search')
LANGUAGE_RUN_ALL = _('Run All')
LANGUAGE_RUN_ONE = _('Run Test')
LANGUAGE_ANALYZE_THE_RESULTS = _('Analyze the results')
LANGUAGE_GRAPHICS_FROM_RESULTS = _('Create charts from results')
LANGUAGE_CHECK_CONFIG = _('Check configs')
LANGUAGE_EXIT = _('Exit')
LANGUAGE_CHOICE_MENU = _("Enter the choice number: ")
LANGUAGE_FINISH_PROGRAM = _('Finish')

LANGUAGE_CREATE_GRAPHICS = _("Create graphics")

LANGUAGE_CALCULATING_USER_MODEL_WGENRE_DISTRIBUTION = _("Calculating User Model Weight Genre Distribution")

LANGUAGE_TRAINING_THE_RECOMMENDER = _("Training the recommender")

LANGUAGE_WEIGHT = _('Weight')
