import os

from src.config.labels import TEST_RECOMMENDERS
from src.config.language_strings import LANGUAGE_MENU_EXTRACT_AND_CLEAN, \
    LANGUAGE_MENU_STATISTICAL_RAW_AND_CLEAN_DATASET, LANGUAGE_MENU, LANGUAGE_GRAPHICS_FROM_DATASET, \
    LANGUAGE_GRID_SEARCH, LANGUAGE_RUN_ALL, LANGUAGE_RUN_ONE, LANGUAGE_ANALYZE_THE_RESULTS, \
    LANGUAGE_GRAPHICS_FROM_RESULTS, LANGUAGE_CHECK_CONFIG, LANGUAGE_EXIT, LANGUAGE_FINISH_PROGRAM, LANGUAGE_CHOICE_MENU
from src.config.logging import setup_logging
from src.config.variables import MOVIELENS_20M_DATASET, OMS_FULL_DATASET
from src.view_analysis_results import analysis_results, coefficient_for_calibration
from src.view_clean_and_mining_data import link_and_clean_datasets
from src.view_database_analysis import database_analysis
from src.view_database_graphics import database_graphics
from src.view_graphics_results import generating_results_graphics
from src.view_grid_search import grid_search
from src.view_recommender_process import one_recommender_process, recommender_process, merge_all_results


def main_menu():
    """
    Main menu function, print all option to be choice
    :return: The chosen option number [0,9], otherwise -1
    """
    print('#' * 50)
    print(" ".join(['#' * 22, LANGUAGE_MENU, '#' * 22]))
    print('#' * 50)
    print(" ".join(['# 1 -', LANGUAGE_MENU_EXTRACT_AND_CLEAN]))
    print(" ".join(['# 2 -', LANGUAGE_MENU_STATISTICAL_RAW_AND_CLEAN_DATASET]))
    print(" ".join(['# 3 -', LANGUAGE_GRAPHICS_FROM_DATASET]))
    print(" ".join(['# 4 -', LANGUAGE_GRID_SEARCH]))
    print(" ".join(['# 5 -', LANGUAGE_RUN_ONE]))
    print(" ".join(['# 6 -', LANGUAGE_RUN_ALL]))
    print(" ".join(['# 7 -', LANGUAGE_ANALYZE_THE_RESULTS]))
    print(" ".join(['# 8 -', LANGUAGE_GRAPHICS_FROM_RESULTS]))
    print(" ".join(['# 9 -', LANGUAGE_CHECK_CONFIG]))
    print(" ".join(['# 0 -', LANGUAGE_EXIT]))
    print('#' * 50)
    chosen_option = int(input(LANGUAGE_CHOICE_MENU))
    return chosen_option


if __name__ == '__main__':
    os.system('clear')
    setup_logging(log_error="menu-error.log", log_info="menu-info.log")
    chosen = -1
    # Menu principal usado para escolher o mo
    while chosen != 0:
        chosen = main_menu()
        if chosen == 1:
            link_and_clean_datasets()
        elif chosen == 2:
            database_analysis()
            chosen = -1
        elif chosen == 3:
            database_graphics()
            chosen = -1
        elif chosen == 4:
            grid_search()
            chosen = -1
        elif chosen == 5:
            one_recommender_process()
            chosen = -1
        elif chosen == 6:
            recommender_process()
            generating_results_graphics()
            chosen = -1
        elif chosen == 7:
            # analysis_results()
            coefficient_for_calibration()
            chosen = -1
        elif chosen == 8:
            generating_results_graphics()
            chosen = -1
        elif chosen == 9:
            chosen = -1
    print(LANGUAGE_FINISH_PROGRAM)
