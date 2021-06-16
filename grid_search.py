import logging
import os
import platform
import socket
import sys
import time

from src.config.labels import SVD_LABEL, FULL_RECOMMENDERS
from src.config.logging import setup_logging
from src.config.variables import N_CORES, MOVIELENS_1M_DATASET, DATASET_LIST, K_FOLDS_VALUES, MEM_RAM
from src.view_grid_search import customized_grid_search

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    """
    Start the Personalized Recommendation
    """
    recommender = SVD_LABEL
    dataset = MOVIELENS_1M_DATASET
    cv = 3
    node = '' or platform.node() or socket.gethostname() or os.uname().nodename
    args_len = len(sys.argv)
    if args_len > 3:
        arg_list = sys.argv[1:]
        for arg in arg_list:
            param, value = arg.split('=')
            if param == '--recommender':
                if value not in FULL_RECOMMENDERS:
                    print('Recommender not found!')
                    exit(1)
                recommender = value
            elif param == '--dataset':
                if value not in DATASET_LIST:
                    print('Dataset not found!')
                    exit(1)
                dataset = value
            elif param == '--cv':
                if int(value) <= 0 or int(value) > K_FOLDS_VALUES:
                    print('Fold not found!')
                    exit(1)
                cv = int(value)
            elif param == '--external-node':
                node = str(value)
    # os.system('cls||clear')
    setup_logging(log_error="-".join(['gs', recommender, dataset, node, "-error.log"]),
                  log_info="-".join(['gs', recommender, dataset, node, "-info.log"]),
                  save_path="logs/grid/")
    logger.info("$" * 50)
    logger.info(" ".join(['>', 'Label:', recommender, '->', 'Dataset:', dataset, '->', 'CV:', str(cv)]))
    logger.info(" ".join(['>', 'N Jobs:', str(N_CORES), 'RAM:', str(MEM_RAM), '->', node]))
    # Start the recommender in a personalized way
    start_time = time.time()
    logger.info('start at ' + time.strftime('%H:%M:%S'))
    customized_grid_search(recommender, dataset, cv)
    finish_time = time.time()
    logger.info('stop at ' + time.strftime('%H:%M:%S'))
    logger.info(" ".join(['>', 'Time Execution:', str(finish_time - start_time)]))
    logger.info(" ".join(['>', 'System shutdown']))
