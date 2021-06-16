import logging
import os
import platform
import socket
import sys
import time

from src.config.logging import setup_logging
from src.config.variables import DATASET_LIST, MOVIELENS_20M_DATASET, N_CORES, MEM_RAM
from src.view_clean_and_mining_data import mining_and_clean_dataset

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    """"
    Mining and clean the dataset
    :argv --dataset: The dataset name set by the variable.py
        ex: --dataset=Movielens-20M
    """
    args_len = len(sys.argv)
    node = '' or platform.node() or socket.gethostname() or os.uname().nodename
    dataset = MOVIELENS_20M_DATASET
    if args_len > 1:
        arg_list = sys.argv[1:]
        for arg in arg_list:
            param, value = arg.split('=')
            if param == '--dataset':
                if value not in DATASET_LIST:
                    print('Dataset not found!')
                    exit(1)
                dataset = value
    setup_logging(log_error="mining-error.log", log_info="mining-info.log")
    logger.info("$" * 50)
    logger.info(" ".join(['->', 'Dataset:', dataset]))
    logger.info(" ".join(['>', 'N Jobs:', str(N_CORES), 'RAM:', str(MEM_RAM), '->', node]))
    start_time = time.time()
    logger.info('start at ' + time.strftime('%H:%M:%S'))
    #
    mining_and_clean_dataset(db=dataset)
    #
    finish_time = time.time()
    logger.info('stop at ' + time.strftime('%H:%M:%S'))
    logger.info(" ".join(['>', 'Time Execution:', str(finish_time - start_time)]))
    logger.info(" ".join(['>', 'System shutdown']))
