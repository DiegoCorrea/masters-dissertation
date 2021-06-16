# libraries
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from src.config.variables import BAR_WIDTH_VALUE, mae_label, rmse_label, mse_label, baselines_results_path, \
    QUALITY_VALUE, DPI_VALUE, algorithm_label
from src.config.language_strings import LANGUAGE_ALGORITHMS, LANGUAGE_MAE, LANGUAGE_MSE, LANGUAGE_RMSE


def algorithm_error_processing(df):
    # set height of bar
    bars1 = df[mae_label].tolist()
    bars2 = df[mse_label].tolist()
    bars3 = df[rmse_label].tolist()

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + BAR_WIDTH_VALUE for x in r1]
    r3 = [x + BAR_WIDTH_VALUE for x in r2]

    # Make the plot
    plt.figure()
    plt.bar(r1, bars1, color='#7f6d5f', width=BAR_WIDTH_VALUE, edgecolor='white', label=LANGUAGE_MAE, hatch='//')
    plt.bar(r2, bars2, color='#557f2d', width=BAR_WIDTH_VALUE, edgecolor='white', label=LANGUAGE_MSE, hatch="\\")
    plt.bar(r3, bars3, color='#2d7f5e', width=BAR_WIDTH_VALUE, edgecolor='white', label=LANGUAGE_RMSE, hatch='*')

    # Add xticks on the middle of the group bars
    plt.xlabel(LANGUAGE_ALGORITHMS, fontweight='bold')
    plt.xticks([r + BAR_WIDTH_VALUE for r in range(len(bars1))], df[algorithm_label].tolist())

    # Create legend & Save graphic
    plt.legend(loc=0)
    if not os.path.exists(baselines_results_path):
        os.makedirs(baselines_results_path)
    plt.savefig(
        baselines_results_path
        + 'algorithm_error_processing'
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE
    )
    plt.close('all')


# ####
#
# ####
def algorithm_error_postprocessing(df):
    # set height of bar
    bars1 = df[mae_label].tolist()
    bars2 = df[mse_label].tolist()
    bars3 = df[rmse_label].tolist()

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + BAR_WIDTH_VALUE for x in r1]
    r3 = [x + BAR_WIDTH_VALUE for x in r2]

    # Make the plot
    plt.figure()
    plt.bar(r1, bars1, color='#7f6d5f', width=BAR_WIDTH_VALUE, edgecolor='white', label=LANGUAGE_MAE, hatch='//')
    plt.bar(r2, bars2, color='#557f2d', width=BAR_WIDTH_VALUE, edgecolor='white', label=LANGUAGE_MSE, hatch="\\")
    plt.bar(r3, bars3, color='#2d7f5e', width=BAR_WIDTH_VALUE, edgecolor='white', label=LANGUAGE_RMSE, hatch='*')

    # Add xticks on the middle of the group bars
    plt.xlabel(LANGUAGE_ALGORITHMS, fontweight='bold')
    plt.xticks([r + BAR_WIDTH_VALUE for r in range(len(bars1))], df[algorithm_label].tolist())

    # Create legend & Save graphic
    plt.legend(loc=0)
    if not os.path.exists(baselines_results_path):
        os.makedirs(baselines_results_path)
    plt.savefig(
        baselines_results_path
        + 'algorithm_error_processing'
        + '.png',
        format='png',
        dpi=DPI_VALUE,
        quality=QUALITY_VALUE
    )
    plt.close('all')
