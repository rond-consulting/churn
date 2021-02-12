"""
This script contains util functions for loading data

Author: Hans Weda
Date: 12 february 2021
"""

import os

import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from kaggle.api.kaggle_api_extended import KaggleApi


DATA_DIR = os.path.join("..", "data", "raw")


def load_from_kaggle(owner: str, dataset_name: str) -> pd.DataFrame:
    """
    This function downloads data from kaggle into DATA_DIR and returns the first file (alphabetically).
    Following directions on https://medium.com/analytics-vidhya/fetch-data-from-kaggle-with-python-9154a4c610e3
    The documentation of the kaggle api can be found here: https://github.com/Kaggle/kaggle-api/tree/master/kaggle
    :param owner: the owner of the dataset on Kaggle
    :param dataset_name: the name of the dataset
    :return: the alphabetically first file from the downloaded dataset
    """
    # make sure raw data directory exists
    target_directory = os.path.join(DATA_DIR, dataset_name)
    os.makedirs(target_directory, exist_ok=True)

    # connect to kaggle
    api = KaggleApi()
    api.authenticate()

    # download if it doesn't exist already and unzip
    api.dataset_download_files(dataset=f"{owner}/{dataset_name}", path=target_directory, unzip=True)

    # read the first file and return
    df = pd.read_csv(os.path.join(target_directory, os.listdir(target_directory)[0]))

    return df
