#!/usr/bin/env python3
"""Split data into training and test set.

Author:
    Erik Johannes Husom

Date:
    2021-02-24

"""
import os
import random
import shutil
import sys

import numpy as np
import yaml

from config import DATA_SPLIT_PATH
from preprocess_utils import find_files


def split(dir_path):
    """Split data into train and test set.

    Training files and test files are saved to different folders.

    Args:
        dir_path (str): Path to directory containing files.

    """

    with open("params.yaml", "r", encoding="UTF-8") as infile:
        params = yaml.safe_load(infile)["split"]

    shuffle_files = params["shuffle_files"]

    DATA_SPLIT_PATH.mkdir(parents=True, exist_ok=True)

    filepaths = find_files(dir_path, file_extension=".npy")

    # Handle special case where there is only one data file.
    if isinstance(filepaths, str) or len(filepaths) == 1:
        filepath = filepaths[0]

        data = np.load(filepath)

        train_size = int(len(data) * params["train_split"])

        data_train = None
        data_test = None

        data_train = data[:train_size, :]
        data_test = data[train_size:, :]

        np.save(
            DATA_SPLIT_PATH / os.path.basename(filepath).replace("featurized", "train"),
            data_train,
        )

        np.save(
            DATA_SPLIT_PATH / os.path.basename(filepath).replace("featurized", "test"),
            data_test,
        )

    else:

        if shuffle_files:
            random.shuffle(filepaths)

        # Parameter 'train_split' is used to find out no. of files in training set
        file_split = int(len(filepaths) * params["train_split"])

        training_files = []
        test_files = []

        training_files = filepaths[:file_split]
        test_files = filepaths[file_split:]

        for filepath in filepaths:

            if filepath in training_files:
                shutil.copyfile(
                    filepath,
                    DATA_SPLIT_PATH
                    / os.path.basename(filepath).replace("featurized", "train"),
                )

            elif filepath in test_files:
                shutil.copyfile(
                    filepath,
                    DATA_SPLIT_PATH
                    / os.path.basename(filepath).replace("featurized", "test"),
                )

if __name__ == "__main__":

    np.random.seed(2029)

    split(sys.argv[1])
