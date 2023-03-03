#!/usr/bin/env python3
"""Split data into sequences.

Prepare the data for input to a neural network. A sequence with a given history
size is extracted from the input data, and matched with the appropriate target
value(s).

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

from config import (
    DATA_PATH,
    DATA_SEQUENTIALIZED_PATH,
    NON_DL_METHODS,
    NON_SEQUENCE_LEARNING_METHODS,
    OUTPUT_FEATURES_PATH,
)
from preprocess_utils import find_files, flatten_sequentialized, split_sequences


def sequentialize(dir_path):
    """Make sequences out of tabular data."""

    filepaths = find_files(dir_path, file_extension=".npz")

    DATA_SEQUENTIALIZED_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["sequentialize"]
    learning_method = yaml.safe_load(open("params.yaml"))["train"]["learning_method"]
    classification = yaml.safe_load(open("params.yaml"))["clean"]["classification"]
    future_predict = params["future_predict"]
    overlap = params["overlap"]

    window_size = params["window_size"]

    if classification:
        target_size = 1
    else:
        target_size = params["target_size"]

    output_columns = np.array(pd.read_csv(OUTPUT_FEATURES_PATH, index_col=0)).reshape(
        -1
    )

    n_output_cols = len(output_columns)

    for filepath in filepaths:

        infile = np.load(filepath)

        X = infile["X"]
        y = infile["y"]

        # Combine y and X to get correct format for sequentializing
        data = np.hstack((y, X))

        # Split into sequences
        X, y = split_sequences(
            data,
            window_size,
            target_size=target_size,
            n_target_columns=n_output_cols,
            future_predict=future_predict,
            overlap=overlap,
        )

        if learning_method in NON_SEQUENCE_LEARNING_METHODS:
            X = flatten_sequentialized(X)

        if params["shuffle_samples"]:
            permutation = np.random.permutation(X.shape[0])
            X = np.take(X, permutation, axis=0)
            y = np.take(y, permutation, axis=0)

        # Save X and y into a binary file
        np.savez(
            DATA_SEQUENTIALIZED_PATH
            / (os.path.basename(filepath).replace("scaled.npz", "sequentialized.npz")),
            X=X,
            y=y,
        )


if __name__ == "__main__":

    np.random.seed(2029)

    sequentialize(sys.argv[1])
