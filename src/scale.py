#!/usr/bin/env python3
"""Scaling the inputs of the data set.

Possible scaling methods

TODO:
    Implement scaling when there is only one workout file.

Author:
    Erik Johannes Husom

Created:
    2020-09-16

"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from config import (
    DATA_SCALED_PATH,
    INPUT_SCALER_PATH,
    OUTPUT_FEATURES_PATH,
    OUTPUT_SCALER_PATH,
    SCALER_PATH,
)
from preprocess_utils import find_files


def scale(dir_path):
    """Scale training and test data.

    Args:
        dir_path (str): Path to directory containing files.

    """

    filepaths = find_files(dir_path, file_extension=".npy")

    DATA_SCALED_PATH.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["scale"]
    input_method = params["input"]
    output_method = params["output"]
    classification = yaml.safe_load(open("params.yaml"))["clean"]["classification"]
    onehot_encode_target = yaml.safe_load(open("params.yaml"))["clean"][
        "onehot_encode_target"
    ]

    if input_method == "standard":
        input_scaler = StandardScaler()
    elif input_method == "minmax":
        input_scaler = MinMaxScaler()
    elif input_method == "robust":
        input_scaler = RobustScaler()
    elif input_method is None:
        input_scaler = StandardScaler()
    else:
        raise NotImplementedError(f"{input_method} not implemented.")

    if output_method == "standard":
        output_scaler = StandardScaler()
    elif output_method == "minmax":
        output_scaler = MinMaxScaler()
    elif output_method == "robust":
        output_scaler = RobustScaler()
    elif output_method is None:
        output_scaler = StandardScaler()
    else:
        raise NotImplementedError(f"{output_method} not implemented.")

    train_inputs = []
    train_outputs = []

    data_overview = {}

    output_columns = np.array(pd.read_csv(OUTPUT_FEATURES_PATH, index_col=0)).reshape(
        -1
    )

    n_output_cols = len(output_columns)

    for filepath in filepaths:

        data = np.load(filepath)

        # Split into input (X) and output/target (y)
        X = data[:, n_output_cols:].copy()
        y = data[:, 0:n_output_cols].copy()

        # If we have a one-hot encoding of categorical labels, shape of y stays
        # the same, otherwise it is reshaped.
        # TODO: Make a better test
        # if classification and len(np.unique(y, axis=-1)) > 2:
        #     pass
        # else:
        if not onehot_encode_target:
            y = y.reshape(-1, 1)

        if "train" in filepath:
            train_inputs.append(X)
            train_outputs.append(y)
            category = "train"
        elif "test" in filepath:
            category = "test"
        elif "calibrate" in filepath:
            category = "calibrate"

        data_overview[filepath] = {"X": X, "y": y, "category": category}

    X_train = np.concatenate(train_inputs)
    y_train = np.concatenate(train_outputs)

    # Fit a scaler to the training data
    input_scaler = input_scaler.fit(X_train)

    if not classification:
        output_scaler = output_scaler.fit(y_train)

    for filepath in data_overview:

        # Scale inputs
        if input_method == None:
            X = data_overview[filepath]["X"]
        else:
            X = input_scaler.transform(data_overview[filepath]["X"])

        # Scale outputs
        if output_method == None or classification:
            y = data_overview[filepath]["y"]
        else:
            y = output_scaler.transform(data_overview[filepath]["y"])

        # Save X and y into a binary file
        np.savez(
            DATA_SCALED_PATH
            / (
                os.path.basename(filepath).replace(
                    data_overview[filepath]["category"] + ".npy",
                    data_overview[filepath]["category"] + "-scaled.npz",
                )
            ),
            X=X,
            y=y,
        )

        joblib.dump(input_scaler, INPUT_SCALER_PATH)
        joblib.dump(output_scaler, OUTPUT_SCALER_PATH)


if __name__ == "__main__":

    np.random.seed(2020)

    scale(sys.argv[1])
