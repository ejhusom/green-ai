#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean up data.

Author:
    Erik Johannes Husom

Created:
    2021-06-30

"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from config import (
    DATA_CLEANED_PATH,
    DATA_PATH_RAW,
    FEATURES_PATH,
    OUTPUT_FEATURES_PATH,
    PROFILE_PATH,
    REMOVABLE_FEATURES,
)
from preprocess_utils import find_files


def clean(dir_path=DATA_PATH_RAW, inference_df=None):
    """Clean up inputs.

    Args:
        dir_path (str): Path to directory containing files.
        inference_df (DataFrame): Dataframe containing data to use for
            inference.

    """

    # Load parameters
    dataset_name = yaml.safe_load(open("params.yaml"))["profile"]["dataset"]
    params = yaml.safe_load(open("params.yaml"))
    combine_files = params["clean"]["combine_files"]
    target = params["clean"]["target"]
    classification = params["clean"]["classification"]
    onehot_encode_target = params["clean"]["onehot_encode_target"]

    # If no name of data set is given, all files present in 'assets/data/raw'
    # will be used.
    if dataset_name is not None and inference_df is None:
        dir_path += "/" + dataset_name

    FEATURES_PATH.mkdir(parents=True, exist_ok=True)

    if inference_df is None:
        # Find removable variables from profiling report
        removable_features = parse_profile_warnings()
        pd.DataFrame(removable_features).to_csv(REMOVABLE_FEATURES)

        # Find input files
        filepaths = find_files(dir_path, file_extension=".csv")

        dfs = []

        for filepath in filepaths:
            dfs.append(pd.read_csv(filepath))
    else:
        # Remove features that should not be used with the current model.
        removable_features = np.array(
            pd.read_csv(REMOVABLE_FEATURES, index_col=0)
        ).reshape(-1)

        dfs = [inference_df]

    dfs = remove_features(dfs, removable_features)
    combined_df = pd.concat(dfs, ignore_index=True)

    if inference_df is not None:
        if target in inference_df.columns:
            del combined_df[target]

        return combined_df

    if classification:

        if onehot_encode_target and len(np.unique(combined_df[target])) > 2:
            encoder = LabelBinarizer()
        else:
            if onehot_encode_target:
                raise ValueError(
                    "Parameter 'onehot_encode_target' is set to True, but target is binary. Change parameter to False in order to use this pipeline."
                )
            encoder = LabelEncoder()

        target_col = np.array(combined_df[target]).reshape(-1)
        encoder.fit(target_col)
        # print(f"Classes: {encoder.classes_}")
        # print(f"Encoded classes: {encoder.transform(encoder.classes_)}")

        combined_df, output_columns = encode_target(encoder, combined_df, target)

        for i in range(len(dfs)):
            dfs[i], _ = encode_target(encoder, dfs[i], target)

    else:
        output_columns = [target]

    DATA_CLEANED_PATH.mkdir(parents=True, exist_ok=True)

    if combine_files:
        combined_df.to_csv(DATA_CLEANED_PATH / (os.path.basename("data-cleaned.csv")))
    else:
        for filepath, df in zip(filepaths, dfs):
            df.to_csv(
                DATA_CLEANED_PATH
                / (os.path.basename(filepath).replace(".", "-cleaned."))
            )

    pd.DataFrame(output_columns).to_csv(OUTPUT_FEATURES_PATH)


def remove_features(dfs, removable_features):
    """Read data and delete removable features.

    Args:
        dfs (list of DataFrames): Data frames to read.
        removable_features (list): Features/columns to remove.

    Returns:
        cleaned_dfs (list of DataFrames): Data frames with removeable features
            removed.

    """

    cleaned_dfs = []

    for df in dfs:

        # If the first column is an index column, remove it.
        if df.iloc[:, 0].is_monotonic:
            df = df.iloc[:, 1:]

        for column in removable_features:
            if column in df:
                del df[column]

        df.dropna(inplace=True)

        cleaned_dfs.append(df)

    return cleaned_dfs


def encode_target(encoder, df, target):
    """Encode a target variable based on a fitted encoder.

    Args:
        encoder: A fitted encoder.
        df (DataFrame): DataFrame containing the target variable.
        target (str): Name of the target variable.

    Returns:
        df (DataFrame): DataFrame with the original target variable removed,
            substituted by a onehot encoding of the variable.
        output_columns (list): List of the names of the target columns.

    """

    output_columns = []

    target_col = np.array(df[target]).reshape(-1)
    target_encoded = encoder.transform(target_col)

    del df[target]

    if len(target_encoded.shape) > 1:
        for i in range(target_encoded.shape[-1]):
            column_name = f"{target}_{i}"
            df[column_name] = target_encoded[:, i]
            output_columns.append(column_name)
    else:
        df[target] = target_encoded
        output_columns.append(target)

    return df, output_columns


def parse_profile_warnings():
    """Read profile warnings and find which columns to delete.

    Returns:
        removable_features (list): Which columns to delete from data set.

    """
    params = yaml.safe_load(open("params.yaml"))["clean"]
    target = params["target"]

    profile_json = json.load(open(PROFILE_PATH / "profile.json"))

    # In some versions of pandas-profiling, 'messages' are called 'alerts'.
    try:
        messages = profile_json["messages"]
    except:
        messages = profile_json["alerts"]

    variables = list(profile_json["variables"].keys())
    correlations = profile_json["correlations"]["pearson"]

    removable_features = []

    percentage_zeros_threshold = params["percentage_zeros_threshold"]
    input_max_correlation_threshold = params["input_max_correlation_threshold"]

    for message in messages:
        message = message.split()
        warning = message[0]
        variable = " ".join(message[message.index("column") + 1 :])

        if warning == "[CONSTANT]":
            removable_features.append(variable)
            print(f"Removed variable '{variable}' because it is constant.")
        if warning == "[ZEROS]":
            p_zeros = profile_json["variables"][variable]["p_zeros"]
            if p_zeros > percentage_zeros_threshold:
                removable_features.append(variable)
                print(
                    f"Removed variable '{variable}' because % of zeros exceeds {percentage_zeros_threshold*100}%."
                )
        if warning == "[HIGH_CORRELATION]":
            try:
                correlation_scores = correlations[variables.index(variable)]
                for correlated_variable in correlation_scores:
                    if (
                        correlation_scores[correlated_variable]
                        > input_max_correlation_threshold
                        and variable != correlated_variable
                        and variable != target
                        and correlated_variable != target
                        and variable not in removable_features
                    ):

                        removable_features.append(correlated_variable)
                        print(
                            f"Removed variable '{correlated_variable}' because of high correlation ({correlation_scores[correlated_variable]:.2f}) with variable '{variable}'."
                        )
            except:
                # Pandas profiling might not be able to compute correlation
                # score for some variables, for example some categorical
                # variables.
                print(f"{variable}: Could not find correlation score.")

    removable_features = list(set(removable_features))

    if target in removable_features:
        print("Warning related to target variable. Check profile for details.")
        removable_features.remove(target)

    return removable_features


if __name__ == "__main__":

    clean(sys.argv[1])
