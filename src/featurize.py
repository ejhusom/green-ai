#!/usr/bin/env python3
"""Clean up inputs and add features to data set.

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype
from scipy.signal import find_peaks
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from config import (
    DATA_FEATURIZED_PATH,
    FEATURES_PATH,
    INPUT_FEATURES_PATH,
    OUTPUT_FEATURES_PATH,
    PROFILE_PATH,
)
from preprocess_utils import find_files, move_column


def featurize(dir_path="", inference=False, inference_df=None):
    """Clean up inputs and add features to data set.

    Args:
        dir_path (str): Path to directory containing files.
        inference (bool): When creating a virtual sensor, the
            results should be saved to file for more efficient reruns of the
            pipeline. When running the virtual sensor, there is no need to save
            these intermediate results to file.

    """

    # Load parameters
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    features = params["featurize"]["variables_to_include"]
    remove_features = params["featurize"]["remove_features"]
    target = params["clean"]["target"]

    if inference:
        output_columns = np.array(
            pd.read_csv(OUTPUT_FEATURES_PATH, index_col=0, dtype=str)
        ).reshape(-1)

        df = _featurize(inference_df, features, remove_features, params, output_columns)

        input_columns = pd.read_csv(INPUT_FEATURES_PATH, index_col=0)
        input_columns = [feature for feature in input_columns["0"]]

        for expected_col in input_columns:
            if expected_col not in df.columns:
                raise ValueError(f"Variable {expected_col} not in input data.")

        for actual_col in df.columns:
            if actual_col not in input_columns:
                del df[actual_col]

        return df

    filepaths = find_files(dir_path, file_extension=".csv")

    DATA_FEATURIZED_PATH.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.mkdir(parents=True, exist_ok=True)

    output_columns = np.array(
        pd.read_csv(OUTPUT_FEATURES_PATH, index_col=0, dtype=str)
    ).reshape(-1)

    for filepath in filepaths:

        # Read csv
        df = pd.read_csv(filepath, index_col=0)

        df = _featurize(df, features, remove_features, params, output_columns)

        # Move target column(s) to the beginning of dataframe
        for col in output_columns[::-1]:
            df = move_column(df, column_name=col, new_idx=0)

        np.save(
            DATA_FEATURIZED_PATH
            / os.path.basename(filepath).replace("cleaned.csv", "featurized.npy"),
            df.to_numpy(),
        )

    # Save list of features used
    input_columns = [col for col in df.columns if col not in output_columns]
    pd.DataFrame(input_columns).to_csv(INPUT_FEATURES_PATH)


def _featurize(df, features, remove_features, params, output_columns):
    """Process individual DataFrames."""

    # If no features are specified, use all columns as features
    if not isinstance(features, list):
        features = df.columns

    # Check if wanted features from params.yaml exists in the data
    for feature in features:
        if feature not in df.columns:
            print(f"Feature {feature} not found!")

    for col in df.columns:
        # Remove feature from input. This is useful in the case that a raw
        # feature is used to engineer a feature, but the raw feature itself
        # should not be a part of the input.
        if (col not in features) and (col not in output_columns):
            del df[col]

        # Remove feature if it is non-numeric
        elif not is_numeric_dtype(df[col]):
            print(f"Removing feature {col} because it is non-numeric.")
            del df[col]

        # Convert boolean feature to integer
        elif df.dtypes[col] == bool:
            print(f"Converting feature {col} from boolean to integer.")
            df[col] = df[col].replace({True: 1, False: 0})

    df = compute_rolling_features(df, params, ignore_columns=output_columns)

    if isinstance(remove_features, list):
        for col in remove_features:
            if col in df:
                del df[col]

    return df


def compute_rolling_features(df, params, ignore_columns=None):
    """
    This function adds features to the input data, based on the arguments
    given in the features-list.

    Available features (TODO):

    - mean
    - std
    - autocorrelation
    - abs_energy
    - absolute_maximum
    - absolute_sum_of_change

    Args:
    df (pandas DataFrame): Data frame to add features to.
    features (list): A list containing keywords specifying which features to
        add.

    Returns:
        df (pandas DataFrame): Data frame with added features.

    """

    columns = [col for col in df.columns if col not in ignore_columns]

    if params["featurize"]["use_all_engineered_features_on_all_variables"]:
        features_to_add = [p for p in params["featurize"] if p.startswith("add_")]

        for feature in features_to_add:
            params["featurize"][feature] = columns

    if isinstance(params["featurize"]["add_sum"], list):
        for var in params["featurize"]["add_sum"]:
            # df[f"{var}_sum"] = (
            #     df[var].rolling(params["featurize"]["rolling_window_size_sum"]).sum()
            # )
            df = pd.concat([
                pd.Series(
                    df[var].rolling(params["featurize"]["rolling_window_size_sum"]).sum(),
                    name=f"{var}_sum"
                    ), 
                df], axis=1
            )

    if isinstance(params["featurize"]["add_gradient"], list):
        for var in params["featurize"]["add_gradient"]:
            df = pd.concat([pd.Series(np.gradient(df[var]), name=f"{var}_gradient"), df], axis=1)
            # print(df)
            # df[f"{var}_gradient"] = np.gradient(df[var])

    if isinstance(params["featurize"]["add_mean"], list):
        for var in params["featurize"]["add_mean"]:
            # df[f"{var}_mean"] = (
            #     df[var].rolling(params["featurize"]["rolling_window_size_mean"]).mean()
            # )
            df = pd.concat([
                    pd.Series(
                        df[var].rolling(params["featurize"]["rolling_window_size_mean"]).mean(),
                        name=f"{var}_mean"),
                df], axis=1
            )

    if isinstance(params["featurize"]["add_maximum"], list):
        for var in params["featurize"]["add_maximum"]:
            # df[f"{var}_maximum"] = (
            #     df[var]
            #     .rolling(params["featurize"]["rolling_window_size_max_min"])
            #     .max()
            # )
            df = pd.concat([
                    pd.Series(
                        df[var].rolling(params["featurize"]["rolling_window_size_max_min"]).max(),
                        name=f"{var}_max"),
                df], axis=1
            )

    if isinstance(params["featurize"]["add_minimum"], list):
        for var in params["featurize"]["add_minimum"]:
            # minimum = (
            #     df[var]
            #     .rolling(params["featurize"]["rolling_window_size_max_min"])
            #     .min()
            # )
            df = pd.concat([
                    pd.Series(
                        df[var].rolling(params["featurize"]["rolling_window_size_max_min"]).max(),
                        name=f"{var}_min"),
                df], axis=1
            )

    if isinstance(params["featurize"]["add_min_max_range"], list):
        for var in params["featurize"]["add_min_max_range"]:
            maximum = (
                df[var]
                .rolling(params["featurize"]["rolling_window_size_max_min"])
                .max()
            )
            minimum = (
                df[var]
                .rolling(params["featurize"]["rolling_window_size_max_min"])
                .min()
            )
            # df[f"{var}_min_max_range"] = maximum - minimum
            df = pd.concat([
                    pd.Series(
                        maximum - minimum,
                        name=f"{var}_min_max_range"),
                df], axis=1
            )

    if isinstance(params["featurize"]["add_slope"], list):
        for var in params["featurize"]["add_slope"]:
            # df[f"{var}_slope"] = calculate_slope(df[var])
            df = pd.concat([
                    pd.Series(
                        calculate_slope(df[var]),
                        name=f"{var}_slope"),
                df], axis=1
            )

    if isinstance(params["featurize"]["add_slope_sin"], list):
        for var in params["featurize"]["add_slope_sin"]:
            slope = calculate_slope(df[var])
            # df[f"{var}_slope_sin"] = np.sin(slope)
            df = pd.concat([
                    pd.Series(
                        np.sin(calculate_slope(df[var])),
                        name=f"{var}_slope_sin"),
                df], axis=1
            )

    if isinstance(params["featurize"]["add_slope_cos"], list):
        for var in params["featurize"]["add_slope_cos"]:
            slope = calculate_slope(df[var])
            # df[f"{var}_slope_cos"] = np.cos(slope)
            df = pd.concat([
                    pd.Series(
                        np.cos(calculate_slope(df[var])),
                        name=f"{var}_slope_cos"),
                df], axis=1
            )

    if isinstance(params["featurize"]["add_standard_deviation"], list):
        for var in params["featurize"]["add_standard_deviation"]:
            # df[f"{var}_standard_deviation"] = (
            #     df[var]
            #     .rolling(params["featurize"]["rolling_window_size_standard_deviation"])
            #     .std()
            # )
            df = pd.concat([
                    pd.Series(
                        df[var]
                        .rolling(params["featurize"]["rolling_window_size_standard_deviation"])
                        .std(),
                        name=f"{var}_standard_deviation"),
                df], axis=1
            )

    if isinstance(params["featurize"]["add_variance"], list):
        for var in params["featurize"]["add_variance"]:
            # df[f"{var}_variance"] = np.var(df[var])
            df = pd.concat([
                    pd.Series(
                        np.var(df[var]),
                        name=f"{var}_variance"),
                df], axis=1
            )

    if isinstance(params["featurize"]["add_peak_frequency"], list):
        for var in params["featurize"]["add_peak_frequency"]:
            # df[f"{var}_peak_frequency"] = calculate_peak_frequency(df[var])
            df = pd.concat([
                    pd.Series(
                        calculate_peak_frequency(df[var]),
                        name=f"{var}_peak_frequency"),
                df], axis=1
            )

    df = df.dropna()

    return df


def calculate_peak_frequency(series, rolling_mean_window=200):
    """Calculate peak frequency of a time series.

    Args:
        series (array): Time series in where to find peaks.
        rolling_mean_window (int): The size of the rolling window.

    Returns:
        freq (Series): A Pandas Series of the rolling mean of the peak
            frequency.

    """

    peaks_indices = find_peaks(series, distance=5)[0]
    peaks = np.zeros(len(series))
    peaks[peaks_indices] = 1

    freq = []
    frequency = 0
    counter = 0

    for peak in peaks:

        if peak == 1:
            frequency = 10 / counter
            counter = 0
        else:
            counter += 1

        freq.append(frequency)

    freq = pd.Series(freq).rolling(rolling_mean_window).mean()

    return freq


def calculate_slope(series, shift=2, rolling_mean_window=1, absvalue=False):
    """Calculate slope.

    Args:
        series (array): Data for slope calculation.
        shift (int): How many steps backwards to go when calculating the slope.
            For example: If shift=2, the slope is calculated from the data
            point two time steps ago to the data point at the current time
            step.
        rolling_mean_window (int): Window for calculating rolling mean.

    Returns:
        slope (array): Array of slope angle.

    """

    v_dist = series - series.shift(shift)
    h_dist = 0.1 * shift

    slope = np.arctan(v_dist / h_dist)

    if absvalue:
        slope = np.abs(slope)

    slope = slope.rolling(rolling_mean_window).mean()

    return slope


# ===============================================
# TODO: Automatic encoding of categorical input variables
# def encode_categorical_input_variables():

#     # Read all data to fit one-hot encoder
#     dfs = []

#     for filepath in filepaths:
#         df = pd.read_csv(filepath, index_col=0)
#         dfs.append(df)

#     combined_df = pd.concat(dfs, ignore_index=True)

#     # ct = ColumnTransformer([('encoder', OneHotEncoder(), [38])],
#     #         remainder='passthrough')

#     # ct.fit(combined_df)

#     categorical_variables = find_categorical_variables()

#     # Remove target and variables that was removed in the cleaning process
#     categorical_variables = [
#         var
#         for var in categorical_variables
#         if var in combined_df.columns and var != target
#     ]

#     print(combined_df)
#     print(f"Cat: {categorical_variables}")
#     print(combined_df[categorical_variables])

#     column_transformer = ColumnTransformer(
#         [("encoder", OneHotEncoder(), categorical_variables)], remainder="passthrough"
#     )

    # combined_df = column_transformer.fit_transform(combined_df)

    # print(combined_df)
    # print(combined_df.shape)
    # print(combined_df[categorical_variables])

    # categorical_encoder = OneHotEncoder()
    # categorical_encoder.fit(combined_df)
    # ===============================================

def find_categorical_variables():
    """Find categorical variables based on profiling report.

    Returns:
        categorical_variables (list): List of categorical variables.

    """

    with open("params.yaml", "r", encoding="UTF-8") as infile:
        params = yaml.safe_load(infile)["clean"]

    target = params["target"]

    with open(PROFILE_PATH / "profile.json", "r", encoding="UTF-8") as infile:
        profile_json = json.load(infile)

    variables = list(profile_json["variables"].keys())
    correlations = profile_json["correlations"]["pearson"]

    categorical_variables = []

    for var in variables:

        try:
            n_categories = profile_json["variables"][var]["n_category"]
            categorical_variables.append(var)
        except:
            pass

    # categorical_variables = list(set(categorical_variables))

    # if target in categorical_variables:
    #     print("Warning related to target variable. Check profile for details.")
    #     categorical_variables.remove(target)

    return categorical_variables


if __name__ == "__main__":

    np.random.seed(2020)

    featurize(sys.argv[1])
