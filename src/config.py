#!/usr/bin/env python3
"""Global parameters for project.

Example:

    >>> from config import *
    >>> file = DATA_PATH / "filename.txt"

Author:   Erik Johannes Husom
Created:  2020-09-16

"""

from pathlib import Path

# PARAMETERS

# Deep learning methods
DL_METHODS = ["dnn", "cnn", "rnn"]

# Non-deep learning methods
NON_DL_METHODS = [
    "dt",
    "rf",
    "explainableboosting",
    "xgboost",
    "lda",
    "qda",
    "svm",
    "linearregression",
    "ridgeregression",
    "lasso",
    "lars",
    "bayesianridge",
    "ardregression",
    "elasticnet",
    "sgd",
    "kneighbors",
    "gb",
    "gradientboosting",
]

# Learning methods that can process sequences
SEQUENCE_LEARNING_METHODS = ["cnn", "rnn"]

# Learning methods that cannot process sequences
NON_SEQUENCE_LEARNING_METHODS = [
    "dnn",
    "dt",
    "rf",
    "xgboost",
    "explainableboosting",
    "lda",
    "qda",
    "svm",
    "linearregression",
    "ridgeregression",
    "lasso",
    "lars",
    "bayesianridge",
    "ardregression",
    "elasticnet",
    "sgd",
    "kneighbors",
    "gb",
    "gradientboosting",
]


# PATHS

PARAMS_FILE_PATH = Path("./params.yaml")
"""Path to params file."""

ASSETS_PATH = Path("./assets")
"""Path to all assets of project."""

PROFILE_PATH = ASSETS_PATH / "profile"
"""Path to profiling reports."""

PROFILE_JSON_PATH = PROFILE_PATH / "profile.json"
"""Path to profiling report in JSON format."""

FEATURES_PATH = ASSETS_PATH / "features"
"""Path to files containing input and output features."""

INPUT_FEATURES_PATH = FEATURES_PATH / "input_columns.csv"
"""Path to file containing input features."""

INPUT_FEATURES_SEQUENCE_PATH = FEATURES_PATH / "input_columns_sequence.csv"
"""Path to file containing input features."""

OUTPUT_FEATURES_PATH = FEATURES_PATH / "output_columns.csv"
"""Path to file containing output features."""

REMOVABLE_FEATURES = FEATURES_PATH / "removable_features.csv"
"""Path to file containing removable features."""

DATA_PATH = ASSETS_PATH / "data"
"""Path to data."""

DATA_PATH_RAW = DATA_PATH / "raw"
"""Path to raw data."""

DATA_FEATURIZED_PATH = DATA_PATH / "featurized"
"""Path to data that is has added features."""

DATA_CLEANED_PATH = DATA_PATH / "cleaned"
"""Path to data that is cleaned."""

DATA_SEQUENTIALIZED_PATH = DATA_PATH / "sequentialized"
"""Path to data that is split into sequences."""

DATA_SPLIT_PATH = DATA_PATH / "split"
"""Path to data that is split into train and test set."""

DATA_SCALED_PATH = DATA_PATH / "scaled"
"""Path to scaled data."""

DATA_COMBINED_PATH = DATA_PATH / "combined"
"""Path to combined data, ready for training."""

MODELS_PATH = ASSETS_PATH / "models"
"""Path to models."""

MODELS_FILE_PATH = MODELS_PATH / "model.h5"
"""Path to model file."""

API_MODELS_PATH = ASSETS_PATH / "models_api.json"

METRICS_PATH = ASSETS_PATH / "metrics"
"""Path to folder containing metrics file."""

METRICS_FILE_PATH = METRICS_PATH / "metrics.json"
"""Path to file containing metrics."""

PREDICTIONS_PATH = ASSETS_PATH / "predictions"
"""Path to folder containing predictions file."""

PREDICTIONS_FILE_PATH = PREDICTIONS_PATH / "predictions.csv"
"""Path to file containing predictions."""

PLOTS_PATH = ASSETS_PATH / "plots"
"""Path to folder plots."""

PREDICTION_PLOT_PATH = PLOTS_PATH / "prediction.png"
"""Path to file containing prediction plot."""

INTERVALS_PLOT_PATH = PLOTS_PATH / "intervals.png"
"""Path to file containing intervals plot."""

TRAININGLOSS_PLOT_PATH = PLOTS_PATH / "trainingloss.png"
"""Path to file containing training loss plot."""

SCALER_PATH = ASSETS_PATH / "scalers"
"""Path to folder containing scalers."""

INPUT_SCALER_PATH = SCALER_PATH / "input_scaler.z"
"""Path to input scaler."""

OUTPUT_SCALER_PATH = SCALER_PATH / "output_scaler.z"
"""Path to output scaler."""
