#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Erdre ML pipeline for classification.

Author:
    Erik Johannes Husom

Created:
    2021-08-11

"""
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

import pandas as pd
import yaml


class TestCNCMilling(unittest.TestCase):
    """Test ML pipeline using the CNC Milling Tool Wear data set.

    Asserts that the accuracy / R2-score is above a certain threshold for
    simple test cases.

    """

    def test_cnc_milling_classification_cnn(self):
        """Test pipeline on CNC milling data with classification and CNN."""

        experiment = RunExperiment(
            target="tool_condition",
            classification=True,
            onehot_encode_target=False,
            learning_method="cnn",
        )

        accuracy = experiment.run()

        assert accuracy > 0.9

    def test_cnc_milling_classification_xgboost(self):
        """Test pipeline on CNC milling data with classification and XGBoost."""

        experiment = RunExperiment(
            target="tool_condition",
            classification=True,
            onehot_encode_target=False,
            learning_method="cnn",
        )

        accuracy = experiment.run()

        assert accuracy > 0.9

    def test_cnc_milling_regression_cnn(self):
        """Test pipeline on CNC milling data with regression and CNN."""

        experiment = RunExperiment(
            target="X1_ActualPosition",
            classification=False,
            onehot_encode_target=False,
            learning_method="cnn",
        )

        r2_score = experiment.run()

        assert r2_score > 0.8


class RunExperiment:
    def __init__(
        self,
        target,
        classification=False,
        onehot_encode_target=False,
        learning_method="cnn",
    ):

        self.target = target
        self.learning_method = learning_method
        self.classification = classification
        self.onehot_encode_target = onehot_encode_target

    def run(self):
        """Run ML pipeline for test case.

        Returns:
            metric (float): The evaluation metric for the performance of the
                model created using the test case.

        """

        prepare_dataset()
        self.create_params_file()

        run_experiment = subprocess.Popen(["dvc", "repro"], cwd="../")
        run_experiment.wait()

        restore_params_file()

        with open("../assets/metrics/metrics.json", "r") as infile:
            metrics = json.load(infile)

        if self.classification:
            metric = metrics["accuracy"]
        else:
            metric = metrics["r2"]

        return metric

    def create_params_file(self):
        """Create parameter file for test cases."""

        params_string = """
profile:
    dataset: cnc_milling

clean:
    target: tool_condition
    classification: True
    onehot_encode_target: False
    combine_files: True
    percentage_zeros_threshold: 1.0
    correlation_metric: pearson
    input_max_correlation_threshold: 1.0

featurize:
    features:
    add_rolling_features: False
    rolling_window_size: 100
    remove_features:
    target_min_correlation_threshold: 0.0
    train_split: 0.6
    shuffle_files: False
    calibrate_split: 0.0

split:
    train_split: 0.6
    shuffle_files: False
    calibrate_split: 0.0

scale:
    input: standard
    output: minmax

sequentialize:
    window_size: 20
    target_size: 1
    shuffle_samples: True

train:
    learning_method: xgboost
    n_epochs: 10
    batch_size: 512
    kernel_size: 5
    early_stopping: False
    patience: 10

evaluate:
        """

        shutil.move("../params.yaml", "../params.yaml.bak")

        params = yaml.safe_load(params_string)
        params["clean"]["target"] = self.target
        params["clean"]["classification"] = self.classification
        params["clean"]["onehot_encode_target"] = self.onehot_encode_target
        params["train"]["learning_method"] = self.learning_method

        with open("../params.yaml", "w") as outfile:
            yaml.dump(params, outfile)


def prepare_dataset():
    """Download data set and place it in correct folder.

    Will skip downloading if data set already exists in correct folder.

    """

    dataset_name = "cnc_milling"
    dataset_path = Path("../assets/data/raw/" + dataset_name)
    cnc_milling_url = "https://raw.githubusercontent.com/ejhusom/cnc_milling_tool_wear/master/data/experiment_{:02d}.csv"
    n_files = 10

    files_missing = []

    for i in range(1, n_files + 1):
        if os.path.exists(dataset_path / "{:02d}.csv".format(i)):
            files_missing.append(False)
        else:
            files_missing.append(True)

    if any(files_missing):
        print("Data set not present, downloading files...")
        dataset_path.mkdir(parents=True, exist_ok=True)

        for i in range(1, n_files + 1):
            df = pd.read_csv(cnc_milling_url.format(i))
            df.to_csv(dataset_path / "{:02d}.csv".format(i))


def restore_params_file():
    """Restore original params file."""

    shutil.move("../params.yaml.bak", "../params.yaml")


if __name__ == "__main__":

    unittest.main()
