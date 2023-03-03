#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profile dataset.

Created:
    2021-06-30

"""
import sys

import pandas as pd
import yaml
from pandas_profiling import ProfileReport

from config import PROFILE_PATH
from preprocess_utils import find_files


def profiling(dir_path):
    """Creates a profile report of a data set.

    Reads data from a set of input files, and creates a report containing
    profiling of the data. This profiling consists of various statistical
    properties. The report is stored in two formats:

    - HTML: For visual inspection
    - JSON: For subsequent automatic processing of results

    Args:
        dir_path (str): Path to directory containing files.

    """

    # Reading the name of data set, which must be the name of subfolder of
    # 'assets/data/raw', in where to look for data.
    with open("params.yaml", "r", encoding="UTF-8") as infile:
        dataset = yaml.safe_load(infile)["profile"]["dataset"]

    # If no name of data set is given, all files present in 'assets/data/raw'
    # will be used.
    if dataset is not None:
        dir_path += "/" + dataset

    filepaths = find_files(dir_path, file_extension=".csv")

    dfs = []

    for filepath in filepaths:
        dfs.append(pd.read_csv(filepath))

    combined_df = pd.concat(dfs, ignore_index=True)

    # Generate report.
    profile = ProfileReport(
        combined_df,
        title="Profiling Analysis",
        config_file="src/profile.yaml",
        lazy=False,
        sort=None,
    )

    # Create folder for profiling report
    PROFILE_PATH.mkdir(parents=True, exist_ok=True)

    # Save report to files.
    profile.to_file(PROFILE_PATH / "profile.html")
    profile.to_file(PROFILE_PATH / "profile.json")


if __name__ == "__main__":

    profiling(sys.argv[1])
