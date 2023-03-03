#!/usr/bin/env python3
# ============================================================================
# File:     utils.py
# Author:   Erik Johannes Husom
# Created:  2020-09-03
# ----------------------------------------------------------------------------
# Description:
# Utilities.
# ============================================================================
import argparse
import os


def get_terminal_size():
    """Get size of terminal.

    Returns
    -------
    rows, columns : (int, int)
        Number of rows and columns in current terminal window.

    """

    with os.popen("stty size", "r") as f:
        termsize = f.read().split()

    return int(termsize[0]), int(termsize[1])


def print_horizontal_line(length=None, symbol="="):
    """Print horizontal line in terminal.

    Parameters
    ----------
    length : int
        Character length of line.
    symbol : character, string of length 1
        What symbol to use for the line.

    """

    if length == None:
        try:
            _, length = get_terminal_size()
        except:
            length = 50

    print(symbol * length)


def to_bool(string):
    """Converting various values to True or False."""

    true_values = ["True", True, 1]
    false_values = ["False", False, 0]

    if string in true_values:
        return True
    elif string in false_values:
        return False
    else:
        raise ValueError("Ambigious string, could not convert to boolean.")


def parse_arguments():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print what the program does"
    )
    parser.add_argument(
        "-g", "--gnuplotlib", action="store_true", help="use gnuplotlib for plotting"
    )
    parser.add_argument("--plotly", action="store_true", help="use plotly for plotting")

    # PREPROCESSING ARGUMENT
    parser.add_argument(
        "-d",
        "--data_file",
        default="../data/20200813-2012-merged.csv",
        help="which data file to use",
    )
    parser.add_argument(
        "-s",
        "--hist_size",
        type=int,
        default=50,
        help="""how many deciseconds of history to use for power estimation,
            default=5""",
    )
    parser.add_argument(
        "--train_split", type=float, default="0.6", help="training/test ratio"
    )
    parser.add_argument(
        "--reverse_train_split",
        action="store_true",
        help="""use first part of data set for testing and second part for
            training""",
    )
    parser.add_argument(
        "--remove",
        nargs="+",
        default=[],
        help="Remove features by writing the keyword after this flag." "",
    )
    parser.add_argument(
        "-f",
        "--features",
        nargs="+",
        default="",
        help="""
    Add extra features by writing the keyword after this flag. Available:
    - nan: No features available yet
    """,
    )

    # MODEL ARGUMENTS
    parser.add_argument(
        "-n",
        "--net",
        default="cnn",
        help="which network architectyre to use, default=cnn",
    )
    parser.add_argument(
        "-e",
        "--n_epochs",
        type=int,
        default=100,
        help="number of epochs to run for NN, default=100",
    )
    parser.add_argument("-t", "--train", action="store_true", help="trains model")
    parser.add_argument(
        "-p", "--predict", action="store_true", help="predicts on test set"
    )

    # LOAD MODEL ARGUMENTS
    parser.add_argument("-m", "--model", help="loads pretrained model")
    parser.add_argument("--scaler", help="data scaler object")

    # LOAD CONFIG FROM FILE
    parser.add_argument("-c", "--config", help="load parameters from file")

    return parser.parse_args()
