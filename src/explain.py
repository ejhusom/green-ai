#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explain predictions of machine learning model.

Author:
    Erik Johannes Husom

Created:
    2022-11-28 mandag 16:01:00 

"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import yaml

from joblib import load
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras import models
tf.compat.v1.disable_v2_behavior()

from config import (
    DATA_PATH,
    INPUT_FEATURES_PATH,
    INPUT_FEATURES_SEQUENCE_PATH,
    INTERVALS_PLOT_PATH,
    METRICS_FILE_PATH,
    NON_DL_METHODS,
    NON_SEQUENCE_LEARNING_METHODS,
    OUTPUT_FEATURES_PATH,
    PLOTS_PATH,
    PREDICTION_PLOT_PATH,
    PREDICTIONS_FILE_PATH,
    PREDICTIONS_PATH,
    SEQUENCE_LEARNING_METHODS
)

colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,l))
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

def explain(
        model_filepath,
        train_filepath,
        test_filepath,
    ):

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["explain"]
    params_train = yaml.safe_load(open("params.yaml"))["train"]
    window_size = yaml.safe_load(open("params.yaml"))["sequentialize"]["window_size"]
    onehot_encode_target = yaml.safe_load(open("params.yaml"))["clean"][
        "onehot_encode_target"
    ]
    number_of_background_samples = params["number_of_background_samples"]
    number_of_summary_samples = params["number_of_summary_samples"]
    learning_method = params_train["learning_method"]

    # Load training data
    train = np.load(train_filepath)
    X_train = train["X"]

    test = np.load(test_filepath)
    X_test = test["X"]
    y_test = test["y"]

    if learning_method in NON_DL_METHODS:
        model = load(model_filepath)
        y_pred = model.predict(X_test)
    else:
        model = models.load_model(model_filepath)
        y_pred = model.predict(X_test)

    # Read name of input columns
    input_columns = pd.read_csv(INPUT_FEATURES_PATH, header=None)

    # Convert the input columns into a list
    input_columns = input_columns.iloc[1:,1].to_list()

    if learning_method in NON_SEQUENCE_LEARNING_METHODS:
        if window_size > 1:
            input_columns_sequence = []

            for c in input_columns:
                for i in range(window_size):
                    input_columns_sequence.append(c + f"_{i}")

            input_columns = input_columns_sequence

        # Extract a summary of the training inputs, to reduce the amount of
        # compute needed to use SHAP
        X_train_background = shap.kmeans(X_train, number_of_background_samples)
        X_test_summary = shap.sample(X_train, number_of_summary_samples)

        # Use a SHAP explainer on the summary of training inputs
        explainer = shap.KernelExplainer(model.predict, X_train_background)

        # Single prediction explanation
        single_sample = X_test[0]
        single_shap_value = explainer.shap_values(single_sample)
        shap_values = explainer.shap_values(X_test_summary)

        if type(single_shap_value) == list:
            single_shap_value = single_shap_value[0]
            shap_values = shap_values[0]

        # SHAP force plot: Single prediction
        shap_force_plot_single = shap.force_plot(explainer.expected_value, single_shap_value,
                np.around(single_sample), show=True, feature_names=input_columns)
        shap.save_html(str(PLOTS_PATH) + "/shap_force_plot_single.html",
                shap_force_plot_single)

        # SHAP force plot: Multiple prediction
        shap_force_plot = shap.force_plot(explainer.expected_value, shap_values,
                X_test_summary, show=True, feature_names=input_columns)
        shap.save_html(str(PLOTS_PATH) + "/shap_force_plot.html", shap_force_plot)

        # SHAP summary plot
        shap.summary_plot(shap_values, X_test_summary,
                feature_names=input_columns, plot_size=(8,5), show=False)
        plt.savefig(PLOTS_PATH / "shap_summary_plot.png", bbox_inches='tight', dpi=300)
    else:
        # Extract a summary of the training inputs, to reduce the amount of
        # compute needed to use SHAP
        X_train_background = shap.sample(X_train, number_of_background_samples)
        X_test_summary = shap.sample(X_train, number_of_summary_samples)

        # Use a SHAP explainer on the summary of training inputs
        explainer = shap.DeepExplainer(model, X_train_background)

        # Single prediction explanation
        single_sample = X_test[:1]
        single_shap_value = explainer.shap_values(single_sample)[0]
        shap_values = explainer.shap_values(X_test_summary)[0]

        # SHAP force plot: Single prediction
        shap_force_plot_single = shap.force_plot(explainer.expected_value, shap_values[0,:],
                X_test_summary[0,:], feature_names=input_columns)
        shap.save_html(str(PLOTS_PATH) + "/shap_force_plot_single.html",
                shap_force_plot_single)

        X_test_summary = np.expand_dims(X_test_summary, axis=3)
        shap_values = np.expand_dims(shap_values, axis=3)
        # SHAP image plot
        # shap_image_plot = shap.image_plot(shap_values, X_test_summary,
        # shap_image_plot = image_plot(shap_values[0,:], X_test_summary[0,:],
        shap_image_plot = image_plot(shap_values[:5,:], X_test_summary[:5,:],
                show=False)
        plt.savefig(PLOTS_PATH / "shap_image_plot.png", bbox_inches='tight', dpi=300)


def image_plot(shap_values,
          pixel_values = None,
          labels = None,
          true_labels = None,
          width = 20,
          aspect = 0.2,
          hspace = 0.2,
          labelpad = None,
          cmap = red_transparent_blue,
          show = True):
    """ Plots SHAP values for image inputs.
    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shap (# samples x width x height x channels), and the
        length of the list is equal to the number of model outputs that are being explained.
    pixel_values : numpy.array
        Matrix of pixel values (# samples x width x height x channels) for each image. It should be the same
        shape as each array in the shap_values list of arrays.
    labels : list or np.ndarray
        List or np.ndarray (# samples x top_k classes) of names for each of the model outputs that are being explained.
    true_labels: list
        List of a true image labels to plot
    width : float
        The width of the produced matplotlib plot.
    labelpad : float
        How much padding to use around the model output labels.
    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """

    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        # feature_names = [shap_exp.feature_names]
        # ind = 0
        if len(shap_exp.output_dims) == 1:
            shap_values = [shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])]
        elif len(shap_exp.output_dims) == 0:
            shap_values = shap_exp.values
        else:
            raise Exception("Number of outputs needs to have support added!! (probably a simple fix)")
        if pixel_values is None:
            pixel_values = shap_exp.data
        if labels is None:
            labels = shap_exp.output_names

    multi_output = True
    if not isinstance(shap_values, list):
        multi_output = False
        shap_values = [shap_values]

    if len(shap_values[0].shape) == 3:
        shap_values = [v.reshape(1, *v.shape) for v in shap_values]
        pixel_values = pixel_values.reshape(1, *pixel_values.shape)

    # labels: (rows (images) x columns (top_k classes) )
    if labels is not None:
        if isinstance(labels, list):
            labels = np.array(labels).reshape(1, -1)

    # if labels is not None:
    #     labels = np.array(labels)
    #     if labels.shape[0] != shap_values[0].shape[0] and labels.shape[0] == len(shap_values):
    #         labels = np.tile(np.array([labels]), shap_values[0].shape[0])
    #     assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
    #     if multi_output:
    #         assert labels.shape[1] == len(shap_values), "Labels must have a column for each output in shap_values!"
    #     else:
    #         assert len(labels[0].shape) == 1, "Labels must be a vector for single output shap_values."

    label_kwargs = {} if labelpad is None else {'pad': labelpad}

    # plot our explanations
    x = pixel_values
    fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]
    fig, axes = plt.subplots(nrows=x.shape[0], ncols=len(shap_values) + 1, figsize=fig_size)
    if len(axes.shape) == 1:
        axes = axes.reshape(1, axes.size)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure we have a 2D array for grayscale
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])

        # if x_curr.max() > 1:
        #     x_curr /= 255.

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (
                    0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2])  # rgb to gray
            x_curr_disp = x_curr
        elif len(x_curr.shape) == 3:
            x_curr_gray = x_curr.mean(2)

            # for non-RGB multi-channel data we show an RGB image where each of the three channels is a scaled k-mean center
            flat_vals = x_curr.reshape([x_curr.shape[0] * x_curr.shape[1], x_curr.shape[2]]).T
            flat_vals = (flat_vals.T - flat_vals.mean(1)).T
            means = shap.kmeans(flat_vals, 3, round_values=False).data.T.reshape([x_curr.shape[0], x_curr.shape[1], 3])
            x_curr_disp = (means - np.percentile(means, 0.5, (0, 1))) / (
                    np.percentile(means, 99.5, (0, 1)) - np.percentile(means, 1, (0, 1)))
            x_curr_disp[x_curr_disp > 1] = 1
            x_curr_disp[x_curr_disp < 0] = 0
        else:
            x_curr_gray = x_curr
            x_curr_disp = x_curr

        axes[row, 0].imshow(x_curr_disp, cmap=plt.get_cmap('gray'))
        if true_labels:
            axes[row, 0].set_title(true_labels[row], **label_kwargs)
        axes[row, 0].axis('off')
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)
        for i in range(len(shap_values)):
            if labels is not None:
                axes[row, i + 1].set_title(labels[row, i], **label_kwargs)
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            axes[row, i + 1].imshow(x_curr_gray, cmap=plt.get_cmap('gray'), alpha=0.15,
                                    extent=(-1, sv.shape[1], sv.shape[0], -1))
            im = axes[row, i + 1].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
            axes[row, i + 1].axis('off')
    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal",
                      aspect=fig_size[0] / aspect)
    cb.outline.set_visible(False)
    if show:
        plt.show()


if __name__ == "__main__":

    if len(sys.argv) < 3:
        try:
            explain(
                "assets/models/model.h5",
                "assets/data/combined/train.npz",
                "assets/data/combined/test.npz"
            )
        except:
            print("Could not find model and test set.")
            sys.exit(1)
    else:
        explain(sys.argv[1], sys.argv[2], sys.argv[3])
