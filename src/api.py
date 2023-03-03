#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API for Erdre.

Author:
    Erik Johannes Husom

Created:
    2021-11-10 Wednesday 11:22:39

"""
import json
import os
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

import flask
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import requests
import yaml
from flask_restful import Api, Resource, reqparse
from plotly.subplots import make_subplots

from config import METRICS_FILE_PATH, API_MODELS_PATH
from evaluate import plot_prediction
from virtualsensor import VirtualSensor

app = flask.Flask(__name__)
api = Api(app)
app.config["DEBUG"] = True


@app.route("/")
def home():
    return flask.render_template("index.html")


@app.route("/create_model_form")
def create_model_form():

    models = get_models()

    return flask.render_template(
        "create_model_form.html", length=len(models), models=models
    )


@app.route("/inference")
def inference():

    models = get_models()

    return flask.render_template("inference.html", models=models)


@app.route("/result")
def result(plot_div):
    return flask.render_template("result.html", plot=flask.Markup(plot_div))


@app.route("/prediction")
def prediction():
    return flask.render_template("prediction.html")


def get_models():

    try:
        models = json.load(open(API_MODELS_PATH))
    except:
        models = {}

    return models


class CreateModel(Resource):
    def get(self):

        try:
            models = json.load(open(API_MODELS_PATH))
            return models, 200
        except:
            return {"message": "No models exist."}, 401

    def post(self):

        try:
            # Read params file
            params_file = flask.request.files["file"]
            params = yaml.safe_load(params_file)
        except:
            params = yaml.safe_load(open("params_default.yaml"))
            params["profile"]["dataset"] = flask.request.form["dataset"]
            params["clean"]["target"] = flask.request.form["target"]
            params["train"]["learning_method"] = flask.request.form["learning_method"]
            params["split"]["train_split"] = (
                float(flask.request.form["train_split"]) / 10
            )
            print(params)

        # Create dict containing all metadata about models
        model_metadata = {}
        # The ID of the model is given an UUID.
        model_id = str(uuid.uuid4())
        model_metadata["id"] = model_id
        model_metadata["params"] = params

        # Save params to be used by DVC when creating model.
        yaml.dump(params, open("params.yaml", "w"), allow_unicode=True)

        # Run DVC to create model.
        subprocess.run(["dvc", "repro"], check=True)

        metrics = json.load(open(METRICS_FILE_PATH))
        model_metadata["metrics"] = metrics

        try:
            models = json.load(open(API_MODELS_PATH))
        except:
            models = {}

        models[model_id] = model_metadata
        print(models)

        json.dump(models, open(API_MODELS_PATH, "w+"))

        return flask.redirect("create_model_form")


class InferGUI(Resource):
    def get(self):
        return 200

    def post(self):

        model_id = flask.request.form["id"]
        csv_file = flask.request.files["file"]
        inference_df = pd.read_csv(csv_file)

        models = get_models()
        model = models[model_id]
        params = model["params"]
        target = params["clean"]["target"]

        vs = VirtualSensor(params_file=params)

        # Run DVC to fetch correct assets.
        subprocess.run(["dvc", "repro"], check=True)

        y_pred = vs.run_virtual_sensor(inference_df=inference_df)
        window_size = params["sequentialize"]["window_size"]

        x = np.linspace(0, y_pred.shape[0] - 1, y_pred.shape[0])

        if flask.request.form.get("plot"):
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            if len(y_pred.shape) > 1:
                y_pred = y_pred[:, -1].reshape(-1)

            # If the input data contains the target value, show it in the plot
            try:
                original_target_values = inference_df[params["clean"]["target"]][
                    ::window_size
                ]
                if len(original_target_values.shape) > 1:
                    original_target_values = original_target_values[:, -1].reshape(-1)

                original_target_values = original_target_values[-len(y_pred) :]
                x_orig = np.linspace(
                    0,
                    original_target_values.shape[0] - 1,
                    original_target_values.shape[0],
                )

                fig.add_trace(
                    go.Scatter(x=x_orig, y=original_target_values, name="original"),
                    secondary_y=False,
                )
            except:
                pass

            fig.add_trace(
                go.Scatter(x=x, y=y_pred, name="pred"),
                secondary_y=False,
            )
            fig.update_layout(title_text="Original vs predictions")
            fig.update_xaxes(title_text="time step")
            fig.update_yaxes(title_text="target variable", secondary_y=False)
            fig.write_html("src/templates/prediction.html")

            return flask.redirect("prediction")
        else:
            x = x.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

            output_data = np.concatenate([x, y_pred], axis=1)
            output_data = output_data.tolist()

            # Put output data into JSON schema
            output = {}
            output["param"] = {"modeluid": model_id}
            output["scalar"] = {"headers": ["date", target], "data": output_data}

            return output


class Infer(Resource):
    def get(self):
        return 200

    def post(self):

        input_json = flask.request.get_json()
        model_id = str(input_json["param"]["modeluid"])

        inference_df = pd.DataFrame(
            input_json["scalar"]["data"],
            columns=input_json["scalar"]["headers"],
        )
        inference_df.set_index("date", inplace=True)

        models = get_models()
        model = models[model_id]
        params = model["params"]
        target = params["clean"]["target"]

        vs = VirtualSensor(params_file=params)

        # Run DVC to fetch correct assets.
        subprocess.run(["dvc", "repro"], check=True)

        y_pred = vs.run_virtual_sensor(inference_df=inference_df)
        window_size = params["sequentialize"]["window_size"]

        x = np.linspace(0, y_pred.shape[0] - 1, y_pred.shape[0])
        x = x.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

        output_data = np.concatenate([x, y_pred], axis=1)
        output_data = output_data.tolist()

        # Put output data into JSON schema
        output = {}
        output["param"] = {"modeluid": model_id}
        output["scalar"] = {"headers": ["date", target], "data": output_data}

        return output


if __name__ == "__main__":

    api.add_resource(CreateModel, "/create_model")
    api.add_resource(InferGUI, "/infer_gui")
    api.add_resource(Infer, "/infer")
    app.run(host="0.0.0.0")
