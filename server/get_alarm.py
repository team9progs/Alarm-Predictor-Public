#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime as dt
import requests
import json

from flask import Flask, request

API_TOKEN = ""
RapidAPI_TOKEN = ""

app = Flask(__name__)


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


def get_alarm(location=None):
    with open('data/last_pred.json') as f:
        data = json.load(f)
    if (location == None):
        return data
    try:
        return {"last_model_train_time": data['last_model_train_time'],
                "last_prediction_time": data['last_prediction_time'],
                "region_forecast": {location: data['region_forecast'][location]}
                }
    except:
        return {'there is a typo in the region or it is not available'}


@app.route("/")
def home_page():
    return "<p><h2>KMA: You can get alarm forecast by location here.</h2></p>"


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route(
    "/content/api/v1/integration/get_alarm",
    methods=["POST"],
)
def alarm_endpoint():
    json_data = request.get_json()
    #     if token != API_TOKEN:
    #         raise InvalidUsage("wrong API token", status_code=403)

    result = get_alarm(json_data.get("location"))

    return result
