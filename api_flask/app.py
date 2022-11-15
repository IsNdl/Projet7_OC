# -*- coding: utf-8 -*-
# To run : flask run

# Load librairies
import pandas as pd
import pickle
from flask import Flask, jsonify

# Load the data
# path = "C:/Users/I-NL/Documents/Projet7_OC/"
data = pd.read_csv("small_dataset_for_api.csv",
                   nrows=200)
# Load the model
model = pickle.load(open("pipeline_sgd_model.pkl", 'rb'))


# API
app = Flask(__name__)


@app.route("/")
def loaded():
    return "API, model and data loaded"


# @app.route('/{id_client}', methods=['GET'])
# def get_id_client():
    # id_client = request.args.get("id_client", None)
    # id_client à taper sur la barre d'adresse
    # reprendre ID avec methode get --> int
    #  return jsonify(id_client)


@app.route("/predict/<id_client>", methods=['GET'])
def predict_client_with_id(id_client):
    """
    Prediction of default payment probability of a client, identified by ID
    input : id_client, str
    return: proba_json : probability of client to not repay loan, str in json file
    """
    id_client = int(float(id_client))
    client = data[data['SK_ID_CURR'] == id_client].drop(['SK_ID_CURR', 'TARGET'], axis=1)
    print(client)
    proba = model.predict_proba(client)[:, 1][0]  # or [0][0]
    return jsonify(proba)


if __name__ == '__main__':
    app.run()
