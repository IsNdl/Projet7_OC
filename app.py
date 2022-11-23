# -*- coding: utf-8 -*-
# To run : flask run

# Load librairies
import pandas as pd
import pickle
from flask import Flask, jsonify

# Load the data
data = pd.read_csv("small_dataset_for_api.csv",
                   nrows=200)
x = data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = data['TARGET']
# Load the model
model = pickle.load(open("pipeline_sgd_model.pkl", 'rb'))
# model = pickle.load(open(path+"lgbm_model.pkl", 'rb'))
model.fit(x, y)

# API
app = Flask(__name__)


@app.route("/")
def loaded():
    return "API, model and data loaded"


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
    proba = model.predict_proba(client)[:, 1][0]  # [0][1]  # or [0][0]  # or [:, 1][0]
    return jsonify(str(proba))


if __name__ == '__main__':
    app.run(debug=True)
