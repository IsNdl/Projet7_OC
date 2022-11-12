# -*- coding: utf-8 -*-
# To run : flask run

# Load librairies
import pandas as pd
import pickle
from flask import Flask, jsonify
# from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.metrics import fbeta_score

"""
def credit_score(y_true, y_pred, tp_cost=0, tn_cost=1, fp_cost=0, fn_cost=-10):
    mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = mat.ravel()
    cost = (tp*tp_cost + tn*tn_cost + fn*fn_cost + fp*fp_cost)/y_true.size
    best = ((tn+fp)*tn_cost + (tp+fn)*tp_cost)/y_true.size
    baseline = ((tn+fp)*tn_cost + (tp+fn) * fn_cost)/y_true.size
    score = (cost-baseline)/(best-baseline)
    return score
"""

# Load the data
path = "C:/Users/I-NL/Documents/Projet7_OC/"
data = pd.read_csv(path+"p7_clean_dataset_for_ml.csv",
                   nrows=200)
threshold = 0.458607  # Optimal threshold determined in notebook
y_true = data[["TARGET", "SK_ID_CURR"]]
# y_true.ravel()
# credit_scorer = make_scorer(credit_score, greater_is_better=True)
# Load the model
model = pickle.load(open(path+"pipeline_sgd_model.pkl", 'rb'))

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
    y_true_client = y_true[y_true["SK_ID_CURR"] == id_client].drop(['TARGET'], axis=1)
    # y_true.ravel()
    print(client)
    proba = model.predict_proba(client)[:, 1][0]  # or [0][0]
    y_pred = model.predict(client)
    y_pred.ravel()
    score = fbeta_score(y_true_client, y_pred, beta=2)
    return jsonify({"probability": proba, "score": score})


@app.route("/score/<id_client>", methods=['GET'])
def score_client_with_id(id_client):
    """
    Scoring of a client, identified by ID
    input : id_client, str
    return: score : custom score related to probability to repay loan, str in json file
    """
    id_client = int(float(id_client))
    client = data[data['SK_ID_CURR'] == id_client].drop(['SK_ID_CURR', 'TARGET'], axis=1)
    # print(client)
    y_pred = model.predict(client)
    score = fbeta_score(y_true, y_pred, beta=2)
    return jsonify(score)


if __name__ == '__main__':
    app.run()
