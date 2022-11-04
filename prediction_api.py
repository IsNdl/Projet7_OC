import pickle
import pandas as pd
# import numpy as np


def load_dataset(sample_size):
    data = pd.read_csv("modules_api/df_final.csv", nrows=sample_size)
    return data


'''Load fitted preprocessor '''


def load_preprocessor():
    preproc = pickle.load(open("modules_api/preprocessor.sav", 'rb'))
    return preproc


'''Transformer X avec le preprocessor '''


def preprocessing(X):
    preprocessor = load_preprocessor()
    return preprocessor.transform(X)


'''Charger un modele entrainné '''


def load_model():
    model = pickle.load(open("sgd_model.pkl", 'rb'))
    return model


'''Prédire un client avec un modele  '''


def predict_client(model, X):
    X = X.drop(['SK_ID_CURR'], axis=1)
    X_transformed = preprocessing(X)
    model = load_model(model)
    y_pred = model.predict(X_transformed)
    y_proba = model.predict_proba(X_transformed)
    return y_pred, y_proba


'''Prédire un client par son ID dans le dataset '''


def predict_client_par_ID(id_client):
    sample_size = 20000
    data_set = load_dataset(sample_size)
    client = data_set[data_set['SK_ID_CURR'] == id_client].drop(['SK_ID_CURR', 'TARGET'], axis=1)
    print(client)
    client_preproceced = preprocessing(client)
    model = load_model()
    y_pred = model.predict(client_preproceced)
    y_proba = model.predict_proba(client_preproceced)
    return y_pred, y_proba