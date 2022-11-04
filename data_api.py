import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components


###---------- load data --------
def load_all_data(sample_size=100):
    data = pd.read_csv("p7_clean_dataset_for_ml.csv", nrows=sample_size)
    # y_pred_test_export = pd.read_csv("data/y_pred_test_export.csv")

    # Preparation des données age
    data['DAYS_BIRTH'] = data['DAYS_BIRTH'] / -365
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    data['age_bins'] = (pd.cut(data['DAYS_BIRTH'], bins=bins)).astype(str)

    train_set = pd.read_csv('application_train.csv', nrows=sample_size)

    return data, train_set  # , y_pred_test_export,


# ------------- Affichage des infos client en HTML------------------------------------------
def display_client_info(id, revenu, age, nb_ann_travail):
    components.html(
        """
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
        <div class="card" style="width: 500px; margin:10px;padding:0">
            <div class="card-body">
                <h5 class="card-title">Info Client</h5>
    
                <ul class="list-group list-group-flush">
                    <li class="list-group-item"> <b>ID                           : </b>""" + id + """</li>
                <li class="list-group-item"> <b>Revenu                       : </b>""" + revenu + """</li>
                <li class="list-group-item"> <b>Age                          : </b>""" + age + """</li>
                <li class="list-group-item"> <b>Nombre d'années travaillées  : </b>""" + nb_ann_travail + """</li>
            </ul>
        </div>
    </div>
    """,
        height=300

    )