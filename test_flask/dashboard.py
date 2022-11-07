import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib
import seaborn as sns
import plotly.figure_factory as ff
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import time
import streamlit.components.v1 as components

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import os

from data_api import *
from app import *

### Data
def show_data():
    st.write(data.head(10))

### Solvency
def pie_chart(thres):
    # st.write(100* (data['TARGET']>thres).sum()/data.shape[0])
    percent_sup_seuil = 100 * (data['TARGET'] > thres).sum() / data.shape[0]
    percent_inf_seuil = 100 - percent_sup_seuil
    d = {'col1': [percent_sup_seuil, percent_inf_seuil], 'col2': ['% Non Solvable', '% Solvable', ]}
    df = pd.DataFrame(data=d)
    fig = px.pie(df, values='col1', names='col2', title=' Pourcentage de solvabilité des clients di dataset')
    st.plotly_chart(fig)


def show_overview():
    st.title("Risque")
    risque_threshold = st.slider(label='Seuil de risque', min_value=0.0,
                                 max_value=1.0,
                                 value=0.5,
                                 step=0.1)
    # st.write(risque_threshold)
    pie_chart(risque_threshold)


### Graphs
def filter_graphs():
    st.subheader("Filtre des Graphes")
    col1, col2, col3 = st.beta_columns(3)
    is_educ_selected = col1.radio("Graph Education", ('non', 'oui'))
    is_statut_selected = col2.radio('Graph Statut', ('non', 'oui'))
    is_income_selected = col3.radio('Graph Revenu', ('non', 'oui'))

    return is_educ_selected, is_statut_selected, is_income_selected


def hist_graph():
    st.bar_chart(data['DAYS_BIRTH'])
    df = pd.DataFrame(data[:200], columns=['DAYS_BIRTH', 'AMT_CREDIT'])
    df.hist()
    st.pyplot()


def education_type():
    ed = data.groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    u_ed = data.NAME_EDUCATION_TYPE.unique()
    # fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
    # st.plotly_chart(fig)

    fig = go.Figure(data=[go.Bar(
        x=u_ed,
        y=ed
    )])
    fig.update_layout(title_text='Data education')

    st.plotly_chart(fig)

    ed_solvable = data[data['TARGET'] == 0].groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    ed_non_solvable = data[data['TARGET'] == 1].groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    u_ed = data.NAME_EDUCATION_TYPE.unique()
    # fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
    # st.plotly_chart(fig)

    fig = go.Figure(data=[
        go.Bar(name='Solvable', x=u_ed, y=ed_solvable),
        go.Bar(name='Non Solvable', x=u_ed, y=ed_non_solvable)
    ])
    fig.update_layout(title_text='Solvabilité Vs education')

    st.plotly_chart(fig)


def statut_plot():
    ed = data.groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
    u_ed = data.NAME_FAMILY_STATUS.unique()
    # fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
    # st.plotly_chart(fig)

    fig = go.Figure(data=[go.Bar(
        x=u_ed,
        y=ed
    )])
    fig.update_layout(title_text='Data situation familiale')

    st.plotly_chart(fig)

    ed_solvable = data[data['TARGET'] == 0].groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
    ed_non_solvable = data[data['TARGET'] == 1].groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
    u_ed = data.NAME_FAMILY_STATUS.unique()
    # fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
    # st.plotly_chart(fig)

    fig = go.Figure(data=[
        go.Bar(name='Solvable', x=u_ed, y=ed_solvable),
        go.Bar(name='Non Solvable', x=u_ed, y=ed_non_solvable)
    ])
    fig.update_layout(title_text='Solvabilité Vs situation familiale')

    st.plotly_chart(fig)


def income_type():
    ed = data.groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
    u_ed = data.NAME_INCOME_TYPE.unique()
    # fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
    # st.plotly_chart(fig)

    fig = go.Figure(data=[go.Bar(
        x=u_ed,
        y=ed
    )])
    fig.update_layout(title_text='Data Type de Revenu')

    st.plotly_chart(fig)

    ed_solvable = data[data['TARGET'] == 0].groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
    ed_non_solvable = data[data['TARGET'] == 1].groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
    u_ed = data.NAME_INCOME_TYPE.unique()
    # fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
    # st.plotly_chart(fig)

    fig = go.Figure(data=[
        go.Bar(name='Solvable', x=u_ed, y=ed_solvable),
        go.Bar(name='Non Solvable', x=u_ed, y=ed_non_solvable)
    ])
    fig.update_layout(title_text='Solvabilité Vs Type de Revenu')

    st.plotly_chart(fig)


###------------------------ Distribution ------------------------
def filter_distribution():
    st.subheader("Filtre des Distribution")
    col1, col2 = st.beta_columns(2)
    is_age_selected = col1.radio("Distribution Age ", ('non', 'oui'))
    is_incomdis_selected = col2.radio('Distribution Revenus ', ('non', 'oui'))

    return is_age_selected, is_incomdis_selected


def age_distribution():
    df = pd.DataFrame({'Age': data['DAYS_BIRTH'],
                       'Solvabilite': data['TARGET']})

    dic = {0: "solvable", 1: "non solvable"}
    df = df.replace({"Solvabilite": dic})

    # fig = ff.create_distplot([revenus_solvable],['solvable'] ,bin_size=.25)
    fig = px.histogram(df, x="Age", color="Solvabilite", nbins=40)
    st.subheader("Distribution des ages selon la sovabilité")
    st.plotly_chart(fig)


def revenu_distribution():
    df = pd.DataFrame({'Revenus': data['AMT_INCOME_TOTAL'],
                       'Solvabilite': data['TARGET']})

    dic = {0: "solvable", 1: "non solvable"}
    df = df.replace({"Solvabilite": dic})

    # fig = ff.create_distplot([revenus_solvable],['solvable'] ,bin_size=.25)
    fig = px.histogram(df, x="Revenus", color="Solvabilite", nbins=40)
    st.subheader("Distribution des revenus selon la sovabilité")
    st.plotly_chart(fig)


# --------------------------- Client Predection --------------------------

def show_client_predection():
    client_id = st.number_input("Donnez Id du Client", 100002)
    if st.button('Voir Client'):
        client = data[data['SK_ID_CURR'] == client_id]

        display_client_info(str(client['SK_ID_CURR'].values[0]), str(client['AMT_INCOME_TOTAL'].values[0]),
                            str(round(client['DAYS_BIRTH'].values[0])),
                            str(round(client['DAYS_EMPLOYED'] / -365).values[0]))

        # st.header('ID :'+str(client['SK_ID_CURR'][0]))
        # st.write(data['age_bins'].value_counts())
        y_pred, y_proba = predict_client_par_ID(client_id)
        st.info('Prediction du client : ' + str(int(100 * y_proba[0][0])) + ' %')
        client_prediction = st.progress(0)
        for percent_complete in range(int(100 * client['pred_prob'].values[0])):
            time.sleep(0.01)

        client_prediction.progress(percent_complete + 1)
        if (client['pred_prob'].values[0] < seuil_risque):
            st.success('Client solvable')
        if (client['pred_prob'].values[0] >= seuil_risque):
            st.error('Client non solvable')

        st.subheader("Tous les détails du client :")
        st.write(client)

        # Bar Chart
        age_bins = data['age_bins'].value_counts(sort=False)

        d = {'Ages par Decennie': age_bins.index, 'Nombre de clients par Decennie': age_bins.values}
        ages_decinnie = pd.DataFrame(data=d)

        ages_decinnie['Ages par Decennie'] = ages_decinnie['Ages par Decennie'].astype(str)
        idx_decinnie = ages_decinnie[ages_decinnie['Ages par Decennie'] == client['age_bins'].values[0]].index

        colors = ['lightslategray', ] * len(ages_decinnie['Nombre de clients par Decennie'])
        colors[idx_decinnie.values[0]] = 'crimson'

        fig = go.Figure(data=[go.Bar(
            x=ages_decinnie['Ages par Decennie'],
            y=ages_decinnie['Nombre de clients par Decennie'],
            marker_color=colors  # marker color can be a single color value or an iterable
        )])
        fig.update_layout(title_text='Nombre de Clients par Décinnie')

        st.plotly_chart(fig)

        # Line Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data['pred_prob'], x=data['AMT_INCOME_TOTAL'], mode='markers',
                                 name='Revenus des autres clients'))
        fig.add_trace(
            go.Scatter(y=client['pred_prob'], x=client['AMT_INCOME_TOTAL'], mode='markers', name='Revenu de ce Client',
                       marker=dict(size=[25])))
        st.plotly_chart(fig)


# --------------------------- model analysis -------------------------
### Confusion matrixe
def matrix_confusion(X, y):
    cm = confusion_matrix(X, y)
    print('\nTrue Positives(TP) = ', cm[0, 0])
    print('\nTrue Negatives(TN) = ', cm[1, 1])
    print('\nFalse Positives(FP) = ', cm[0, 1])
    print('\nFalse Negatives(FN) = ', cm[1, 0])
    return cm


### ----------------------- Prédiction d'un client ----------------

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Selectionner un fichier client', filenames)
    return os.path.join(folder_path, selected_filename)


def show_client_prediction():
    st.subheader("Selectionner source des données du client")
    selected_choice = st.radio("", ('Client existant dans le dataset', 'Nouveau client'))

    if selected_choice == 'Client existant dans le dataset':
        client_id = st.number_input("Donnez Id du Client", 100002)
        if st.button('Prédire Client'):
            y_pred, y_proba = predict_client_par_ID(client_id)
            st.info('Probabilité de solvabilité du client : ' + str(100 * y_proba[0][0]) + ' %')
            st.info("Notez que 100% => Client non slovable ")

            if (y_proba[0][0] < seuil_risque):
                st.success('Client prédit comme solvable')
            if (y_proba[0][0] >= seuil_risque):
                st.error('Client prédit comme non solvable !')

    if selected_choice == 'Nouveau client':
        filename = file_selector()
        st.write('Fichier du nouveau client selectionné `%s`' % filename)

        if st.button('Prédire Client'):
            nouveau_client = pd.read_csv(filename)
            y_pred, y_proba = predict_client(nouveau_client)
            st.info('Probabilité de solvabilité du client : ' + str(100 * y_proba[0][0]) + ' %')
            st.info("Notez que 100% => Client non slovable ")

            if (y_proba[0][0] < seuil_risque):
                st.success('Client prédit comme solvable')
            if (y_proba[0][0] >= seuil_risque):
                st.error('Client prédit comme non solvable !')


### Title
st.title('Home Credit Default Risk')

### Sidebar
st.sidebar.title("Menus")
sidebar_selection = st.sidebar.radio(
    'Select Menu:',
    ['Overview', 'Data Analysis', 'Model & Prediction', 'Prédire solvabilité client'],
)

if sidebar_selection == 'Overview':
    selected_item = ""
    with st.spinner('Data load in progress...'):
        time.sleep(2)
    st.success('Data loaded')
    show_data()
    show_overview()

if sidebar_selection == 'Data Analysis':
    selected_item = st.sidebar.selectbox('Select Menu:',
                                         ('Graphs', 'Distributions'))

if sidebar_selection == 'Model & Prediction':
    selected_item = st.sidebar.selectbox('Select Menu:',
                                         ('Prediction', 'Model'))

if sidebar_selection == 'Prédire solvabilité client':
    selected_item = "predire_client"

seuil_risque = st.sidebar.slider("Seuil de Solvabilité", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if selected_item == 'Data':
    show_data()

if selected_item == 'Solvency':
    show_overview()

if selected_item == 'Graphs':
    # hist_graph()
    is_educ_selected, is_statut_selected, is_income_selected = filter_graphs()
    if (is_educ_selected == "oui"):
        education_type()
    if (is_statut_selected == "oui"):
        statut_plot()
    if (is_income_selected == "oui"):
        income_type()

if selected_item == 'Distributions':
    is_age_selected, is_incomdis_selected = filter_distribution()
    if (is_age_selected == "oui"):
        age_distribution()
    if (is_incomdis_selected == "oui"):
        revenu_distribution()

if selected_item == 'Prediction':
    show_client_predection()

if selected_item == 'predire_client':
    show_client_prediction()
