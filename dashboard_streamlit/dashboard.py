# Librairies import
import streamlit as st
import pandas as pd
import requests
import pickle
import plotly.graph_objects as go

path = "C:/Users/I-NL/Documents/Projet7_OC/"

# Initialise sections of dashboard
st.title("Prêt à dépenser - Application d'aide à la décision pour les demandes de prêt")
data = pd.read_csv(r"C:\Users\I-NL\Documents\Projet7_OC\p7_clean_dataset_for_ml.csv", nrows=200)
list_id_client = data["SK_ID_CURR"]

header = st.container()
with header:
    st.header("Bienvenue !")
    st.write("Cette application a été créée pour vous aider à analyser les dossiers de demandes de prêt.")
    st.write("Attention, elle n'est pas infaillible, étudiez les données en détails pour confirmer votre décision.")

with st.sidebar:
    id_client = st.selectbox(label="Quel client souhaitez-vous sélectionner ?", options=list_id_client)
    st.write('Vous avez sélectionné :', id_client)

client_info = st.container()
with client_info:
    st.header("Informations du client")
    # st.write(data.head(10))
    # st.write(list_id_client)
    # Features importantes :
    main_features = [["CNT_CHILDREN",
                      "FLAG_OWN_CAR",
                      "DAYS_EMPLOYED",
                      "AMT_GOOD_PRICE",
                      "PREV_APP_CREDIT_PERC_MIN",
                      "REG_REGION_NOT_WORK_REGION",
                      "AMT_TOTAL_INCOME",
                      "REGION_POPULATION_RELATIVE",
                      "FLAG_DOCUMENT_3",
                      "DAYS_BIRTH",
                      "CODE_GENDER"
                      ]]
    client = data[data["SK_ID_CURR"] == id_client]  # Remplacer par data avant ml process, filtre features importantes

    st.table(data=client[["CNT_CHILDREN", "FLAG_OWN_CAR", "DAYS_EMPLOYED", "PREV_APP_CREDIT_PERC_MIN",
                          "REG_REGION_NOT_WORK_REGION", "REGION_POPULATION_RELATIVE",  # "AMT_TOTAL_INCOME",
                          "FLAG_DOCUMENT_3", "DAYS_BIRTH", "CODE_GENDER"]])

credit_decision = st.container()
with credit_decision:
    st.header("Recommendation sur l'accord du prêt")

    # Load the model
    model = pickle.load(open(path + "pipeline_sgd_model.pkl", 'rb'))
    # Predict for one client
    url = "http://127.0.0.1:5000/predict/"
    # url de l'api avec la prediction pour le client
    response = requests.get((url + str(id_client)), params={"id_client": id_client})
    # retourne la réponse de l'api sous format json
    response.raise_for_status()  # raises exception when not a 2xx response
    if response.status_code != 204:
        proba = float(response.json())

    # decision du modèle selon la proba et le seuil
    # proba = predict_client_with_id(id_client)
    threshold = 0.458607
    if proba <= threshold:
        decision = "Le prêt peut être accordé au client"
    else:
        decision = "Le prêt est refusé"
    st.write(decision)
    # jauge de visualisation

    fig = go.Figure(go.Indicator(
        mode="number+gauge", value=(1-proba),  # proba is refering to default class
        domain={'x': [0.1, 1], 'y': [0, 1]},
        title={'text': "<b> Score </b>"},
        gauge={
            'shape': "bullet",
            'axis': {'range': [None, 1]},
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.8,
                'value': 0.45},
            'steps': [
                {'range': [0, 0.4], 'color': "red"},
                {'range': [0.4, 0.5], 'color': "orange"},
                {'range': [0.5, 1], 'color': "lightgreen"}]}))
    fig.update_layout(height=250)
    st.plotly_chart(figure_or_data=fig, sharing="streamlit")

graph_visualisation = st.container()
with graph_visualisation:
    st.header("Visualisation des données")
    # Feature importance globale et locale
    sel_col, disp_col = st.columns(2)  # Creates 2 columns for the selection and display

    # Analyses univariées avec menu déroulant pour sélectionner la variable
    # univar_feature = sel_col.selectbox() # Variable à afficher
    # disp_col.barchart to display
    # Analyse bivariées avec menu déroulant idem
