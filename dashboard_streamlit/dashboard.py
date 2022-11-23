# Librairies import
import streamlit as st
import pandas as pd
import requests
import pickle
import plotly.graph_objects as go
import plotly.express as px
import shap
import streamlit.components.v1 as components
from PIL import Image
import matplotlib
# matplotlib.use('TkAgg')

st.set_page_config(page_title="Application d'aide à la décision d'octroi de prêt",
                   page_icon="logo_small.png",
                   layout="wide")
# initial_sidebar_state="expanded")

data = pd.read_csv("small_dataset_for_api.csv", nrows=200)
list_id_client = data["SK_ID_CURR"]
st.title("Application d'aide à la décision d'octroi de prêt")

# Initialise sections of dashboard
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')

with col2:
    image = Image.open('logo_small.png')
    st.image(image, caption="Prêt à dépenser", use_column_width='auto')

with col3:
    st.write(' ')

with st.sidebar:
    hello = st.container()
    with hello:
        st.header("Bienvenue !")
        st.write("Cette application a été créée pour vous aider à analyser les dossiers de demandes de prêt.")
        st.write("Attention, elle n'est pas infaillible ! ")
        st.write("Étudiez les données en détails pour confirmer votre décision.")

    # Selection of client
    id_client = st.selectbox(label="Quel client souhaitez-vous sélectionner ?", options=list_id_client)
    st.write('Vous avez sélectionné :', id_client)

# Display client information
    client_info = st.container()
    with client_info:
        st.header("Informations")
        # Features importantes : "CNT_CHILDREN", "FLAG_OWN_CAR", "DAYS_EMPLOYED", "AMT_GOOD_PRICE",
        # "PREV_APP_CREDIT_PERC_MIN","REG_REGION_NOT_WORK_REGION", "AMT_TOTAL_INCOME", "REGION_POPULATION_RELATIVE",
        # "FLAG_DOCUMENT_3", "DAYS_BIRTH","CODE_GENDER", "PAYMENT_RATE", "FLAG_OWN_REALTY", "CNT_CHILDREN",
        # "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "DAYS_BIRTH",
        # "DAYS_EMPLOYED", "OCCUPATION_TYPE"

        client = data[data["SK_ID_CURR"] == id_client]
        app_info = pd.read_csv("client_info.csv", nrows=400)
        app_info['AGE'] = app_info['DAYS_BIRTH'] / -365

        client_info = app_info[["SK_ID_CURR", "AGE", "CODE_GENDER", "DAYS_EMPLOYED", "OCCUPATION_TYPE",
                                "CNT_CHILDREN", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "AMT_INCOME_TOTAL",
                                "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]]
        client_info = client_info[client_info["SK_ID_CURR"] == id_client]
        client_info['DAYS_EMPLOYED'] = client_info['DAYS_EMPLOYED'] / -1
        client_info = client_info.set_index("SK_ID_CURR")
        client_info = client_info.T
        st.table(data=client_info)

credit_decision = st.container()
with credit_decision:
    st.header("Recommandation sur l'accord du prêt")

    # Load the model
    model = pickle.load(open("pipeline_sgd_model.pkl", 'rb'))
    # model = pickle.load(open("lgbm_model.pkl", 'rb'))
    # Predict for one client
    url = "https://bank-credit-score.herokuapp.com/predict/"
    # url de l'api avec la prediction pour le client
    response = requests.get((url + str(id_client)), params={"id_client": id_client})
    # retourne la réponse de l'api sous format json
    response.raise_for_status()  # raises exception when not a 2xx response
    if response.status_code != 204:
        proba = float(response.json())

    # decision du modèle selon la proba et le seuil
    threshold = 0.090056
    if proba <= threshold:
        decision = "Le prêt peut être accordé au client"
    else:
        decision = "Le prêt est refusé"
    st.write(decision)

# jauge de visualisation
    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=(1-proba),  # proba is referring to default class
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

shap_visualisation = st.container()
with shap_visualisation:
    st.header("Interprétation du modèle pour le client sélectionné")
    # Feature importance globale et locale

    # @st.cache(suppress_st_warning=True)
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)


    # explain the model's predictions using SHAP
    # shap_model = SGDClassifier(
    #   alpha=0.009500000000000001,
    #   epsilon=0.1,
    #   l1_ratio=0.15,
    #   learning_rate='optimal',
    #   loss='log_loss',
    #   penalty='l2',
    #   random_state=77,
    #   validation_fraction=0.2
    # )
    explainer = pickle.load(open("shap_explainer.pkl", 'rb'))
    shap_data = data.drop(['Unnamed: 0', 'SK_ID_CURR', 'TARGET'], axis=1)
    shap_values = explainer.shap_values(shap_data)

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    shap.initjs()
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0, :], shap_data.iloc[0, :]))
    # SHAP global feature importance
    st.header("Variables les plus impactantes pour la modélisation")
    st.image("sgd_feature_importance.png")

    # plt.title('Feature importance based on SHAP values')
    # fig, ax = plt.subplots()
    # shap.summary_plot(shap_values, shap_data)
    # plt.savefig("global_feature_importance.png")
    # st.image("global_feature_importance.png")

    # plt.title('Feature importance based on SHAP values (Bar)')
    # fig, ax_1 = plt.subplots()
    # shap.summary_plot(shap_values, shap_data, plot_type="bar")
    # st.pyplot(fig, bbox_inches='tight')

    # st_shap(shap.waterfall_plot(shap_values[0], max_display=20, show=True))
    # shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0])
    # visualize the training set predictions
    # st_shap(shap.summary_plot(shap_values, shap_data, feature_names=shap_data.columns, plot_type="bar"))
    # is working but opens in a new window

graph_visualisation = st.container()
with graph_visualisation:
    st.header("Analyse des variables du client")
    # Analyses univariées avec menu déroulant pour sélectionner la variable
    univar_feature = st.selectbox(label="Quelle variable souhaitez-vous analyser ?",
                                  options=['AMT_CREDIT', 'AMT_INCOME_TOTAL', "EXT_SOURCE_2", "EXT_SOURCE_3",
                                           "AMT_GOODS_PRICE", "CODE_GENDER", "PAYMENT_RATE", "FLAG_DOCUMENT_3",
                                           "FLAG_OWN_CAR", "APPROVED_CNT_PAYMENT_SUM", "AMT_ANNUITY",
                                           "BURO_MONTHS_BALANCE_SIZE_SUM"])

    st.write("NB: la variable TARGET est encodée comme ceci : 0 = Solvable, 1 = Risque de non solvabilité")
    fig = px.histogram(data, x=univar_feature, color='TARGET', width=1000, height=800)
    fig.add_vline(x=client[univar_feature].iloc[0], line_width=3, line_dash="dash")
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig)

    # Analyse bivariées avec menu déroulant idem
    bivar_features = st.multiselect(label="Quelles variables souhaitez-vous analyser ?",
                                    options=['AMT_CREDIT', 'AMT_INCOME_TOTAL', "EXT_SOURCE_1", "EXT_SOURCE_2",
                                             "EXT_SOURCE_3", "AMT_GOODS_PRICE", "CODE_GENDER", "PAYMENT_RATE",
                                             "FLAG_DOCUMENT_3", "FLAG_OWN_CAR", "APPROVED_CNT_PAYMENT_SUM",
                                             "AMT_ANNUITY", "CNT_CHILDREN", "DAYS_EMPLOYED", "PREV_APP_CREDIT_PERC_MIN",
                                             "PAYMENT_RATE", "FLAG_OWN_REALTY", "BURO_MONTHS_BALANCE_SIZE_SUM",
                                             "ACTIVE_DAYS_CREDIT_MAX"], max_selections=3)
    x = bivar_features[0]
    y = bivar_features[1]
    st.write("NB: la variable CODE_GENDER est encodée comme ceci : 0 = F, 1 = M")
    st.write("NB: la variable TARGET est encodée comme ceci : 0 = Solvable, 1 = Risque de non solvabilité")
    fig_2 = px.scatter(data, x=x, y=y, color='TARGET', width=1000, height=800)
    fig_2.add_trace((go.Scatter(x=client[x], y=client[y], mode='markers', name='Client')))
    fig_2.update_traces(marker_size=14)
    fig_2.update_yaxes(automargin=True)
    st.plotly_chart(fig_2)
