# Librairies import
import streamlit as st
import pandas as pd
import requests
import pickle
import plotly.graph_objects as go
import plotly.express as px
import shap
import streamlit.components.v1 as components
from sklearn.linear_model import SGDClassifier
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt  # ['nbAgg', , 'QtCairo', , 'QtAgg', 'Qt5Cairo', 'TkAgg', 'TkCairo',
# 'WX', 'WXAgg', 'WXCairo', 'agg',  'pdf', 'pgf', 'ps', 'svg'
# plt.switch_backend('agg')
# import tkinter

path = "C:/Users/I-NL/Documents/Projet7_OC/"
data = pd.read_csv(r"C:\Users\I-NL\Documents\Projet7_OC\p7_clean_dataset_for_ml.csv", nrows=200)
list_id_client = data["SK_ID_CURR"]

# Initialise sections of dashboard
image = Image.open(path + 'logo_small.png')
st.image(image, caption="Prêt à dépenser", use_column_width='auto')
st.title("Application d'aide à la décision d'octroi de prêt")

with st.sidebar:
    hello = st.container()
    with hello:
        st.header("Bienvenue !")
        st.write("Cette application a été créée pour vous aider à analyser les dossiers de demandes de prêt.")
        st.write("Attention, elle n'est pas infaillible ! ")
        st.write("Etudiez les données en détails pour confirmer votre décision.")

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
        app_info = pd.read_csv(path+"client_info.csv", nrows=400)
        app_info['AGE'] = app_info['DAYS_BIRTH'] / -365

        client_info = app_info[["SK_ID_CURR", "AGE", "CODE_GENDER", "DAYS_EMPLOYED", "OCCUPATION_TYPE",
                                "CNT_CHILDREN", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "AMT_INCOME_TOTAL",
                                "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]]
        client_info = client_info[client_info["SK_ID_CURR"] == id_client]
        client_info['DAYS_EMPLOYED'] = client_info['DAYS_EMPLOYED'] / -1
        client_info = client_info.set_index("SK_ID_CURR")
        client_info = client_info.T
        st.table(data=client_info)
        # [["DAYS_BIRTH", "CODE_GENDER", "DAYS_EMPLOYED", "OCCUPATION_TYPE", "CNT_CHILDREN",
        #                          "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "AMT_INCOME_TOTAL",
        #                           "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]])

credit_decision = st.container()
with credit_decision:
    st.header("Recommandation sur l'accord du prêt")

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

shap_visualisation = st.container()
with shap_visualisation:
    st.header("Feature importance")
    # Feature importance globale et locale

    # @st.cache(suppress_st_warning=True)
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)


    # explain the model's predictions using SHAP
    shap_model = SGDClassifier(
        alpha=0.009500000000000001,
        epsilon=0.1,
        l1_ratio=0.15,
        learning_rate='optimal',
        loss='log_loss',
        penalty='l2',
        random_state=77,
        validation_fraction=0.2
    )
    explainer = pickle.load(open(path + "shap_explainer.pkl", 'rb'))
    shap_data = data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    shap_values = explainer.shap_values(shap_data)

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    shap.initjs()
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0, :], shap_data.iloc[0, :]))
    # SHAP global feature importance

    #plt.title('Feature importance based on SHAP values')
    #fig, ax = plt.subplots()
    #shap.summary_plot(shap_values, shap_data)
    #plt.savefig("global_feature_importance.png")
    #st.image("global_feature_importance.png")

    #plt.title('Feature importance based on SHAP values (Bar)')
    #fig, ax_1 = plt.subplots()
    #shap.summary_plot(shap_values, shap_data, plot_type="bar")
    #st.pyplot(fig, bbox_inches='tight')
    # st_shap(shap.waterfall_plot(shap_values[0], max_display=20, show=True))
    # shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0])
    # visualize the training set predictions
    # st_shap(shap.summary_plot(shap_values, shap_data, feature_names=shap_data.columns, plot_type="bar"))
    # is working but opens in a new window

    # Other try:
    # st_shap(shap.plots.waterfall(shap_values[0]), height=300)
    # st_shap(shap.plots.beeswarm(shap_values[0]), height=300)

    # st_shap(plt.plot(shap.summary_plot(shap_values, shap_data)))  # feature_names=shap_data.columns, plot_type='bar'))

    # st_shap(shap.force_plot(explainer.expected_value, shap_values, shap_data))

graph_visualisation = st.container()
with graph_visualisation:
    st.header("Analyse des variables du client")
    # Analyses univariées avec menu déroulant pour sélectionner la variable

    # sel_col, disp_col = st.columns(2)  # Creates 2 columns for the selection and display

    univar_feature = st.selectbox(label="Quelle variable souhaitez-vous analyser ?",
                                  options=['AMT_CREDIT', 'AMT_INCOME_TOTAL', "EXT_SOURCE_2", "EXT_SOURCE_3",
                                           "AMT_GOODS_PRICE", "CODE_GENDER", "PAYMENT_RATE", "FLAG_DOCUMENT_3",
                                           "FLAG_OWN_CAR", "APPROVED_CNT_PAYMENT_SUM", "AMT_ANNUITY",
                                           "BURO_MONTHS_BALANCE_SIZE_SUM"])
    # Variable à afficher : à compléter ?
    # PENSER à utiliser les données non encodées !!

    # st.bar_chart(data=data, x="SK_ID_CURR", y=univar_feature, use_container_width=True)
    fig = px.histogram(data, x=univar_feature, color='TARGET')
    fig.add_vline(x=client[univar_feature].iloc[0], line_width=3, line_dash="dash")
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
    fig_2 = px.scatter(data, x=x, y=y, color='TARGET')
    fig_2.add_trace((go.Scatter(x=client[x], y=client[y], mode='markers', name='Client')))
    fig_2.update_traces(marker_size=14)
    # fig_2.add_vline(x=client[x].iloc[0], line_width=3, line_dash="dash")
    # fig_2.add_hline(y=client[y].iloc[0], line_width=3, line_dash="dash")
    st.plotly_chart(fig_2)

    # Graphs
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
        u_ed = data['NAME_EDUCATION_TYPE'].unique()
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
        u_ed = data['NAME_FAMILY_STATUS'].unique()
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
        u_ed = data['NAME_FAMILY_STATUS'].unique()
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

#    ###------------------------ Distribution ------------------------

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
