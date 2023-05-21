# TO RUN : $streamlit run dashboard/dashboard.py
# Local URL: http://localhost:8501
# Network URL: http://192.168.0.50:8501
# Online URL : http://15.188.179.79

#

import streamlit as st

# Utilisation de SK_IDS dans st.sidebar.selectbox
import seaborn as sns
import os
import plotly.express as px
import streamlit as st
from PIL import Image
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import time
import pickle
import plotly.graph_objects as go
import shap
from streamlit_shap import st_shap
import numpy as np
###---------- load data -------- 

#------------- Affichage des infos client en HTML------------------------------------------
def display_client_info(id,revenu,age,nb_ann_travail):
   
    components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div class="card" style="width: 500px; margin:10px;padding:0">
        <div class="card-body">
            <h5 class="card-title">Info Client</h5>
            
            <ul class="list-group list-group-flush">
                <li class="list-group-item"> <b>ID                           : </b>"""+id+"""</li>
                <li class="list-group-item"> <b>Revenu                       : </b>"""+revenu+"""</li>
                <li class="list-group-item"> <b>Age                          : </b>"""+age+"""</li>
                <li class="list-group-item"> <b>Nombre d'années travaillées  : </b>"""+nb_ann_travail+"""</li>
            </ul>
        </div>
    </div>
    """,
    height=300
    
    )
#==============================================================================================
def predict():
    lgbm = pickle.load(open("lgbm.pkl", 'rb'))
    if lgbm:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)           
            #X_transformed = preprocessing(query)
            y_pred = randomForest.predict(X_train)
            y_proba = randomForest.predict_proba(X_train)
            
            return jsonify({'prediction': y_pred,'prediction_proba':y_proba[0][0]})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')

def predictByClientId():
    lgbm = pickle.load(open("lgbm.pkl", 'rb'))
    if lgbm:
        try:
            json_ = request.json
            print(json_)
            sample_size = 10000
            
            print(json_)  

            sample_size= 20000
            #data_set = data = pd.read_csv("df_final.csv",nrows=sample_size)
            client=data_set[X_train['SK_ID_CURR']==json_['SK_ID_CURR']].drop(['SK_ID_CURR','TARGET'],axis=1)
            print(client)
 
            
            preproc = pickle.load(open("preprocessor.sav", 'rb'))
            #X_transformed =preproc.transform(client)
            y_pred = randomForest.predict(client)
            y_proba = randomForest.predict_proba(client)
            
            return jsonify({'prediction': str(y_pred[0]),'prediction_proba':str(y_proba[0][0])})


        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')


def get_sk_id_list():
        # API_URL = "http://127.0.0.1:5000/api/"
        API_URL = https://app-p7-4.herokuapp.com/api/
        
        # URL of the sk_id API
        SK_IDS_API_URL = API_URL + "sk_ids/"

        # Requesting the API and saving the response
        response = requests.get(SK_IDS_API_URL)

        # Convert from JSON format to Python dict
        content = json.loads(response.content)

        # Getting the values of SK_IDS from the content
        SK_IDS = content['data']
        print ('MESSAGE 1')
        return SK_IDS
    
        

    ##################################################
    ##################################################
    ##################################################

### Data
def show_data(data):
    st.write(data.head(10))

    print("je suis dans la fonction")
### Solvency
def pie_chart(thres,data):
    #st.write(100* (data['TARGET']>thres).sum()/data.shape[0])
    print("je suis dans la fonction")
    percent_sup_seuil =100* (data['TARGET']>thres).sum()/data.shape[0]
    percent_inf_seuil = 100-percent_sup_seuil
    d = {'col1': [percent_sup_seuil,percent_inf_seuil], 'col2': ['% Non Solvable','% Solvable',]}
    df = pd.DataFrame(data=d)
    #fig = plt.pie(df,values='col1', names='col2', title=' Pourcentage de solvabilité des clients di dataset')
    fig = plt.pie(df)
    #plt.pie(y, labels = mylabels, colors=colors,explode = explodevalues, autopct='%1.1f%%', shadow = True)

    st.plotly_chart(fig)

def show_overview(data):
    st.title("Risque")
    risque_threshold = st.slider(label = 'Seuil de risque', min_value = 0.0,
                    max_value = 1.0 ,
                     value = 0.5,
                     step = 0.1)
    #st.write(risque_threshold)
    pie_chart(risque_threshold,data) 

 
### Graphs
def filter_graphs():
    st.subheader("Filtre des Graphes")
    col1, col2,col3 = st.columns(3)
    is_educ_selected = col1.radio("Graph Education",('non','oui'))
    is_statut_selected = col2.radio('Graph Statut',('non','oui'))
    is_income_selected = col3.radio('Graph Revenu',('non','oui'))

    return is_educ_selected,is_statut_selected,is_income_selected

def hist_graph ():
    st.bar_chart(data['DAYS_BIRTH'])
    df = pd.DataFrame(data[:200],columns = ['DAYS_BIRTH','AMT_CREDIT'])
    df.hist()
    st.pyplot()

def education_type(train_set):
    ed = train_set.groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    u_ed = train_set.NAME_EDUCATION_TYPE.unique() 
    fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
    st.plotly_chart(fig)


def main():

    SK_IDS = get_sk_id_list()

    # Logo "Prêt à dépenser"
    image = Image.open('logo.png')
    st.sidebar.image(image, width=280)
    st.title('Tableau de bord - "Prêt à dépenser"')

    ### Title
    st.title('Home Credit Default Risk')

    ##################################################
    # Selecting applicant ID
    select_sk_id = st.sidebar.selectbox('Select SK_ID from list:', SK_IDS, key=1)
    st.write('You selected: ', select_sk_id)


#################################################################


if __name__ == "__main__":

    lecture_X_test_original()
    lecture_X_test_clean()
    lecture_description_variables()

    # Titre 1
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                1. Quel est le score de votre client ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")




def test_load():
    assert load_dataset(1).size == 500  
