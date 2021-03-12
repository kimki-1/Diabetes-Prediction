import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
import matplotlib
matplotlib.use('Agg')
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle

def run_ml_app() :
    st.subheader('Machine Learning')

    
    age = st.slider('나이', 1, 120)
    pregnancies = st.number_input('임신횟수', 0)
    glucose = st.number_input('공복 혈당', 0)
    bloodpressure = st.number_input('이완기 혈압(mm Hg)', 0)
    skinthickness = st.number_input('팔 삼두근 뒤쪽의 피하지방(mm)', 0)
    insulin = st.number_input('혈청 인슐린(mu U/ml)', 0)
    bmi = st.number_input('체질량 지수(wight in kg/(height in m)²', 0)
    diabetespedigreefunction = st.number_input('당뇨 내력 가중치(가족당뇨환자영향력)', 0)

    model = joblib.load('data/best_model.pkl')

    new_data = np.array( [ pregnancies, glucose, bloodpressure, skinthickness,
                    insulin, bmi, diabetespedigreefunction, age  ])
    new_data = new_data.reshape(1, -1)
    
    y_pred = model.predict(new_data)
    if st.button('예측') :
        if y_pred == 0 :
            st.markdown('### 당신은 당뇨가 아닙니다.')
        elif y_pred == 1 :
            st.markdown('### 당신은 당뇨입니다.')
