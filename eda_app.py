import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
import matplotlib
matplotlib.use('Agg')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle


def run_eda_app():
    st.subheader('EDA 화면 입니다.')

    df = pd.read_csv('data/diabetes.csv')

    side_radio_menu = [
        '데이터프레임 and 통계치','컬럼별 데이터프레임','상관계수'
    ]
    selected_side_radio = st.sidebar.radio('선택하세요', side_radio_menu)


    if selected_side_radio == '데이터프레임 and 통계치' :
        radio_menu = ['데이터 프레임', '통계치']
        selected_radio = st.radio('선택하세요', radio_menu)

        if selected_radio == '데이터 프레임' :
            st.dataframe(df)

        elif selected_radio == '통계치' :
            st.dataframe(df.describe())



    elif selected_side_radio == '컬럼별 데이터프레임' :
        columns = df.columns
        columns = list(columns)
        
        selected_columns = st.multiselect('컬럼을 선택하시오', columns)
        
        if len(selected_columns) != 0 :
            st.dataframe(df[selected_columns])
        else :
            st.write('선택한 컬럼이 없습니다.')



    elif selected_side_radio == '상관계수' :
        corr_columns = df.columns[ df.dtypes != object ] 
        selected_corr = st.multiselect('상관계수 컬럼을 선택', corr_columns)


        if len(selected_corr) != 0 :
            
            fig = sns.pairplot(data = df[selected_corr], corner=True) 
            st.pyplot(fig)
            st.dataframe(df[selected_corr].corr())
        else :
            st.write('선택한 컬럼이 없습니다.')

    

    
