import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from dateutil import parser
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
from eda_app import run_eda_app
from ml_app import run_ml_app



def main():
    st.title('당뇨병 예측')
    menu = [ 'Home', 'EDA', 'ML' ]
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home' :
        st.write('이 앱은 당뇨병을 예측하는 앱입니다.')
        st.write('왼쪽의 사이드바에서 선택하세요')
        st.write('컬럼정보')
        st.write(
            '''
            Pregnancies : Number of times pregnant 임신횟수

            Glucose : 공복혈당 Plasma glucose concentration a 2 hours in an oral glucose tolerance test

            BloodPressure : Diastolic blood pressure (mm Hg)

            SkinThickness : Triceps skin fold thickness (mm)

            Insulin : 2-Hour serum insulin (mu U/ml)

            BMI : Body mass index (weight in kg/(height in m)^2)

            Diabetes pedigree function

            Age (years)

            COutcome : class variable (0 or 1) 268 of 768 are 1, the others are 0
            '''
        )
    
    elif choice == 'EDA' :
        run_eda_app()

    elif choice == 'ML' :
        run_ml_app()


if __name__ == '__main__' :
    main()