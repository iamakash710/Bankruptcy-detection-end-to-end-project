import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
import pickle

st.title('Bankruptcy Detection')

st.sidebar.header('Input parameters')

def input_features():
    competitiveness = st.sidebar.selectbox('Management risk', ('1.0', '0.5', '0.0'))

    credibility = st.sidebar.selectbox('Credibility', ('1.0', '0.5', '0.0'))

    financial_flexibility = st.sidebar.selectbox('Financial flexibility', ('1.0', '0.5', '0.0'))

    data = {'competitiveness': competitiveness,
             'credibility': credibility,
             'financial_flexibility': financial_flexibility}

    features = pd.DataFrame(data=data, index=[0])
    return features


df = input_features()
st.subheader('Input parameters')
st.write(df)

model = pickle.load(open('logistic_model_bankruptcy.pkl', 'rb'))

pred = model.predict(df)

st.subheader('-----------------------------------------------------------------------------')
st.write('The given company is predicted as :')

st.subheader('Bankrupt' if pred ==1.0 else 'Non-Bankrupt')