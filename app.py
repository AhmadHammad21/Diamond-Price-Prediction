import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# loading our saved model
model =  joblib.load("rf_pipeline.sav")

# for encoding the categorical variables
le =  LabelEncoder()


def transform(X):
    for col in X.columns:
        if col in ['clarity', 'color', 'cut']:
            X[col] = le.fit_transform(X[col])

    return X


st.title("How much does this diamond worth?")

carat = st.slider("Select Carat", 0.2, 5.02)

cut = st.select_slider("Select Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])

color = st.select_slider("Select your diamond's color", ['J', 'I', 'H', 'G', 'F', 'E', 'D'])

clarity = st.select_slider("Select which Clarity is your Diamond", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

x = st.slider("Select your X length in mm", 0.0, 10.74)

y = st.slider("Select your y width in mm", 0.0, 58.9)

z = st.slider("Select your z depth in mm", 0.0, 31.8)

table = st.slider("Select your table width in mm", 43, 95)


depth = z / ((x + y + 0.1) / 2)

volume = x * y * z

columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z' ,'volume']

def predict():
    row = np.array([carat, cut, color, clarity, depth, table, x, y, z, volume])
    
    X = pd.DataFrame([row], columns=columns)
    
    X = transform(X)
    prediction = model.predict(X)[0]


    st.success("Your diamond price prediction is: {}".format(prediction))

# run it streamlit run appname.py
st.button("Predict", on_click=predict)