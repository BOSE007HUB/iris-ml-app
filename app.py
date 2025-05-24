import streamlit as st
from sklearn.model_selection import train_test_split 
import pandas as pd 
import pickle 
# Load model
model = pickle.load(open("model.pkl", "rb")) 
st.title("Iris Flower Classifier") 
st.write("Enter the flower details to predict its species") 
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0) 
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5) 
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0) 
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5)

if st.button("Predict"):
 data = [[sepal_length, sepal_width, petal_length, petal_width]] 
 prediction = model.predict(data) 
 st.success(f"Predicted Species: {prediction[0]}")
