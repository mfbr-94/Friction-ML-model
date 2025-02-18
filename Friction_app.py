#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pickle
import requests

# IMPORTANT: Include these if your model/pipeline references them
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------------
# 1. Define the custom LabelEncoderWrapper class used in the training script.
#    This must match the exact name and structure as in your training code.
class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        
    def fit(self, X, y=None):
        for i in range(X.shape[1]):
            le = LabelEncoder()
            le.fit(X[:, i])
            self.encoders[i] = le
        return self 

    def transform(self, X):
        X_transformed = X.copy()
        for i in range(X.shape[1]):
            X_transformed[:, i] = self.encoders[i].transform(X[:, i])
        return X_transformed

# --------------------------------------------------------------------------------
# 2. Streamlit App Title & Description
st.title("Friction Coefficient Predictor")

st.write("""
This app predicts the **Response friction Coefficient u (steady values)** based on the following inputs:
- MoDTC in model oil A (wt%) - (0 to 0.54)
- Maximum pressure B (Mpa) - (370 to 650)
- Temperature C (°C) - (30 to 90)
- Sliding speed D (m/s) - (0.55 to 1.1)
- Surface roughness (block) E (µm) - (0.02 to 0.12)
""")

# --------------------------------------------------------------------------------
# 3. Load the model from GitHub (or any raw URL)
#    Replace the URL below with the raw link to your pickle file on GitHub.
model_url = "https://github.com/mfbr-94/Friction-ML-model/raw/refs/heads/main/Friction_Coef_model.pkl"

@st.cache_resource  # Cache the model so it won't be reloaded each time
def load_model(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure we got a valid response (status 200)
        model = pickle.loads(response.content)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_url)

# --------------------------------------------------------------------------------
# 4. If the model loads, present input widgets and predict
if model is not None:
    st.success("Model loaded successfully from GitHub!")

    # Input widgets for the five parameters
    modtc = st.number_input('MoDTC in model oil A (wt%)', min_value=0.0, format="%.2f")
    max_pressure = st.number_input('Maximum pressure B (Mpa)', min_value=0.0, format="%.2f")
    temperature = st.number_input('Temperature C (°C)', format="%.2f", value=25.0)
    sliding_speed = st.number_input('Sliding speed D (m/s)', min_value=0.0, format="%.2f")
    surface_roughness = st.number_input('Surface roughness (block) E (µm)', min_value=0.0, format="%.2f")

    # When user clicks "Predict", make a prediction
    if st.button("Predict Friction Coefficient"):
        try:
            # Combine inputs into a NumPy array with shape (1, 5)
            input_features = np.array([[modtc, max_pressure, temperature, sliding_speed, surface_roughness]])
            
            # Make prediction using the loaded model
            prediction = model.predict(input_features)
            
            st.success(f"Predicted friction coefficient: {prediction[0]:.4f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.error("The model could not be loaded. Please check the GitHub URL or model file.")

