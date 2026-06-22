import os
import streamlit as st
import numpy as np
import joblib

# BASE PATH (safe for all systems)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models correctly
model = joblib.load(os.path.join(BASE_DIR, "models", "random_forest.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "models", "columns.pkl"))

st.title("💼 Salary Prediction App")

# Inputs
age = st.number_input("Age", 18, 65)
experience = st.number_input("Years of Experience", 0, 40)

gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])

# Encoding
gender_val = 1 if gender == "Male" else 0

edu_bachelor = 1 if education == "Bachelor" else 0
edu_master = 1 if education == "Master" else 0
edu_phd = 1 if education == "PhD" else 0

# Create empty feature vector
input_data = np.zeros(len(columns))
input_dict = dict(zip(columns, input_data))

# Fill values safely (must match training columns)
if "Age" in input_dict:
    input_dict["Age"] = age

if "Years of Experience" in input_dict:
    input_dict["Years of Experience"] = experience

if "Gender" in input_dict:
    input_dict["Gender"] = gender_val

# Convert to model input
final_input = np.array(list(input_dict.values())).reshape(1, -1)

# Scale input
final_input_scaled = scaler.transform(final_input)

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(final_input_scaled)
    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")