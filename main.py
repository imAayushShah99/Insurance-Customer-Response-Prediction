import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")

# Define input fields
st.title("Insurance Policy Response Prediction")
st.write("Enter the details below to predict if a customer will respond to the insurance policy.")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
region_code = st.number_input("Region Code", min_value=1, max_value=50, value=10)
previously_insured = st.selectbox("Previously Insured", ["No", "Yes"])
vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
vehicle_damage = st.selectbox("Vehicle Damage", ["No", "Yes"])
annual_premium = st.number_input("Annual Premium", min_value=1000, max_value=100000, value=30000)
policy_sales_channel = st.number_input("Policy Sales Channel", min_value=1, max_value=200, value=26)

# Encode categorical variables
gender_encoded = 1 if gender == "Male" else 0
previously_insured_encoded = 1 if previously_insured == "Yes" else 0
vehicle_age_encoded = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}[vehicle_age]
vehicle_damage_encoded = 1 if vehicle_damage == "Yes" else 0

# Create input array
input_data = np.array([[age, gender_encoded, region_code, previously_insured_encoded, vehicle_age_encoded, vehicle_damage_encoded, annual_premium, policy_sales_channel]])

# Scale numerical features
input_data[:, [0, 6]] = scaler.transform(input_data[:, [0, 6]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.success(f"The customer is likely to respond. Probability: {probability:.2f}")
    else:
        st.error(f"The customer is unlikely to respond. Probability: {probability:.2f}")
