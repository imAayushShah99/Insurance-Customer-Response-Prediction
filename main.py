import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("random_forest_model.joblib")

# Define expected feature names
expected_columns = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
                    'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

st.title("Insurance Purchase Prediction")

# User input fields
Gender = st.selectbox("Gender - (0:Female , 1:Male)", [0, 1])  # Assuming 0: Female, 1: Male
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Driving_License = st.selectbox("Driving License - (0:No , 1:Yes)", [0, 1])  # 0: No, 1: Yes
Region_Code = st.number_input("Region Code", min_value=0.0, max_value=50.0, value=10.0)
Previously_Insured = st.selectbox("Previously Insured - (0:No , 1:Yes)", [0, 1])
Vehicle_Age = st.selectbox("Vehicle Age", ['< 1 Year', '1-2 Year', '> 2 Years'])
Vehicle_Damage = st.selectbox("Vehicle Damage", [0, 1])  # 0: No, 1: Yes
Annual_Premium = st.number_input("Annual Premium", min_value=1000.0, max_value=100000.0, value=30000.0)
Policy_Sales_Channel = st.number_input("Policy Sales Channel", min_value=0.0, max_value=150.0, value=26.0)
Vintage = st.number_input("Vintage (Days)", min_value=0, max_value=300, value=100)

# Convert categorical data
vehicle_age_mapping = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
Vehicle_Age = vehicle_age_mapping[Vehicle_Age]

# Create input array
input_data = pd.DataFrame([[Gender, Age, Driving_License, Region_Code, Previously_Insured,
                            Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, Vintage]],
                          columns=expected_columns)

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.write("Prediction:", "You seem to be interested" if prediction == 1 else "You seem to be not interested")
