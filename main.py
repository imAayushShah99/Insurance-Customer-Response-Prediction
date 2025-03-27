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
Gender = st.selectbox("Gender - (0:Female , 1:Male)", [0, 1])  
Age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)  # Ensure integer input
Driving_License = st.selectbox("Driving License - (0:No , 1:Yes)", [0, 1])  
Region_Code = st.number_input("Region Code", min_value=0.0, max_value=50.0, value=10.0)  
Previously_Insured = st.selectbox("Previously Insured - (0:No , 1:Yes)", [0, 1])  
Vehicle_Age = st.selectbox("Vehicle Age", ['< 1 Year', '1-2 Year', '> 2 Years'])  
Vehicle_Damage = st.selectbox("Vehicle Damage", [0, 1])  
Annual_Premium = st.number_input("Annual Premium", min_value=1000.0, max_value=100000.0, value=30000.0)  
Policy_Sales_Channel = st.number_input("Policy Sales Channel", min_value=0.0, max_value=150.0, value=26.0)  
Vintage = st.number_input("Vintage (Days)", min_value=0, max_value=300, value=100, step=1)  # Ensure integer input  

# Convert categorical data
vehicle_age_mapping = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
Vehicle_Age = vehicle_age_mapping[Vehicle_Age]

# Ensure correct data types
input_data = pd.DataFrame([[Gender, int(Age), Driving_License, Region_Code, Previously_Insured,
                            Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, int(Vintage)]],
                          columns=expected_columns)

# Check column names to match model expectations
if list(input_data.columns) != expected_columns:
    st.write("Error: Feature names do not match model expectations.")

# Perform prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        st.write("Prediction:", "You seem to be interested" if prediction == 1 else "You seem to be not interested")
    except Exception as e:
        st.write("Error in prediction:", str(e))
