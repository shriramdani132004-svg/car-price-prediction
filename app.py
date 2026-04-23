# ==========================================
# STREAMLIT APP (PIPELINE VERSION)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("model/car_model.pkl", "rb"))

st.title("🚗 Car Price Prediction App")

st.write("Enter car details below:")

# ==============================
# USER INPUTS
# ==============================

registration_year = st.number_input("Registration Year", 2000, 2025)
seats = st.number_input("Seats", 2, 10)
kms_driven = st.number_input("Kms Driven", 0, 200000)

manufacturing_year = st.number_input("Manufacturing Year", 2000, 2025)
mileage = st.number_input("Mileage (kmpl)", 0.0, 50.0)
engine = st.number_input("Engine (cc)", 500.0, 5000.0)
power = st.number_input("Max Power (bhp)", 50.0, 5000.0)
torque = st.number_input("Torque (Nm)", 50.0, 1000.0)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Ownership", ["First Owner", "Second Owner", "Third Owner"])
insurance = st.selectbox("Insurance", ["Comprehensive", "Third Party", "Zero Dep"])

# ==============================
# PREDICTION
# ==============================

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "registration_year": [registration_year],
        "seats": [seats],
        "kms_driven": [kms_driven],
        "manufacturing_year": [manufacturing_year],
        "mileage(kmpl)": [mileage],
        "engine(cc)": [engine],
        "max_power(bhp)": [power],
        "torque(Nm)": [torque],
        "fuel_type": [fuel],
        "transmission": [transmission],
        "ownsership": [owner],
        "insurance_validity": [insurance]
    })

    # Feature engineering SAME as training
    input_data["car_age"] = 2026 - input_data["manufacturing_year"]
    input_data["kms_per_year"] = input_data["kms_driven"] / (input_data["car_age"] + 1)

    input_data = input_data.drop(["manufacturing_year"], axis=1)

    # Prediction (log → inverse)
    pred_log = model.predict(input_data)
    prediction = np.expm1(pred_log)

    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f} Lakhs")