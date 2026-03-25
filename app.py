import streamlit as st
import pickle
import numpy as np
import os


try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("💳 Credit Risk Prediction App")


limit_bal = st.number_input("Credit Limit (LIMIT_BAL)")
utilization = st.number_input("Utilization")
avg_pay_delay = st.number_input("Average Payment Delay")

if st.button("Predict Risk"):
    try:
        input_data = np.array([[limit_bal, utilization, avg_pay_delay]])
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]

        if prob > 0.4:
            st.error(f"⚠️ High Risk (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low Risk (Probability: {prob:.2f})")

    except Exception as e:
        st.error(f"Prediction error: {e}")