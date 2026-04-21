import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="ML Predictor", page_icon="🤖", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("Model (8).pkl", "rb"))

model = load_model()

# Title
st.title("🤖 Smart Prediction App")
st.markdown("### Enter feature values to get predictions")

# Sidebar
st.sidebar.header("⚙️ Input Features")

# ⚠️ CHANGE THIS BASED ON YOUR MODEL
feature_names = ["Feature 1", "Feature 2", "Feature 3"]

inputs = []
for feature in feature_names:
    val = st.sidebar.number_input(f"{feature}", value=0.0)
    inputs.append(val)

input_array = np.array([inputs])

# Main button
if st.button("🚀 Predict"):
    try:
        prediction = model.predict(input_array)

        st.success("✅ Prediction Successful!")
        st.metric(label="Prediction Result", value=prediction[0])

        # If probability available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_array)
            st.subheader("📊 Prediction Probability")
            st.write(proba)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Ready for Deployment 🚀")
