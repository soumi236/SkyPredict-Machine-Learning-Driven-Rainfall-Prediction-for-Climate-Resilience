import streamlit as st
import pickle
import pandas as pd

# Load the trained model and feature names
with open("Rainfall_Prediction_model.pkl", "rb") as file:
    model_data = pickle.load(file)
model = model_data["model"]
feature_names = model_data["feature_names"]

# Streamlit UI
st.title("Rainfall Prediction App")
st.write(
    """
    Enter the weather parameters below and click **Predict** to know if rainfall is expected.
    """
)

# Create input fields for each feature
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature.capitalize()}", value=0.0, format="%.2f")

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("Rainfall")
    else:
        st.info("No Rainfall")

st.markdown("---")
st.markdown("**How to run this app:**")
st.code("streamlit run app.py", language="bash")