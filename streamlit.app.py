import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Real-World AI Model Deployment 🚀")

st.write("""
This app demonstrates deploying a **real AI model** to solve a practical problem — all running live on Streamlit Cloud!
""")

st.header("Predict Housing Prices")

bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of bathrooms", min_value=1, max_value=10, value=2)
sqft_living = st.number_input("Square footage of living space", min_value=300, max_value=10000, value=1500)
location_score = st.slider("Location desirability score (1-10)", 1, 10, 5)

if st.button("Predict Price"):
    features = np.array([[bedrooms, bathrooms, sqft_living, location_score]])
    prediction = model.predict(features)[0]
    st.success(f"🏠 Estimated House Price: ${prediction:,.2f}")

st.markdown("---")
st.caption("Model trained on real housing data. Deploying AI solutions made easy with Streamlit Cloud!")
