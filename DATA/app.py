import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Current folder of app.py
BASE_DIR = Path(__file__).parent

# Load Model
model = joblib.load(BASE_DIR / "house_price_model.pkl")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 AI House Price Prediction")
st.write("Enter House Details")

# Numerical Inputs
area = st.number_input("Area (sq ft)", min_value=1000, value=7420)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=4)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
stories = st.number_input("Stories", min_value=1, max_value=5, value=3)
parking = st.number_input("Parking", min_value=0, max_value=5, value=2)

# Categorical Inputs
mainroad = st.selectbox("Main Road", ["Yes", "No"])
guestroom = st.selectbox("Guest Room", ["Yes", "No"])
basement = st.selectbox("Basement", ["Yes", "No"])
hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
furnishing = st.selectbox(
    "Furnishing Status",
    ["Unfurnished", "Semi-Furnished", "Furnished"]
)

# Encoding
mainroad = 1 if mainroad == "Yes" else 0
guestroom = 1 if guestroom == "Yes" else 0
basement = 1 if basement == "Yes" else 0
hotwaterheating = 1 if hotwaterheating == "Yes" else 0
airconditioning = 1 if airconditioning == "Yes" else 0
prefarea = 1 if prefarea == "Yes" else 0

furnishing = {
    "Unfurnished": 0,
    "Semi-Furnished": 1,
    "Furnished": 2
}[furnishing]

# Prediction
if st.button("Predict Price"):

    data = pd.DataFrame([[

        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        parking,
        prefarea,
        furnishing

    ]], columns=[
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "parking",
        "prefarea",
        "furnishingstatus"
    ])

    prediction = model.predict(data)

st.success(f"🏡 Estimated House Price")

st.metric(
    label="Predicted Price",
    value=f"₹ {prediction[0]:,.0f}"
)

st.markdown("""
Predict house prices using a Machine Learning model built with **Linear Regression**.

### Features Used
- Area
- Bedrooms
- Bathrooms
- Stories
- Parking
- Main Road
- Guest Room
- Basement
- Air Conditioning
- Preferred Area
- Furnishing Status
""")

st.markdown("---")
st.write("Developed by Daivdeep Solge using Python, Scikit-learn and Streamlit")
    