import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('models/trained_model.pkl','rb'))
label_encoder = pickle.load(open('models/label_encoder.pkl','rb'))
columns = pickle.load(open('models/columns.pkl','rb'))


locations = list(label_encoder.classes_)

st.set_page_config(page_title="Property Price Predictor", page_icon="🏠", layout="centered")
st.title("Property Price Predictor")
st.write("Enter property details below to get the predicted price in lakh")

st.divider()

col1,col2= st.columns(2)

with col1:
    location = st.selectbox("Location", sorted(locations))
    total_sqft = st.number_input("Total Sqft", min_value=300, max_value=10000, value=1000, step=50)

with col2:
    bhk = st.slider("BHK (Bedrooms)", min_value=1, max_value=10, value=2)
    bath = st.slider("Bathrooms", min_value=1, max_value=10, value=2)

st.divider()

if st.button("Predict Price", use_container_width=True):
    location_encoded = label_encoder.transform([location])[0]

    features = np.array([[location_encoded, total_sqft, bath, bhk]])
    prediction = model.predict(features)[0]

    st.success(f"### Predicted Price: ₹ {prediction:.2f} Lakhs")
    st.info(f"**Price Range:** ₹ {prediction * 0.9:.2f}L – ₹ {prediction * 1.1:.2f}L")
    st.balloons()

