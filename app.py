import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')
st.set_page_config(page_title="Spam Detector App", page_icon="🕵️")
st.title("👀 Spam Message Detector")
st.write("Type a message below to check if it's spam or not.")

# User input
message = st.text_area("Message:", height=150)

# Predict button
if st.button("Analyze"):
    if message:
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]
        label = "🚫 Spam" if prediction else "Not Spam"
        st.success(f"Prediction: {label}")
    else:
        st.warning("Please enter a message.")