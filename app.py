
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Title and description
st.title("📰 Fake News Detection App")
st.write("Paste a news article below and I'll tell you if it's FAKE or REAL!")

# Text input
news_text = st.text_area("Paste the news article here:")

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Make prediction
if st.button("Predict"):
    if news_text:
        # Transform the input using the same TF-IDF vectorizer
        text_vector = vectorizer.transform([news_text])
        prediction = model.predict(text_vector)[0]

        # Display result
        if prediction == 1:
            st.error("🚫 Fake News Detected")
        else:
            st.success("✅ Real News")
    else:
        st.warning("Please enter some text to predict.")
