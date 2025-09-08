import streamlit as st
import pickle
import re
import numpy as np

# --- Load Model and Tools ---
model = pickle.load(open(r"C:\Users\Akhila\PROJECTS\sentment_analysis\sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open(r"C:\Users\Akhila\PROJECTS\sentment_analysis\tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open(r"C:\Users\Akhila\PROJECTS\sentment_analysis\label_encoder.pkl", "rb"))

# --- Text Cleaning Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# --- Prediction Function ---
def predict_sentiment(text):
    cleaned = clean_text(text)
    vect_text = vectorizer.transform([cleaned])
    pred_proba = model.predict_proba(vect_text)[0]
    pred_class = np.argmax(pred_proba)
    sentiment = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)
    return sentiment, confidence

# --- UI Setup ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💬 Smart Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a sentence below and find out if it's Positive, Negative, or Neutral.</p>", unsafe_allow_html=True)
st.divider()

# --- Input Area ---
user_input = st.text_input("💡 Your Sentence:", placeholder="Type something like 'I love this product!'")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔍 Predict"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a valid sentence.")
        else:
            sentiment, confidence = predict_sentiment(user_input)

            color_map = {
                "positive": "#2ecc71",
                "neutral": "#f1c40f",
                "negative": "#e74c3c"
            }

            emoji_map = {
                "positive": "😊",
                "neutral": "😐",
                "negative": "😠"
            }

            st.markdown(f"""
                <div style="background-color: {color_map[sentiment]}; padding: 15px; border-radius: 10px;">
                    <h3 style="color: white; text-align: center;">{emoji_map[sentiment]} Sentiment: {sentiment.capitalize()}</h3>
                    <p style="color: white; text-align: center;">Confidence: {confidence}%</p>
                </div>
            """, unsafe_allow_html=True)

