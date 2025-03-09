import streamlit as st
import requests

# API URL
API_URL = "http://127.0.0.1:8000/predict"  

# Streamlit UI
st.title("SENTIMENT ANALYSIS")

# Text input
user_input = st.text_area("Enter text:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        try:
            # Send request to FastAPI backend
            response = requests.post(API_URL, json={"text": user_input})
            if response.status_code == 200:
                result = response.json()
                sentiment = result.get("sentiment", "Unknown")
                
                # Display result
                st.success(f"Predicted Sentiment: **{sentiment.capitalize()}**")
            else:
                st.error("Error: Could not get response from API.")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
    else:
        st.warning("Please enter some text.")


