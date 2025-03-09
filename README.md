# Sentiment Analysis Project

## Project Overview
This project is part of the **Machine Learning Development Assessment** and aims to predict customer sentiment using NLP techniques. The sentiment analysis model is trained on **Tweets.csv**, which contains social media mentions related to airline sentiment.

## Dataset
- **Source:** Tweets.csv (Social Media Mentions)(https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Columns Used:**
  - `airline_sentiment`: Target labels (positive, negative, neutral)
  - `text`: Customer reviews

## Project Structure
```
sentiment_analysis/
│── data/
│   └── Tweets.csv  # Raw dataset
│   └── preprocessed_data.csv  # Processed dataset
│── model/
│── preprocess.py  # Data preprocessing
│── model.py  # Model training
│── app.py  # FastAPI backend for inference
│── streamlitUI.py  # Streamlit UI for sentiment prediction
│── requirements.txt  # Dependencies
```

## Installation & Setup
### 1. Create and Activate Virtual Environment
```sh
python -m venv sentiment
source sentiment/bin/activate  # On Mac/Linux
sentiment\Scripts\activate  # On Windows
```
### 2. Install Dependencies
```sh
pip install -r requirements.txt
```
### 3. Run Preprocessing
```sh
python preprocess.py
```
### 4. Train the Model
```sh
python model.py
```
### 5. Run the FastAPI Backend
```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```
### 6. Run the Streamlit UI
```sh
streamlit run streamlitUI.py
```

## Model Details
- **Pretrained Model:** `distilbert-base-cased`
- **Tokenization:** `AutoTokenizer`
- **Training Data Split:** 70% train, 30% test
- **Evaluation Metrics:** Accuracy, F1 Score

## API Endpoints
- **POST `/predict`** → Sentiment prediction

**Example Request:**
```json
{
  "text": "I love this airline!"
}
```
**Response:**
```json
{
  "text": "I love this airline!",
  "sentiment": "positive"
}
```

## Streamlit UI
The frontend allows users to input text and receive sentiment predictions via the FastAPI backend.

## Documenttion
https://docs.google.com/document/d/136LaySxiC-yt8h0XXo4b0jmS9_htvZ44/edit?usp=drive_link&ouid=111244720786006995641&rtpof=true&sd=true

## Author
Puspa Kamal Oli



