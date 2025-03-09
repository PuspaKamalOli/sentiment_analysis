import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Function to clean text data
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.strip()  # Remove leading and trailing whitespaces
    return text

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_text = " ".join([word for word in words if word not in stop_words])
    return filtered_text

# Function to preprocess text
def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return text

# Function to load dataset and preprocess
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(subset=['text'], inplace=True)  # Drop rows with missing text
    df['cleaned_text'] = df['text'].apply(preprocess_text)  # Apply preprocessing
    return df

if __name__ == "__main__":
    file_path = "data/Tweets.csv"  # Adjust the path accordingly
    processed_df = load_and_preprocess_data(file_path)
    print(processed_df.head())
