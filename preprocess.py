import pandas as pd
import os
import re

file_path = './data/Tweets.csv' 
output_path='./data/preprocessed_data.csv'

def preprocess_data(file_path, output_path):
    # Checks if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return None

    # Load dataset
    df = pd.read_csv(file_path)

    # Selects relevant columns and copy
    df = df[['airline_sentiment', 'text']].copy()

    # Map target sentiment values
    target_map = {'positive': 1, 'negative': 0, 'neutral': 2}
    df['target'] = df['airline_sentiment'].map(target_map)
    
    def clean_text(text):
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Removes emojis
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)  # Keep important punctuation
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text.strip()

    # Apply the clean_text function to the 'text' column
    df['text'] = df['text'].apply(clean_text)

    # Filter and rename columns
    df2 = df[['text', 'target']]
    df2.columns = ['sentence', 'label']

    # Save the preprocessed data to the 'data' folder
    df2.to_csv(output_path, index=False)
    print(f"Preprocessing complete. File saved to '{output_path}'")

    return df2

# Run preprocessing function
preprocessed_data = preprocess_data(file_path, output_path)
if preprocessed_data is not None:
    print("Data preprocessed successfully.")