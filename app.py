from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

# Initializing FastAPI app
app = FastAPI(title="Sentiment Analysis API", description="Predict customer sentiment", version="1.0")

# Define request body schema
class TextRequest(BaseModel):
    text: str

# Path to the model directory
MODEL_PATH = "./model"  
BEST_CHECKPOINT = "checkpoint-3205"  # Best-performing checkpoint

# Helper function to load the model dynamically
def load_model():
    checkpoint_path = os.path.join(MODEL_PATH, BEST_CHECKPOINT)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        model.eval()  # Set model to evaluation mode
        return tokenizer, model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model from checkpoint {BEST_CHECKPOINT}: {str(e)}")

# Load the best model at startup
tokenizer, model = load_model()

@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "Sentiment Analysis API is running"}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    """
    Predict the sentiment of a given text.
    
    Request Body:
    {
        "text": "I love this airline!"
    }
    
    Response:
    {
        "text": "I love this airline!",
        "sentiment": "positive"
    }
    """
    try:
        # Tokenizes input text
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
        
        # Makes prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits.numpy()
        predicted_class = np.argmax(logits, axis=-1)[0]

        LABEL_MAP = {0: "negative", 1: "positive", 2: "neutral"}

        return {
            "text": request.text,
            "sentiment": LABEL_MAP[predicted_class]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
