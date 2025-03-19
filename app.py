from fastapi import FastAPI
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the saved TinyBERT model and tokenizer
MODEL_PATH = "tinybert_sentiment_model_v2"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Define request body
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: TextInput):
    """Predict sentiment for a given text."""
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    
    # Ensure the model outputs four logits for four categories
    if logits.shape[1] != 4:
        return {"error": "Model does not support four-class classification. Retrain with four labels."}
    
    # Define new sentiment labels in correct order
    labels = ["Irrelevant", "Negative", "Neutral", "Positive"]
    
    # Get predicted label and score
    pred_index = np.argmax(logits[0])
    label = labels[pred_index]
    score = float(logits[0][pred_index])
    
    return {"label": label, "score": score}

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running!"}

# Run server if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

