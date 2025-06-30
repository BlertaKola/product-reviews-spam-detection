from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
from typing import Optional
from dotenv import load_dotenv
import os
load_dotenv()

API_KEY = os.getenv("API_KEY")

MODEL_NAME = "IreNkweke/HamOrSpamModel"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

class ReviewInput(BaseModel):
    text: str

class SpamPrediction(BaseModel):
    is_spam: bool
    spam_probability: float
    non_spam_probability: float

app = FastAPI(title="Spam Detection API", version="1.0")

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Invalid or missing API key. Include 'X-API-Key' header."
        )
    return x_api_key

@app.post("/predict", response_model=SpamPrediction)
def predict_spam(data: ReviewInput, api_key: str = Depends(verify_api_key)):
    try:
        inputs = tokenizer(data.text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        spam_prob = probs[1].item()
        non_spam_prob = probs[0].item()
        return SpamPrediction(
            is_spam=spam_prob > 0.5,
            spam_probability=round(spam_prob, 4),
            non_spam_probability=round(non_spam_prob, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

