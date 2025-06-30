from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn


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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Spam Detection API"}

@app.post("/predict", response_model=SpamPrediction)
def predict_spam(data: ReviewInput):
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

