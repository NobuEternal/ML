from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from predict import TicketPredictor
import uvicorn
import logging

app = FastAPI(title="IT Ticket Processing API")
predictor = TicketPredictor()

class Ticket(BaseModel):
    summary: str
    description: str
    components: Optional[List[str]] = []
    priority: Optional[str] = None

class PredictionResponse(BaseModel):
    predicted_priority: str
    priority_confidence: float
    predicted_category: str
    technical_indicators: Dict[str, float]
    urgency_score: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_ticket(ticket: Ticket):
    try:
        result = predictor.predict_ticket(ticket.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
