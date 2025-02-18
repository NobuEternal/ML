import joblib
import pandas as pd
import numpy as np
import logging
from data_preprocessing import clean_data
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

class TicketPredictor:
    def __init__(self, model_path="models/full_pipeline.joblib"):
        self.pipeline = joblib.load(model_path)
        self.feature_pipe = self.pipeline['feature_pipe']
        self.text_pipe = self.pipeline['text_pipe']
        self.model = self.pipeline['model']
        self.label_encoder = self.pipeline['label_encoder']
        self.categories = self.pipeline['technical_categories']
        
    def predict_ticket(self, ticket_data: Dict) -> Dict:
        """Predict priority and category for a single ticket"""
        # Convert single ticket to DataFrame
        df = pd.DataFrame([ticket_data])
        
        # Process features
        X_features = self.feature_pipe.transform(df)
        X_text = self.text_pipe.transform(df)
        X = np.hstack([X_features, X_text])
        
        # Get predictions
        predictions = self.model.predict_proba(X)
        
        # Map predictions to categories
        priority_probs = predictions[0][:len(self.categories)]
        category_idx = np.argmax(priority_probs)
        
        return {
            'predicted_priority': self.categories[category_idx],
            'priority_confidence': float(priority_probs[category_idx]),
            'predicted_category': self.categories[category_idx],
            'category_confidence': float(priority_probs[category_idx]),
            'urgency_score': float(predictions[0][-1]),
            'technical_indicators': {
                cat: float(prob) 
                for cat, prob in zip(self.categories, priority_probs)
                if prob > 0.1  # Show only relevant categories
            }
        }

    def batch_predict(self, tickets: List[Dict]) -> List[Dict]:
        """Process multiple tickets in batch"""
        if not tickets:
            return []
            
        # Clean and process tickets
        df = clean_data(tickets)
        
        # Get predictions for batch
        X_features = self.feature_pipe.transform(df)
        X_text = self.text_pipe.transform(df)
        X = np.hstack([X_features, X_text])
        
        predictions = self.model.predict_proba(X)
        
        results = []
        for idx, pred in enumerate(predictions):
            priority_probs = pred[:len(self.categories)]
            category_idx = np.argmax(priority_probs)
            
            results.append({
                'ticket_id': df.iloc[idx]['_id'],
                'predicted_priority': self.categories[category_idx],
                'priority_confidence': float(priority_probs[category_idx]),
                'predicted_category': self.categories[category_idx],
                'category_confidence': float(priority_probs[category_idx]),
                'urgency_score': float(pred[-1]),
                'technical_indicators': {
                    cat: float(prob)
                    for cat, prob in zip(self.categories, priority_probs)
                    if prob > 0.1
                }
            })
            
        return results

def predict_ticket_priority(ticket: Dict) -> Dict:
    """Convenience function for single ticket prediction"""
    predictor = TicketPredictor()
    return predictor.predict_ticket(ticket)

def batch_predict_tickets(tickets: List[Dict]) -> List[Dict]:
    """Convenience function for batch prediction"""
    predictor = TicketPredictor()
    return predictor.batch_predict(tickets)