from transformers import pipeline
import torch
import logging
from typing import Dict, Any, List
from functools import lru_cache

logger = logging.getLogger(__name__)

class IncidentAnalyzer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IncidentAnalyzer, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
        
    def __init__(self):
        if not self.initialized:
            self.device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Initializing models on device: {self.device}")
            self.models = self._initialize_models()
            self.initialized = True
        
    def _initialize_models(self) -> Dict[str, Any]:
        try:
            return {
                "severity": pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=self.device,
                    batch_size=32  # Add batch processing
                ),
                "incident_type": pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=self.device,
                    batch_size=16  # Add batch processing
                )
            }
        except Exception as e:
            logger.critical(f"Model initialization failed: {e}")
            return None

    def analyze_tickets_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple tickets in batches"""
        if not texts:
            return []
            
        # Filter out None or empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text and isinstance(text, str) and len(text.strip()) > 10]
        
        if not valid_texts:
            return [{"severity": 0.5, "incident_type": "Unknown", "related_types": []} for _ in texts]
            
        indices, filtered_texts = zip(*valid_texts)
        results = [{"severity": 0.5, "incident_type": "Unknown", "related_types": []} for _ in texts]
        
        try:
            if self.models is None:
                return results
                
            # Process valid texts in batch
            severity_results = self.models["severity"](
                [text[:512] for text in filtered_texts]
            )
            
            type_results = self.models["incident_type"](
                [text[:512] for text in filtered_texts],
                candidate_labels=[
                    "Bug Report", "Feature Request",
                    "Security Issue", "Performance Problem",
                    "Documentation Issue"
                ],
                multi_label=False
            )
            
            # Update results only for valid texts
            for idx, (sev, typ) in enumerate(zip(severity_results, type_results)):
                original_idx = indices[idx]
                results[original_idx] = {
                    "severity": 0.8 if sev["label"] == "NEGATIVE" else 0.2,
                    "incident_type": typ["labels"][0],
                    "related_types": typ["labels"][1:2]
                }
                
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            
        return results

# Cache the analyzer instance
@lru_cache(maxsize=1)
def get_analyzer():
    return IncidentAnalyzer()

def analyze_ticket_severity(text: str) -> float:
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze_tickets_batch([text])[0]
        return result.get("severity", 0.5)
    except Exception as e:
        logger.error(f"Severity analysis failed: {e}")
        return 0.5

def extract_incident_patterns(text: str) -> Dict[str, Any]:
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze_tickets_batch([text])[0]
        return {
            "incident_type": result.get("incident_type", "Unknown"),
            "related_types": result.get("related_types", [])
        }
    except Exception as e:
        logger.error(f"Pattern extraction failed: {e}")
        return {
            "incident_type": "Unknown",
            "related_types": []
        }