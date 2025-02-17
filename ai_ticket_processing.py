from transformers import pipeline
import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class IncidentAnalyzer:
    """GPU-accelerated ticket analysis"""
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.models = self._initialize_models()
        
    def _initialize_models(self) -> Dict[str, Any]:
        try:
            return {
                "severity": pipeline(
                    "text-classification",
                    model="j-hartmann/security-severity-bert",
                    device=self.device
                ),
                "incident_type": pipeline(
                    "zero-shot-classification",
                    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                    device=self.device,
                    framework="pt"
                )
            }
        except Exception as e:
            logger.critical(f"Model initialization failed: {e}")
            raise

    def analyze_ticket(self, text: str) -> Dict[str, Any]:
        """Full ticket analysis pipeline"""
        if not text or len(text) < 50:
            return {"error": "Insufficient text"}
            
        try:
            # Severity analysis
            severity_result = self.models["severity"](text[:1024])
            
            # Incident type classification
            candidate_labels = [
                "Unauthorized Access", "Data Breach", 
                "Service Outage", "Configuration Drift",
                "Vulnerability Exploit", "Deployment Failure"
            ]
            type_result = self.models["incident_type"](
                text[:1024],
                candidate_labels,
                multi_label=True
            )
            
            return {
                "severity": severity_result[0]["score"],
                "severity_label": severity_result[0]["label"],
                "primary_type": type_result["labels"][0],
                "secondary_types": type_result["labels"][1:3]
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": str(e)}

def analyze_ticket_severity(text: str) -> float:
    analyzer = IncidentAnalyzer()
    result = analyzer.analyze_ticket(text)
    return result.get("severity", 0.0)

def extract_incident_patterns(text: str) -> Dict[str, Any]:
    analyzer = IncidentAnalyzer()
    result = analyzer.analyze_ticket(text)
    return {
        "incident_type": result.get("primary_type", "Unknown"),
        "related_types": result.get("secondary_types", [])
    }