import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class ModelMonitor:
    def __init__(self):
        self.metrics_history = []
        
    def log_prediction(self, true_label: str, predicted_label: str, confidence: float):
        """Log each prediction for monitoring"""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': confidence
        })
        
    def generate_report(self, save_path: str = "reports"):
        """Generate performance report"""
        df = pd.DataFrame(self.metrics_history)
        
        # Calculate accuracy over time
        df['correct'] = df['true_label'] == df['predicted_label']
        accuracy_trend = df.rolling(window=100)['correct'].mean()
        
        # Plot metrics
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=accuracy_trend)
        plt.title('Model Accuracy Trend')
        plt.savefig(f"{save_path}/accuracy_trend.png")
        
        # Generate classification report
        report = classification_report(
            df['true_label'], 
            df['predicted_label'],
            output_dict=True
        )
        
        with open(f"{save_path}/performance_report.json", 'w') as f:
            json.dump(report, f, indent=2)
