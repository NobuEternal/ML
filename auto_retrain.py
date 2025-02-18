from model_training import train_and_save_model
from monitoring import ModelMonitor
import schedule
import time
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class AutoRetrainer:
    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor
        self.min_accuracy_threshold = 0.85
        self.retrain_threshold = 1000  # Minimum new samples before retraining
        
    def check_performance(self):
        """Check if model needs retraining"""
        recent_accuracy = self.monitor.get_recent_accuracy()
        if recent_accuracy < self.min_accuracy_threshold:
            logger.warning(f"Model accuracy ({recent_accuracy:.2f}) below threshold")
            return True
        return False
        
    def retrain(self):
        """Retrain model if needed"""
        try:
            if self.check_performance():
                logger.info("Starting automated retraining...")
                
                # Backup current model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.rename(
                    "models/full_pipeline.joblib",
                    f"models/full_pipeline_{timestamp}.joblib"
                )
                
                # Retrain with updated data
                train_and_save_model(self.monitor.get_training_data())
                logger.info("Automated retraining completed successfully")
                
        except Exception as e:
            logger.error(f"Automated retraining failed: {e}")

def start_auto_retrain():
    monitor = ModelMonitor()
    retrainer = AutoRetrainer(monitor)
    
    # Schedule daily retraining check
    schedule.every().day.at("00:00").do(retrainer.retrain)
    
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour
