# model_training.py
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
import joblib
import torch

class FeatureEngineer:
    def __init__(self, config_path="field_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
    def transform(self, df):
        # Add temporal features
        df["time_open"] = (df["updated"] - df["created"]).dt.total_seconds()
        df["security_priority"] = df.apply(
            lambda x: x["priority"] * 2 if x["is_security"] else x["priority"], 
            axis=1
        )
        return df

def train_and_save_model(df):
    try:
        # Feature Engineering
        feature_pipe = Pipeline([
            ('features', FeatureEngineer()),
            ('text', TextEmbedder())
        ])
        
        X = feature_pipe.fit_transform(df)
        y = pd.DataFrame({
            "priority": df["priority"],
            "severity": df["severity"],
            "incident_type": df["incident_type"]
        })

        # Train model
        model = XGBClassifier(
            objective='multi:softprob',
            tree_method='gpu_hist' if torch.cuda.is_available() else 'auto',
            n_estimators=200,
            max_depth=7
        )
        model.fit(X, y)

        # Save pipeline
        joblib.dump({
            'feature_pipe': feature_pipe,
            'model': model
        }, "models/full_pipeline.joblib")
        
        print(f"✅ Trained on {len(df)} samples")

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")

class TextEmbedder:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        
    def transform(self, texts):
        import torch
        inputs = self.tokenizer(
            texts.tolist(), 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()