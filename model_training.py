# model_training.py
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
import joblib
import torch
import logging
import os
import re  # Add this import
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from tqdm import tqdm
import bs4
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, config_path="field_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.label_encoders = {}
            
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """Enhanced feature engineering for IT tickets"""
        X = X.copy()
        logger.info(f"Input columns: {X.columns.tolist()}")
        
        # Handle boolean columns first
        bool_cols = ['sla_breached', 'high_urgency', 'has_attachments']
        for col in bool_cols:
            if col in X.columns:
                X[col] = X[col].map({
                    True: 1, 'True': 1, 'true': 1,
                    False: 0, 'False': 0, 'false': 0,
                    None: 0, 'None': 0, np.nan: 0
                }).fillna(0).astype('int32')
        
        # Technical features with proper type conversion
        tech_features = [
            'is_infrastructure', 'is_security', 'is_database',
            'is_application', 'is_deployment', 'is_monitoring'
        ]
        
        # Ensure all technical features exist and convert to proper format
        for col in tech_features:
            if col not in X.columns:
                X[col] = 0
            else:
                # Convert boolean strings and actual booleans to integers
                X[col] = X[col].map({
                    True: 1, 'True': 1, 'true': 1, '1': 1, 1: 1,
                    False: 0, 'False': 0, 'false': 0, '0': 0, 0: 0,
                    'none': 0, 'None': 0, None: 0, np.nan: 0
                }).fillna(0).astype('int32')
        
        # Convert comment_count to numeric
        if 'comment_count' in X.columns:
            X['comment_count'] = pd.to_numeric(X['comment_count'], errors='coerce').fillna(0).astype('int32')
            
        # Handle time-based features
        if "time_to_resolve" in X.columns:
            # Convert to numeric, handling any string or invalid values
            X["time_to_resolve"] = pd.to_numeric(
                X["time_to_resolve"].replace([np.inf, -np.inf, None, '', 'None'], np.nan),
                errors='coerce'
            ).fillna(0)
            
            X["response_speed"] = 1.0 / (1.0 + X["time_to_resolve"])
            X["is_quick_fix"] = (X["time_to_resolve"] < 3600).astype('int32')
        else:
            X["response_speed"] = 0.0
            X["is_quick_fix"] = 0
        
        # Handle technical priority conversion
        if 'technical_priority' in X.columns:
            X['technical_priority'] = pd.to_numeric(
                X['technical_priority'].replace(
                    {'True': '1', 'False': '0', True: '1', False: '0'}
                ),
                errors='coerce'
            ).fillna(0).astype('float32')
        
        # Calculate urgency score safely
        X["urgency_score"] = X.apply(
            lambda x: 4.0 if pd.to_numeric(x.get("sla_breached", 0), errors='coerce') > 0
            else 3.0 if pd.to_numeric(x.get("high_urgency", 0), errors='coerce') > 0
            else 2.0 if pd.to_numeric(x.get("technical_priority", 0), errors='coerce') > 3
            else 1.0,
            axis=1
        )
        
        # Ensure numeric columns are properly typed
        numeric_cols = {
            "tech_score": "float32",
            "impact_score": "float32",
            "technical_priority": "float32",
            "urgency_score": "float32",
            "response_speed": "float32"
        }
        
        for col, dtype in numeric_cols.items():
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(dtype)
        
        # Combine features
        feature_cols = [
            "urgency_score", "tech_score", "response_speed",
            "is_quick_fix", "comment_count"
        ] + tech_features
        
        # Ensure all required columns exist
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        return X[feature_cols]

class TechnicalTextEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel, BartTokenizer, BartForConditionalGeneration
        # Load BART for summarization
        self.summarizer_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.summarizer_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        self.summarizer_model.cpu()  # Force CPU mode initially
        
        # Load CodeBERT for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.model.cpu()  # Force CPU mode initially
        
    def fit(self, X, y=None):
        return self
        
    def summarize_text(self, text):
        """Summarize long text using BART"""
        try:
            if not text or len(text) <= 500:
                return text
                
            # Truncate to max input length for BART
            max_length = 1024
            if len(text) > max_length:
                text = text[:max_length]
                
            inputs = self.summarizer_tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            )
            
            with torch.no_grad():
                summary_ids = self.summarizer_model.generate(
                    inputs["input_ids"],
                    max_length=150,
                    min_length=40,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            summary = self.summarizer_tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )
            
            return summary
            
        except Exception as e:
            logger.warning(f"Summarization failed: {str(e)}")
            return text[:500]  # Fallback to truncation
        
    def preprocess_technical_text(self, text):
        """Enhanced preprocessing for technical text"""
        # Extract code blocks and error messages
        code_pattern = r'```.*?```|`.*?`|\b[A-Za-z]+Exception\b|\b[A-Za-z]+Error\b'
        technical_content = ' '.join(re.findall(code_pattern, text))
        
        # Combine with regular text
        return f"{technical_content} {text}"
        
    def transform(self, X):
        """Transform with technical focus"""
        logger.info("Starting technical text embedding...")
        
        # Combine summary and description
        texts = X.apply(
            lambda x: self.preprocess_technical_text(
                f"{x['summary']} {x['description']}"
            ),
            axis=1
        ).tolist()
        
        logger.info("Starting text embedding process...")
        descriptions = X["description"].fillna("").astype(str).tolist()
        total_samples = len(descriptions)
        logger.info(f"Processing {total_samples} descriptions")
        
        # Process descriptions in batches
        batch_size = 32
        all_embeddings = []
        
        # Create progress bar
        progress = tqdm(
            range(0, total_samples, batch_size),
            desc="Processing Text",
            unit="batch"
        )
        
        for i in progress:
            batch = descriptions[i:min(i + batch_size, total_samples)]
            current_batch_size = len(batch)
            
            # Update progress description
            progress.set_description(
                f"Processing Batch {i//batch_size + 1}/{(total_samples-1)//batch_size + 1}"
            )
            
            # Summarize long texts
            logger.info(f"Summarizing batch {i//batch_size + 1}")
            summaries = []
            for text in batch:
                if len(text) > 500:
                    summary = self.summarize_text(text)
                    summaries.append(summary)
                else:
                    summaries.append(text)
            
            # Get embeddings for batch
            logger.info(f"Generating embeddings for batch {i//batch_size + 1}")
            inputs = self.tokenizer(
                summaries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                all_embeddings.append(embeddings)
                
            # Update progress
            progress.update(1)
            
        progress.close()
        logger.info("Text embedding completed")
                
        # Concatenate all batches
        final_embeddings = np.vstack(all_embeddings)
        logger.info(f"Final embeddings shape: {final_embeddings.shape}")
        return final_embeddings

def validate_training_data(df: pd.DataFrame) -> bool:
    """Validate data before training"""
    required_columns = [
        'summary', 'description', 'priority',
        'tech_score', 'impact_score', 'is_security',
        'is_infrastructure'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if len(df) < 100:
        raise ValueError(f"Insufficient training data: {len(df)} samples")
    
    return True

def train_and_save_model(df):
    try:
        logging.info("Starting model training pipeline")
        
        # Validate input data
        validate_training_data(df)
        
        # Prepare feature engineering pipeline
        feature_pipe = Pipeline([
            ('features', FeatureEngineer()),
        ])
        
        text_pipe = Pipeline([
            ('text', TechnicalTextEmbedder()),
        ])
        
        # Process features with progress tracking
        logging.info("Transforming features...")
        X_features = feature_pipe.fit_transform(df)
        logging.info(f"Feature matrix shape: {X_features.shape}")
        
        # Process text with progress tracking
        logging.info("Processing text data...")
        X_text = text_pipe.fit_transform(df)
        logging.info(f"Text embedding shape: {X_text.shape}")
        
        # Combine features
        X = np.hstack([X_features, X_text])
        
        # Prepare target variables with validation
        le = LabelEncoder()
        y = pd.DataFrame({
            'priority': le.fit_transform(df['priority'].fillna('Medium')),
            'category': le.fit_transform(df['ticket_category'].fillna('Other')),
            'urgency': df['calculated_priority'].fillna(2).astype(int)
        })
        
        # Train model
        logging.info("Training XGBoost model...")
        model = XGBClassifier(
            objective='multi:softprob',
            tree_method='gpu_hist' if torch.cuda.is_available() else 'auto',
            n_estimators=200,
            max_depth=7,
            eval_metric=['mlogloss', 'auc'],
            early_stopping_rounds=10
        )
        
        model.fit(X, y)
        
        # Save pipeline
        os.makedirs("models", exist_ok=True)
        pipeline_data = {
            'feature_pipe': feature_pipe,
            'text_pipe': text_pipe,
            'model': model,
            'label_encoder': le,
            'feature_names': feature_pipe.get_feature_names_out(),
            'technical_categories': list(le.classes_)
        }
        
        joblib.dump(pipeline_data, "models/full_pipeline.joblib")
        logging.info(f"✅ Model trained successfully on {len(df)} samples")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Training failed: {str(e)}")
        raise