import os
import logging
import pandas as pd
from tqdm import tqdm
from data_extraction import fetch_tickets, update_processed_ids, get_total_tickets
from data_preprocessing import clean_data
from model_training import train_and_save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

# Add logger definition
logger = logging.getLogger(__name__)

CLEANED_DATA_FILE = "cleaned_data.csv"
BATCH_SIZE = 1000  # Reduced batch size

def clean_and_validate_csv(file_path: str) -> pd.DataFrame:
    """Clean and validate CSV data with robust error handling"""
    try:
        # First try with c engine
        try:
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                on_bad_lines='skip',
                engine='c',
                low_memory=False,
                quoting=3  # QUOTE_NONE
            )
        except Exception as e:
            logger.warning(f"C engine failed, falling back to python engine: {e}")
            # Fallback to python engine without low_memory option
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                on_bad_lines='skip',
                engine='python',
                quoting=3  # QUOTE_NONE
            )
        
        logging.info(f"Initial data shape: {df.shape}")
        
        # Clean date columns
        date_cols = ['created', 'updated']
        for col in date_cols:
            if (col in df.columns):
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean numeric columns with proper error handling
        numeric_cols = {
            'votes': 'int32',
            'watches': 'int32',
            'is_devops': 'int32',
            'is_security': 'int32',
            'priority_score': 'float32',
            'severity': 'float32',
            'impact_score': 'float32',
            'tech_score': 'float32'
        }
        
        for col, dtype in numeric_cols.items():
            if col in df.columns:
                # Clean and convert numeric values
                df[col] = (
                    df[col]
                    .astype(str)
                    .replace(r'[^0-9.-]', '', regex=True)
                    .replace(r'^\s*$|^nan$|^null$|^None$', '0', regex=True)
                )
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
        
        # Remove rows with critical missing data
        required_cols = ['summary', 'description', 'priority']
        df = df.dropna(subset=required_cols, how='all')
        
        # Ensure required derived columns exist
        if 'tech_score' not in df.columns:
            logging.info("Calculating technical scores...")
            df['tech_score'] = df.apply(
                lambda x: calculate_tech_score(x['description'], x['components']), 
                axis=1
            )
            
        if 'impact_score' not in df.columns:
            logging.info("Calculating impact scores...")
            df['impact_score'] = df.apply(
                lambda x: calculate_impact_score({
                    'technical_priority': x.get('technical_priority', 0),
                    'is_security': x.get('is_security', 0),
                    'comment_count': x.get('comment_count', 0)
                }), axis=1
            )
            
        if 'is_infrastructure' not in df.columns:
            logging.info("Determining infrastructure tickets...")
            df['is_infrastructure'] = df['components'].str.contains(
                r'server|cloud|kubernetes|docker|aws|azure|infra',
                case=False, na=False
            ).astype(int)
            
        # Convert comment_count to numeric if it's not already
        if df['comment_count'].dtype == 'object':
            df['comment_count'] = pd.to_numeric(
                df['comment_count'].str.extract(r'(\d+)', expand=False),
                errors='coerce'
            ).fillna(0).astype(int)
        
        # Log final data quality metrics
        logging.info(f"Final data shape: {df.shape}")
        logging.info(f"Columns found: {df.columns.tolist()}")
        logging.info("Column types:")
        for col in df.columns:
            logging.info(f"{col}: {df[col].dtype}")
        
        return df
        
    except Exception as e:
        logging.error(f"CSV cleaning failed: {str(e)}")
        raise

def calculate_tech_score(description: str, components: str) -> float:
    """Calculate technical complexity score"""
    score = 0.0
    
    # Add points for technical keywords in description
    tech_keywords = [
        'error', 'exception', 'crash', 'failed',
        'server', 'database', 'config', 'network',
        'api', 'authentication', 'performance'
    ]
    
    if isinstance(description, str):
        score += sum(2.0 for keyword in tech_keywords if keyword in description.lower())
    
    # Add points for technical components
    if isinstance(components, str):
        score += sum(2.0 for keyword in tech_keywords if keyword in components.lower())
    
    # Normalize to 0-10 scale
    return min(10.0, score)

def calculate_impact_score(record: dict) -> float:
    """Calculate ticket impact score"""
    score = 0.0
    
    # Technical priority impact
    score += record.get('technical_priority', 0) * 2.0
    
    # Security issues have higher impact
    if record.get('is_security', 0):
        score += 3.0
    
    # Activity level indicates importance
    comment_count = int(record.get('comment_count', 0))
    score += min(comment_count * 0.5, 2.0)
    
    # Normalize to 0-10 scale
    return min(10.0, score)

def main():
    logging.info("🚀 Starting AI Ticketing System")
    
    try:
        if not os.path.exists(CLEANED_DATA_FILE):
            logging.info("No cleaned data found. Starting data processing...")
            process_raw_data()
            
        if os.path.exists(CLEANED_DATA_FILE):
            file_size = os.path.getsize(CLEANED_DATA_FILE)
            logging.info(f"Found cleaned data file ({file_size/1024/1024:.2f} MB)")
            
            logging.info("🧠 Loading training data...")
            df = clean_and_validate_csv(CLEANED_DATA_FILE)
            
            if len(df) > 0:
                logging.info("Starting model training...")
                train_and_save_model(df)
            else:
                logging.error("No valid records found in cleaned data file")
            
        logging.info("✨ Pipeline completed successfully")
        
    except Exception as e:
        logging.critical(f"💀 Critical pipeline failure: {e}")
        raise

def process_raw_data():
    logging.info("📢 Processing raw data...")
    
    total_docs = get_total_tickets()
    logging.info(f"Total documents to process: {total_docs}")
    progress = tqdm(total=total_docs, desc="Processing Tickets")
    
    skip = 0
    while skip < total_docs:
        batch = fetch_tickets(batch_size=BATCH_SIZE, skip=skip)
        if not batch:
            logging.info("No more tickets to fetch.")
            break
            
        logging.info(f"Processing batch of size: {len(batch)}")
        try:
            cleaned_batch = clean_data(batch)
            logging.info(f"Cleaned batch size: {len(cleaned_batch)}")
            if not cleaned_batch.empty:
                logging.info(f"Saving cleaned batch of size: {len(cleaned_batch)}")
                # Save data
                cleaned_batch.to_csv(
                    CLEANED_DATA_FILE,
                    mode="a",
                    index=False,
                    header=not os.path.exists(CLEANED_DATA_FILE)
                )
                # Track processed IDs
                update_processed_ids(cleaned_batch["_id"].tolist())
                
            progress.update(len(batch))
            logging.info(f"Processed {progress.n} / {total_docs} documents")
            
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            continue
            
        skip += BATCH_SIZE
            
    progress.close()
    logging.info("Finished processing raw data")

if __name__ == "__main__":
    main()