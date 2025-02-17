import os
import logging
import pandas as pd
from tqdm import tqdm
from data_extraction import fetch_tickets, update_processed_ids
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

CLEANED_DATA_FILE = "cleaned_data.csv"
BATCH_SIZE = 10000

def main():
    logging.info("🚀 Starting AI Ticketing System")
    
    try:
        if not os.path.exists(CLEANED_DATA_FILE):
            process_raw_data()
            
        if os.path.exists(CLEANED_DATA_FILE):
            logging.info("\n🧠 Loading training data...")
            df = pd.read_csv(CLEANED_DATA_FILE, parse_dates=["created", "updated"])
            train_and_save_model(df)
            
        logging.info("✨ Pipeline completed successfully")
        
    except Exception as e:
        logging.critical(f"💀 Critical pipeline failure: {e}")
        raise

def process_raw_data():
    logging.info("📢 Processing raw data...")
    
    total_docs = sum(1 for _ in fetch_tickets(BATCH_SIZE))
    logging.info(f"Total documents to process: {total_docs}")
    progress = tqdm(total=total_docs, desc="Processing Tickets")
    
    for batch in fetch_tickets(BATCH_SIZE):
        logging.info(f"Processing batch of size: {len(batch)}")
        try:
            cleaned_batch = clean_data(batch)
            if not cleaned_batch.empty:
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
            
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            continue
            
    progress.close()
    logging.info("Finished processing raw data")

if __name__ == "__main__":
    main()