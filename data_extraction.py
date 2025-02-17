import os
from pymongo import MongoClient
from config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_processed_ids():
    """Retrieve processed ticket IDs from a file."""
    if not os.path.exists("processed_ids.txt"):
        return []
    with open("processed_ids.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

def fetch_tickets(batch_size=10000):
    """Fetch unprocessed tickets from MongoDB."""
    processed = get_processed_ids()
    client = MongoClient(MONGODB_URI)
    collection = client[DATABASE_NAME][COLLECTION_NAME]
    
    # Only fetch documents with required fields
    query = {
        "_id": {"$nin": list(processed)},
        "fields": {"$exists": True, "$type": "object"}
    }
    
    logging.info(f"Fetching tickets with batch size: {batch_size}")
    cursor = collection.find(query).batch_size(batch_size)
    tickets = list(cursor)
    logging.info(f"Fetched {len(tickets)} tickets")
    
    return tickets

def update_processed_ids(processed_ids):
    """Update the list of processed ticket IDs in a file."""
    with open("processed_ids.txt", "a") as f:
        for ticket_id in processed_ids:
            f.write(f"{ticket_id}\n")
    logger.info(f"Updated processed IDs: {len(processed_ids)} new IDs added.")