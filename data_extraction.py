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

def get_total_tickets():
    """Get total number of unprocessed tickets in MongoDB."""
    processed = get_processed_ids()
    client = MongoClient(MONGODB_URI)
    collection = client[DATABASE_NAME][COLLECTION_NAME]
    
    query = {
        "_id": {"$nin": list(processed)},
        "fields": {"$exists": True, "$type": "object"}
    }
    
    total = collection.count_documents(query)
    logging.info(f"Found {total} unprocessed tickets")
    return total

def fetch_tickets(batch_size=1000, skip=0):
    """Fetch tickets using pagination."""
    processed = get_processed_ids()
    client = MongoClient(MONGODB_URI)
    collection = client[DATABASE_NAME][COLLECTION_NAME]
    
    query = {
        "_id": {"$nin": list(processed)},
        "fields": {"$exists": True, "$type": "object"}
    }
    
    logging.info(f"Fetching batch of {batch_size} tickets starting from {skip}")
    cursor = collection.find(query).skip(skip).limit(batch_size)
    
    tickets = list(cursor)
    logging.info(f"Fetched {len(tickets)} tickets in this batch")
    
    return tickets

def update_processed_ids(processed_ids):
    """Update the list of processed ticket IDs in a file."""
    with open("processed_ids.txt", "a") as f:
        for ticket_id in processed_ids:
            f.write(f"{ticket_id}\n")
    logger.info(f"Updated processed IDs: {len(processed_ids)} new IDs added.")