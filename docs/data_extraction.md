# Data Extraction Process

This document provides detailed information on the data extraction process used in the AI Ticket Processing System.

## Purpose and Functionality

The data extraction process is responsible for retrieving IT ticket data from a MongoDB database. The extracted data is then used for further processing and analysis in the system.

## Configuration

Before running the data extraction process, ensure that the `config.py` file is properly configured with the MongoDB URI, database name, and collection name.

Example configuration in `config.py`:
```python
# config.py
MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "JiraRepos"     # Database name
COLLECTION_NAME = "Jira"        # Collection with 275K documents (or "JiraEcosystem"/"JFrog")
```

## Running the Data Extraction Process

To run the data extraction process, execute the `data_extraction.py` script. This script will connect to the MongoDB database, retrieve the ticket data, and save the processed ticket IDs to a file.

Example command:
```bash
python data_extraction.py
```

## Functions

The `data_extraction.py` script includes the following functions:

### `get_processed_ids()`

Retrieves the list of processed ticket IDs from a file.

### `get_total_tickets()`

Gets the total number of unprocessed tickets in the MongoDB database.

### `fetch_tickets(batch_size=1000, skip=0)`

Fetches tickets from the MongoDB database using pagination.

### `update_processed_ids(processed_ids)`

Updates the list of processed ticket IDs in a file.

## Example Code

Here is an example of the `data_extraction.py` script:

```python
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
```

## Notes

- Ensure that the MongoDB server is running and accessible before executing the data extraction script.
- The `processed_ids.txt` file is used to keep track of the processed ticket IDs to avoid duplicate processing.
