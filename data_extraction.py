import os
from pymongo import MongoClient
from config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME

# Documentation of fetch_tickets
## Fetch Unprocessed Tickets from MongoDB
# The `fetch_tickets` function retrieves unprocessed ticket documents from a MongoDB collection, specifically from the "JiraRepos" database and "Jira" collection, while ensuring that only documents containing the required fields are fetched. It utilizes a batch size of 10,000 for efficient data handling and excludes any documents whose IDs are present in the set of processed IDs obtained from the "processed_ids.txt" file. The function constructs a query to filter documents based on the existence and type of the "fields" attribute, returning a cursor for further processing.

def fetch_tickets(batch_size=10000):
    processed = get_processed_ids()
    client = MongoClient(MONGODB_URI)
    collection = client[DATABASE_NAME][COLLECTION_NAME]
    
    # Only fetch documents with required fields
    query = {
        "_id": {"$nin": list(processed)},
        "fields": {"$exists": True, "$type": "object"}
    }
    
    # Convert the cursor to a list before returning
    return list(collection.find(query).batch_size(batch_size))