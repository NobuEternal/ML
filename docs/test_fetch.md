# Test Fetch Script

## Purpose
The test fetch script is used to verify the connection to the MongoDB database and retrieve a sample document along with the total count of documents in the specified collection.

## Functionality
The script connects to the MongoDB database using the provided URI, database name, and collection name. It then fetches a sample document and prints it along with the total count of documents in the collection.

## Running the Test Fetch Script
To run the test fetch script, execute the following command:
```bash
python test_fetch.py
```

## Example Output
The script will output a sample document and the total count of documents in the collection. For example:
```
Sample Document: { ... }
Total Documents: 275000
```

## Code
```python
# test_fetch.py
from pymongo import MongoClient
from config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME

client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Print one document and total count
doc = collection.find_one()
count = collection.count_documents({})
print("Sample Document:", doc)
print("Total Documents:", count)
```
