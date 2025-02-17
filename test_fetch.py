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