from llama_index import Document, SimpleIndex

# Load sample data
import json

with open('sample_data.json') as f:
    data = json.load(f)

# Convert to Llama Index documents
documents = [Document(doc["content"], doc["id"]) for doc in data]

# Initialize and build the index
index = SimpleIndex()
for doc in documents:
    index.add_document(doc)

print("Index built successfully.")

