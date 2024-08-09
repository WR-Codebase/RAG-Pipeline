import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Weaviate instance URL and API token
api_url = os.getenv('WEAVIATE_API_URL')
token = os.getenv('WEAVIATE_API_TOKEN')

# Headers for the request
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {token}'
}

# Define the schema for the Chunks class
chunks_class_schema = {
    "class": "Chunks",
    "description": "A class to store document chunks",
    "properties": [
        {
            "name": "name",
            "dataType": ["string"],
            "description": "The name of the document"
        },
        {
            "name": "chunk_number",
            "dataType": ["int"],
            "description": "The chunk number within the document"
        },
        {
            "name": "content",
            "dataType": ["text"],
            "description": "The content of the chunk"
        }
    ]
}

# Function to create the Chunks class
def create_chunks_class():
    response = requests.post(f"{api_url}/schema", headers=headers, data=json.dumps(chunks_class_schema))
    if response.status_code == 200:
        print("Chunks class created successfully.")
    else:
        print(f"Failed to create Chunks class. Status code: {response.status_code}")
        print("Response:", response.json())

# Function to get the Chunks class schema
def get_chunks_class_schema():
    response = requests.get(f"{api_url}/schema/Chunks", headers=headers)
    if response.status_code == 200:
        print("Chunks class schema retrieved successfully.")
        print(response.json())
    else:
        print(f"Failed to retrieve Chunks class schema. Status code: {response.status_code}")
        print("Response:", response.json())

# Create the Chunks class
create_chunks_class()

# Verify the creation of the Chunks class
get_chunks_class_schema()
