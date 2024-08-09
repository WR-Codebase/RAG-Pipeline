import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API URL and token from environment variables
api_url = os.getenv('WEAVIATE_API_URL')
token = os.getenv('WEAVIATE_API_TOKEN')


# Function to delete all objects in the Harriette class
def delete_all_objects(api_url, token):
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Fetch all objects in the Harriette class
    query = """
    {
      Get {
        Chunks {
          _additional {
            id
          }
        }
      }
    }
    """
    response = requests.post(f"{api_url}/graphql", headers=headers, json={"query": query})
    try:
        response_json = response.json()
    except ValueError:
        print("Error parsing JSON response for object fetch:", response.text)
        return
    
    # Extract the IDs of all objects
    if 'data' in response_json and 'Get' in response_json['data'] and 'Chunks' in response_json['data']['Get']:
        objects = response_json['data']['Get']['Chunks']
        object_ids = [obj['_additional']['id'] for obj in objects]
        
        # Delete each object by ID
        for object_id in object_ids:
            delete_response = requests.delete(f"{api_url}/objects/{object_id}", headers=headers)
            if delete_response.status_code == 204:
                print(f"Successfully deleted object with ID: {object_id}")
            else:
                try:
                    print("Failed to delete object with ID:", object_id, delete_response.json())
                except ValueError:
                    print("Failed to delete object with ID:", object_id, "Response:", delete_response.text)
    else:
        print("No objects found or error in fetching objects")

# Execute the delete function
delete_all_objects(api_url, token)
