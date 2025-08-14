# D:\AI\Gits\hebrew-tutor-data-pipeline\test_blob.py
import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv()
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
print("Blob service client initialized successfully!")