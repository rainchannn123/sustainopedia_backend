import os
from dotenv import load_dotenv 
load_dotenv('./RAG-LLM-Backend-small-Ecoinvent/.env')

print('CONNECTION_STRING:', os.environ.get('AZURE_STORAGE_CONNECTION_STRING'))
print('CONTAINER_NAME:', os.environ.get('AZURE_CONTAINER_NAME'))
print('ZIP_BLOB:', os.environ.get('ECOINVENT_ZIP_BLOB'))
