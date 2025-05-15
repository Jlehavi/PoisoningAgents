import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")

# Example usage
print(f"API key loaded: {'✅ Success' if api_key else '❌ Failed'}") 