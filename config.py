# config.py
import os

# Retrieve API keys from environment variables
# Ensure these environment variables are set before running the application

# OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

# VoyageAI API Key
VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')
if VOYAGE_API_KEY is None:
    raise EnvironmentError("VOYAGE_API_KEY environment variable is not set.")
