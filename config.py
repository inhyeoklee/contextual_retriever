import os

class Config:
    """Configuration class to manage API keys and other settings."""
    VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    @staticmethod
    def validate_keys():
        """Validate that all necessary API keys are set."""
        if not Config.VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY is not set. Please set it in your environment variables.")
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Please set it in your environment variables.")

# Validate API keys at import time
Config.validate_keys()
