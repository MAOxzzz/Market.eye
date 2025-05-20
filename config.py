"""
Configuration file for Market Eye AI.
Contains API keys, settings, and other configuration values.
"""

import os

# API Keys - Direct assignment for now
# In production, use environment variables for security
GEMINI_API_KEY = "AIzaSyCcB3tENdnFNKBIvciETMF196ldlUBmnyk"

# Attempt to load from environment if present
if os.getenv("GEMINI_API_KEY"):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
    if os.getenv("GEMINI_API_KEY"):
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
except Exception as e:
    print(f"Notice: .env loading skipped: {e}")

if GEMINI_API_KEY:
    print("Gemini API key found.")
else:
    print("WARNING: Gemini API key not found. AI recommendations will be disabled.")

# API Rate Limiting
GEMINI_REQUESTS_PER_MINUTE = 10  # Adjust based on your API tier
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 2  # seconds

# Models
GEMINI_MODEL = "gemini-1.5-flash"  # Default model

# Paths
DATA_DIR = "data"
RECOMMENDATIONS_DIR = os.path.join(DATA_DIR, "recommendations")
CREW_OUTPUTS_DIR = os.path.join(DATA_DIR, "crew_outputs")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

# Ensure directories exist
for directory in [DATA_DIR, RECOMMENDATIONS_DIR, CREW_OUTPUTS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True) 