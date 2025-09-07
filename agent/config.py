# config.py
from dotenv import load_dotenv
import os

# Load .env (only API_KEY is allowed per requirements)
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("Missing API_KEY in .env")

# ---- Hardcoded constants (as required) ----
# Azure OpenAI (chat+embeddings) deployments
AZURE_ENDPOINT = "https://096290-oai.openai.azure.com"
AZURE_API_VERSION = "2023-05-15"
AZURE_CHAT_DEPLOYMENT = "team6-gpt4o"
AZURE_EMBED_DEPLOYMENT = "team6-embedding"

# Online Vector DB (Qdrant Cloud) — hardcode URL + collection
# QDRANT_URL = "https://YOUR-QDRANT-URL:6333"  # e.g., https://xxxx-xxxx-xxxx.qdrant.tech:6333
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "lunar_notes_v1"

# Per requirement: single env var. Reuse API_KEY for Qdrant auth.
# (Set the same key in Qdrant Cloud → API Keys.)
QDRANT_API_KEY = API_KEY

# Safety knobs (to keep budget tight)
MAX_DOCS_INGEST = 15000      # limit uploaded notes
EMBED_BATCH = 128            # batch size for embeddings
TOP_K = 24                   # default retrieval
