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
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "lunar_notes_v1"
QDRANT_COLLECTION_MATERIALS = "lunar_materials_v1"  # NEW

# Per requirement: single env var. Reuse API_KEY for Qdrant auth.
# (Set the same key in Qdrant Cloud → API Keys.)
QDRANT_API_KEY = API_KEY

# Safety knobs (to keep budget tight)
MAX_DOCS_INGEST = 15000      # limit uploaded notes
EMBED_BATCH = 128            # batch size for embeddings




# --- RAG / Retrieval params ---
TOP_K_NOTES = 24
TOP_K_MATERIALS = 6

# --- Composer / Proportioner ---
ROLE_SPLIT = {"top": 30.0, "mid": 40.0, "base": 30.0}   # percent-of-100g
ROLE_TOLERANCE = 3.0                                     # ±%
ALLOW_MULTI_ROLE = False                                 # dedupe default

# --- Dataset builder ---
DATASET_OUT_DIR = "/Users/ranykhirbawi/Desktop/LunarAIccord/data/lunar_v1"
DATASET_SHARD_SIZE = 1000
