"""
Configuration for the Multi-Agent Product Recommender.
Federated A2A architecture with multiple services.
"""
import os

# --- Ollama / LLM ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = 0.3

# --- Whisper ---
WHISPER_MODEL = "base"  # CPU-friendly: tiny, base, small

# --- PostgreSQL / PGVector ---
# PG_HOST = os.getenv("PG_HOST", "localhost")
# PG_PORT = os.getenv("PG_PORT", "5432")
# PG_USER = os.getenv("PG_USER", "postgres")
# PG_PASSWORD = os.getenv("PG_PASSWORD", "root")
# PG_DATABASE = os.getenv("PG_DATABASE", "product_recommender")

# PG_CONNECTION_STRING = (
#     f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
# )
# # Sync connection string for psycopg2
# PG_SYNC_CONNECTION_STRING = (
#     f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
# )
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "product_recommender"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # for MiniLM

# --- Embeddings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# --- PGVector Collection ---
COLLECTION_NAME = "product_knowledge"

# --- Service Ports ---
ORCHESTRATOR_PORT = 8000
YOUTUBE_RAG_AGENT_PORT = 8001
PRODUCT_SEARCH_AGENT_PORT = 8002

ORCHESTRATOR_URL = f"http://localhost:{ORCHESTRATOR_PORT}"
YOUTUBE_RAG_AGENT_URL = f"http://localhost:{YOUTUBE_RAG_AGENT_PORT}"
PRODUCT_SEARCH_AGENT_URL = f"http://localhost:{PRODUCT_SEARCH_AGENT_PORT}"

# Agent URLs for discovery
AGENT_URLS = [
    YOUTUBE_RAG_AGENT_URL,
    PRODUCT_SEARCH_AGENT_URL,
]

# --- Search ---
SEARCH_PLATFORMS = [
    ("Amazon India", "amazon.in"),
    ("Flipkart", "flipkart.com"),
    ("Myntra", "myntra.com"),
    ("Ajio", "ajio.com"),
    ("Meesho", "meesho.com"),
    ("Lenskart", "lenskart.com"),
]
MAX_SEARCH_RESULTS_PER_PLATFORM = 5

# --- Paths ---
TEMP_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
