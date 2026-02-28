"""
ChromaDB store operations for storing and retrieving product embeddings.
Uses sentence-transformers for embeddings and ChromaDB for vector storage.
"""

import json
import logging
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
)

logger = logging.getLogger(__name__)

# Module-level caches
_embedding_model = None
_chroma_client = None
_collection = None


# ===============================
# Embedding Model
# ===============================

def _get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embedding_model


def get_embedding(text: str) -> list[float]:
    """Generate an embedding vector for a text string."""
    model = _get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


# ===============================
# Chroma Initialization
# ===============================

def _get_collection():
    """Get or create Chroma collection."""
    global _chroma_client, _collection

    if _collection is None:
        _chroma_client = chromadb.Client(
            Settings(
                persist_directory=CHROMA_PERSIST_DIR,
                is_persistent=True,
            )
        )

        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    return _collection


def init_store():
    """Initialize ChromaDB collection."""
    _get_collection()
    logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready.")


# ===============================
# Duplicate Check
# ===============================

def video_exists(video_id: str) -> bool:
    """Check if a video has already been ingested."""
    collection = _get_collection()
    results = collection.get(
        where={"video_id": video_id}
    )
    return len(results["ids"]) > 0


# ===============================
# Store Transcript Chunks
# ===============================

def store_transcript_chunks(
    video_id: str,
    video_url: str,
    video_title: str,
    chunks: list[str],
    chunk_type: str = "transcript",
) -> int:
    """
    Store transcript chunks with embeddings in ChromaDB.
    """

    if not chunks:
        return 0

    model = _get_embedding_model()
    embeddings = model.encode(chunks, normalize_embeddings=True)

    collection = _get_collection()

    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{video_id}_{chunk_type}_{i}")

        metadatas.append({
            "video_id": video_id,
            "video_url": video_url,
            "video_title": video_title,
            "chunk_type": chunk_type,
        })

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(chunks)} chunks for video {video_id}.")
    return len(chunks)


# ===============================
# Store Products
# ===============================

def store_products(
    video_id: str,
    video_url: str,
    video_title: str,
    products: list[dict],
) -> int:

    if not products:
        return 0

    model = _get_embedding_model()
    collection = _get_collection()

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for i, product in enumerate(products):
        # Build rich embedding text
        product_text = (
            f"{product.get('name', '')} by {product.get('brand', '')}. "
            f"Category: {product.get('category', '')}. "
            f"Features: {', '.join(product.get('features', []))}. "
            f"{product.get('description', '')}"
        )

        embedding = model.encode(product_text, normalize_embeddings=True)

        ids.append(f"{video_id}_product_{i}")
        documents.append(product_text)
        embeddings.append(embedding.tolist())

        # 🔥 IMPORTANT: Flatten metadata (ONLY primitives allowed)
        metadatas.append({
            "video_id": str(video_id),
            "video_url": str(video_url),
            "video_title": str(video_title),
            "chunk_type": "product",
            "product_name": str(product.get("name", "Unknown")),
            "product_brand": str(product.get("brand", "Unknown")),
            "product_category": str(product.get("category", "general")),
            "features": ", ".join(product.get("features", [])),  # list → string
            "price_mentioned": str(product.get("price_mentioned") or ""),
            "description": str(product.get("description", "")),
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(products)} products for video {video_id}.")
    return len(products)

# ===============================
# Semantic Search
# ===============================

# ===============================
# Semantic Search
# ===============================

def search_similar(query: str, k: int = 5, video_id: str = None):
    collection = _get_collection()
    query_embedding = get_embedding(query)

    # Add where filter if video_id is provided
    where_filter = {"video_id": video_id} if video_id else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_filter
    )

    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    flattened_results = []

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i] or {}

        flattened_results.append({
            "text": results["documents"][0][i],
            "chunk_type": metadata.get("chunk_type", "transcript"),
            "product_name": metadata.get("product_name"),
            "product_brand": metadata.get("product_brand"),
            "product_category": metadata.get("product_category"),
            "metadata": metadata,
            "video_title": metadata.get("video_title"),
            "video_url": metadata.get("video_url"),
            "similarity": 1 - results["distances"][0][i]
            if results.get("distances")
            else None,
        })

    return flattened_results


# ===============================
# Get Products for Video
# ===============================

def get_products_for_video(video_id: str) -> list[dict[str, Any]]:
    """Get all products extracted from a specific video."""

    collection = _get_collection()

    results = collection.get(
        where={
            "$and": [
                {"video_id": video_id},
                {"chunk_type": "product"},
            ]
        }
    )

    products = []

    for metadata in results["metadatas"]:
        products.append({
            "name": metadata.get("product_name"),
            "brand": metadata.get("product_brand"),
            "category": metadata.get("product_category"),
            "metadata": {
                "features": metadata.get("features"),
                "price_mentioned": metadata.get("price_mentioned"),
                "description": metadata.get("description"),
            }
        })

    return products