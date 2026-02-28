"""
Transcript Manager for YouTube RAG Agent.
Handles subtitle extraction (primary) with Whisper fallback,
duplicate detection, and text chunking for embedding.
"""
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from youtube_transcript_api import YouTubeTranscriptApi

from core.youtube_downloader import download_audio, cleanup_audio
from core.transcriber import transcribe_audio
from core.product_extractor import extract_products, Product
from core.vector_store import (
    init_store,
    video_exists,
    store_transcript_chunks,
    store_products,
)
from utils.helpers import validate_youtube_url, extract_video_id, chunk_text

logger = logging.getLogger(__name__)


def get_transcript_from_subtitles(video_id: str) -> str | None:
    """
    Try to get subtitles directly from YouTube (fast, free).

    Args:
        video_id: YouTube video ID.

    Returns:
        Transcript text, or None if subtitles are unavailable.
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        text = " ".join([entry["text"] for entry in transcript_list])
        logger.info(f"Subtitles extracted successfully for video {video_id}.")
        return text
    except Exception as e:
        logger.warning(f"Subtitles unavailable for video {video_id}: {e}")
        return None


def get_transcript_from_audio(video_url: str) -> tuple[str, dict]:
    """
    Fallback: Download audio and transcribe with Whisper.

    Returns:
        Tuple of (transcript_text, video_metadata).
    """
    logger.info("Falling back to Whisper audio transcription...")
    metadata = download_audio(video_url)
    transcript = transcribe_audio(metadata["audio_path"])
    cleanup_audio(metadata["audio_path"])
    return transcript, metadata


def ingest_video(youtube_url: str) -> dict:
    """
    Full ingestion pipeline for a YouTube video:
    1. Validate URL
    2. Check for duplicates
    3. Extract transcript (subtitles → Whisper fallback)
    4. Extract products via LLM
    5. Chunk and store in PGVector

    Returns:
        Summary dict with ingestion results.
    """
    # Initialize the vector store
    init_store()

    # Step 1: Validate URL
    is_valid, result = validate_youtube_url(youtube_url)
    if not is_valid:
        return {"status": "error", "message": result}

    video_id = result

    # Step 2: Check for duplicates
    if video_exists(video_id):
        logger.info(f"Video {video_id} already ingested. Skipping.")
        return {
            "status": "duplicate",
            "message": f"Video '{video_id}' has already been ingested.",
            "video_id": video_id,
        }

    # Step 3: Extract transcript
    logger.info(f"Ingesting video {video_id}...")
    transcript = get_transcript_from_subtitles(video_id)
    video_metadata = {"title": "Unknown", "channel": "Unknown", "url": youtube_url}

    if transcript:
        # We got subtitles — still get basic metadata
        video_metadata["title"] = f"YouTube Video {video_id}"
        video_metadata["source"] = "subtitles"
    else:
        # Fallback to Whisper
        transcript, audio_metadata = get_transcript_from_audio(youtube_url)
        video_metadata.update(audio_metadata)
        video_metadata["source"] = "whisper"

    if not transcript or len(transcript.strip()) < 20:
        return {
            "status": "error",
            "message": "Could not extract meaningful transcript from the video.",
        }

    logger.info(f"Transcript extracted ({len(transcript)} chars) via {video_metadata.get('source')}.")

    # Step 4: Extract products
    products = extract_products(transcript, video_metadata)
    product_dicts = [p.model_dump() for p in products]

    # Step 5: Chunk transcript and store
    chunks = chunk_text(transcript, chunk_size=500, overlap=50)
    chunks_stored = store_transcript_chunks(
        video_id=video_id,
        video_url=youtube_url,
        video_title=video_metadata.get("title", "Unknown"),
        chunks=chunks,
    )

    # Step 6: Store products
    products_stored = store_products(
        video_id=video_id,
        video_url=youtube_url,
        video_title=video_metadata.get("title", "Unknown"),
        products=product_dicts,
    )

    summary = {
        "status": "success",
        "video_id": video_id,
        "video_title": video_metadata.get("title", "Unknown"),
        "transcript_source": video_metadata.get("source", "unknown"),
        "transcript_length": len(transcript),
        "transcript_chunks_stored": chunks_stored,
        "products_found": len(products),
        "products_stored": products_stored,
        "products": product_dicts,
    }

    logger.info(f"Ingestion complete: {chunks_stored} chunks, {products_stored} products stored.")
    return summary
