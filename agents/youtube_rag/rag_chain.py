"""
Conversational RAG chain for Q&A over video transcripts.
Uses PGVector for retrieval and Qwen2.5 for generation.
"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_ollama import ChatOllama
from core.vector_store import search_similar
from config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

RAG_PROMPT = """You are an AI assistant that answers questions about YouTube video content. Use the provided context from video transcripts to answer the user's question accurately.

CONTEXT FROM VIDEO TRANSCRIPTS:
{context}

CHAT HISTORY:
{chat_history}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the provided context from the video transcripts
- If the context doesn't contain enough information, say so honestly
- Reference specific products, brands, or details mentioned in the video
- Keep your answer concise but informative
- If the user asks about products, list them with available details (brand, features, price)

ANSWER:"""


def format_chat_history(chat_history: list[dict[str, str]]) -> str:
    """Format chat history into a string for the prompt."""
    if not chat_history:
        return "No previous conversation."

    formatted = []
    for msg in chat_history[-6:]:  # Keep last 6 messages for context
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted.append(f"{role.upper()}: {content}")

    return "\n".join(formatted)


from utils.helpers import extract_video_id

def chat_about_video(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
    k: int = 5,
) -> dict:
    if chat_history is None:
        chat_history = []

    # --- NEW: Extract the latest video ID from chat history ---
    current_video_id = None
    for msg in reversed(chat_history):
        content = msg.get("content", "")
        # The UI adds URLs like "📹 https://youtube.com/..."
        if "youtube.com" in content or "youtu.be" in content:
            current_video_id = extract_video_id(content)
            break

    # Pass the video_id to the vector store!
    logger.info(f"RAG query: '{query}' — filtering by video_id: {current_video_id}")
    results = search_similar(query, k=k, video_id=current_video_id)

    if not results:
        return {
            "answer": "I don't have any relevant video content to answer from.",
            "sources": [],
        }

    # Step 2: Build context from retrieved chunks
    context_parts = []
    sources = []
    print("DEBUG RESULTS:", results)
    for r in results:
        chunk_type = r.get("chunk_type", "transcript").upper()
        text = r.get("text", "")

        context_parts.append(f"[{chunk_type}] {text}")

        sources.append({
        "video_title": r.get("video_title", "Unknown"),
        "video_url": r.get("video_url", ""),
        "chunk_type": chunk_type,
        "product_name": r.get("product_name"),
        "similarity": round(r.get("similarity", 0) or 0, 3),
        })

    context = "\n\n".join(context_parts)
    history_str = format_chat_history(chat_history)

    # Step 3: Generate answer with LLM
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    prompt = RAG_PROMPT.format(
        context=context,
        chat_history=history_str,
        question=query,
    )

    logger.info("Generating RAG answer...")
    response = llm.invoke(prompt)
    answer = response.content.strip()

    return {
        "answer": answer,
        "sources": sources,
    }
