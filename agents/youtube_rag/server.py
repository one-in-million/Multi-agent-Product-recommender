"""
Agent 1: YouTube RAG Agent — FastAPI Server.
Exposes Agent Card and handles video ingestion + conversational RAG tasks.
Runs on port 8001.
"""
import json
import sys
import os
import logging
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from a2a.models import A2ATask, A2AResponse, TaskStatus
from agents.youtube_rag.transcript_manager import ingest_video
from agents.youtube_rag.rag_chain import chat_about_video
from config import YOUTUBE_RAG_AGENT_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Agent Card
AGENT_CARD_PATH = os.path.join(os.path.dirname(__file__), "agent_card.json")
with open(AGENT_CARD_PATH) as f:
    AGENT_CARD = json.load(f)

# FastAPI app
app = FastAPI(
    title="YouTube RAG Agent",
    description="A2A-compatible agent for YouTube video transcript extraction and conversational RAG",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/agent-card")
async def get_agent_card():
    """Return the Agent Card describing this agent's capabilities."""
    return AGENT_CARD


@app.post("/tasks")
async def handle_task(task: A2ATask) -> A2AResponse:
    """
    Handle an A2A task.
    Supported capabilities: ingest_video, chat_about_video
    """
    logger.info(f"Received task: {task.task_id} — capability: {task.capability}")

    try:
        if task.capability == "ingest_video":
            return await _handle_ingest(task)
        elif task.capability == "chat_about_video":
            return await _handle_chat(task)
        else:
            return A2AResponse(
                task_id=task.task_id,
                status="failure",
                error={
                    "error_code": "UNKNOWN_CAPABILITY",
                    "message": f"Unknown capability: {task.capability}. "
                               f"Supported: ingest_video, chat_about_video",
                },
            )
    except Exception as e:
        logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)
        return A2AResponse(
            task_id=task.task_id,
            status="failure",
            error={
                "error_code": "INTERNAL_ERROR",
                "message": str(e),
            },
        )


async def _handle_ingest(task: A2ATask) -> A2AResponse:
    """Handle video ingestion task."""
    youtube_url = task.input_data.get("youtube_url")
    if not youtube_url:
        return A2AResponse(
            task_id=task.task_id,
            status="failure",
            error={
                "error_code": "MISSING_INPUT",
                "message": "youtube_url is required in input_data",
            },
        )

    logger.info(f"Ingesting video: {youtube_url}")
    result = ingest_video(youtube_url)

    status = "success" if result.get("status") in ("success", "duplicate") else "failure"

    if status == "failure":
        return A2AResponse(
            task_id=task.task_id,
            status="failure",
            error={
                "error_code": "INGESTION_FAILED",
                "message": result.get("message", "Unknown error"),
            },
        )

    return A2AResponse(
        task_id=task.task_id,
        status="success",
        result=result,
    )


async def _handle_chat(task: A2ATask) -> A2AResponse:
    """Handle conversational Q&A task."""
    query = task.input_data.get("query")
    if not query:
        return A2AResponse(
            task_id=task.task_id,
            status="failure",
            error={
                "error_code": "MISSING_INPUT",
                "message": "query is required in input_data",
            },
        )

    chat_history = task.input_data.get("chat_history", [])


    logger.info(f"RAG query: {query}")
    result = await asyncio.to_thread(
    chat_about_video,
    query,
    chat_history
)

    return A2AResponse(
        task_id=task.task_id,
        status="success",
        result=result,
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "YouTube RAG Agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=YOUTUBE_RAG_AGENT_PORT)
