"""
Orchestrator Agent — FastAPI Server.
Central hub that discovers agents, routes requests, and aggregates responses.
Runs on port 8000.
"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from a2a.models import ChatRequest, ChatResponse
from orchestrator.router import OrchestratorRouter
from config import ORCHESTRATOR_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Orchestrator Agent",
    description="Central orchestrator for the multi-agent product recommender system",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router instance
router = OrchestratorRouter()


@app.on_event("startup")
async def startup():
    """Discover agents on startup."""
    logger.info("Orchestrator starting — discovering agents...")
    agents = await router.discover_agents()
    if agents:
        logger.info(f"Discovered {len(agents)} agents: {list(agents.keys())}")
    else:
        logger.warning("No agents discovered. Make sure agent servers are running.")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Accepts a user message, routes to the appropriate agent,
    and returns the aggregated response.
    """
    logger.info(f"Chat request: '{request.message[:80]}...'")

    result = await router.route_request(
        message=request.message,
        chat_history=request.chat_history,
    )

    return ChatResponse(
        message=result.get("message", "Something went wrong."),
        agent_used=result.get("agent_used"),
        products=result.get("products", []),
        metadata=result.get("metadata", {}),
    )


@app.post("/discover")
async def rediscover_agents():
    """Force re-discovery of agents."""
    agents = await router.discover_agents()
    return {
        "discovered": len(agents),
        "agents": [
            {"name": card.name, "url": card.url, "capabilities": card.capabilities}
            for card in agents.values()
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "Orchestrator",
        "discovered_agents": list(router.agents.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ORCHESTRATOR_PORT)
