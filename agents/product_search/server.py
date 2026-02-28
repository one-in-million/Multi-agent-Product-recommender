"""
Agent 2: Product Search MCP Agent — FastAPI Server.
Exposes Agent Card and handles product search + price comparison tasks.
Runs on port 8002.
"""
import json
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from a2a.models import A2ATask, A2AResponse
from agents.product_search.collector import collect_product_requirements
from agents.product_search.mcp_search import mcp_search_products
from config import PRODUCT_SEARCH_AGENT_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Agent Card
AGENT_CARD_PATH = os.path.join(os.path.dirname(__file__), "agent_card.json")
with open(AGENT_CARD_PATH) as f:
    AGENT_CARD = json.load(f)

# FastAPI app
app = FastAPI(
    title="Product Search MCP Agent",
    description="A2A-compatible agent for product search across e-commerce platforms",
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
    Supported capabilities: search_products, compare_prices
    """
    logger.info(f"Received task: {task.task_id} — capability: {task.capability}")

    try:
        if task.capability in ("search_products", "compare_prices"):
            return await _handle_search(task)
        else:
            return A2AResponse(
                task_id=task.task_id,
                status="failure",
                error={
                    "error_code": "UNKNOWN_CAPABILITY",
                    "message": f"Unknown capability: {task.capability}. "
                               f"Supported: search_products, compare_prices",
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


async def _handle_search(task: A2ATask) -> A2AResponse:
    """Handle product search / price comparison task."""
    product_query = task.input_data.get("product_query")
    if not product_query:
        return A2AResponse(
            task_id=task.task_id,
            status="failure",
            error={
                "error_code": "MISSING_INPUT",
                "message": "product_query is required in input_data",
            },
        )

    context = task.input_data.get("context")

    # Step 1: Collect and validate product requirements
    logger.info(f"Collecting product requirements for: {product_query}")
    requirements = collect_product_requirements(product_query, context)

    # Step 2: Execute MCP-compliant search
    logger.info(f"Executing MCP search with queries: {requirements.get('search_queries', [])}")
    mcp_result = mcp_search_products(
        product_query=requirements.get("product_query", product_query),
        search_queries=requirements.get("search_queries", [product_query]),
        category=requirements.get("category", "general"),
        brand=requirements.get("brand"),
        budget=requirements.get("budget"),
    )

    # Step 3: Return A2A response wrapping the MCP result
    mcp_dict = mcp_result.model_dump()
    is_success = mcp_dict.get("status") == "success"

    if is_success:
        return A2AResponse(
            task_id=task.task_id,
            status="success",
            result=mcp_dict,
        )
    else:
        return A2AResponse(
            task_id=task.task_id,
            status="failure",
            error=mcp_dict,
        )


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "Product Search MCP Agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PRODUCT_SEARCH_AGENT_PORT)
