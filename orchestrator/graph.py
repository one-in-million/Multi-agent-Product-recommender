"""
LangGraph Orchestrator for Multi-Agent Product Recommender.
Replaces the HTTP-based A2A protocol with a unified graph state.
"""
import sys
import os
import logging
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# Import your existing core logic directly!
from orchestrator.router import OrchestratorRouter
from agents.youtube_rag.transcript_manager import ingest_video
from agents.youtube_rag.rag_chain import chat_about_video
from agents.product_search.collector import collect_product_requirements
from agents.product_search.mcp_search import mcp_search_products

logger = logging.getLogger(__name__)
# Setup logger (put this at the top of graph.py)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Define the Graph State ---
class AgentState(TypedDict):
    messages: list[dict]       # Standard chat history format
    current_intent: str        # Where the router wants to go
    agent_used: str | None     # Which agent handled it
    products: list[dict]       # Any products found
    final_response: str        # The text to show the user

# --- 2. Define the Nodes ---

async def router_node(state: AgentState):
    logger.info("🟢 [NODE: ROUTER] Starting intent classification...")
    """Classifies user intent."""
    router = OrchestratorRouter()
    last_message = state["messages"][-1]["content"]
    
    # Let the LLM guess the intent naturally
    intent = await router.classify_intent(last_message)
    
    return {"current_intent": intent}

def ingest_node(state: AgentState):
    logger.info("🟣 [NODE: INGEST] Starting video ingestion...")
    """Handles video ingestion."""
    last_message = state["messages"][-1]["content"]
    
    # Run your exact ingestion logic
    result = ingest_video(last_message.strip())
    
    status = result.get("status")
    if status == "duplicate":
        msg = " This video has already been ingested. You can ask questions about it!"
    elif status == "success":
        products = result.get("products", [])
        product_list = "\n".join([f"  • **{p.get('name', 'Unknown')}** by {p.get('brand', 'Unknown')}" for p in products])
        msg = f"✅ **Video ingested successfully!**\n📹 **Title**: {result.get('video_title')}\n📦 **Products found**: {result.get('products_found')}"
        if product_list:
            msg += f"\n\n**Extracted Products:**\n{product_list}"
    else:
        msg = f"❌ Ingestion failed: {result.get('message')}"

    return {
        "final_response": msg,
        "agent_used": "YouTube RAG Agent",
        "products": result.get("products", [])
    }

def rag_node(state: AgentState):
    logger.info("🔵 [NODE: RAG] Searching vector store and generating answer...")
    """Handles Q&A about the video."""
    query = state["messages"][-1]["content"]
    chat_history = state["messages"][:-1] # Exclude current query
    
    result = chat_about_video(query, chat_history=chat_history)
    
    return {
        "final_response": result.get("answer", "I couldn't find an answer."),
        "agent_used": "YouTube RAG Agent",
        "products": []
    }

def search_node(state: AgentState):
    """Handles searching for products."""
    logger.info("🟠 [NODE: SEARCH] Collecting requirements for product search...")
    query = state["messages"][-1]["content"]
    
    # Safely get context
    context_str = " | ".join([m["content"] for m in state["messages"][-4:]])
    
    requirements = collect_product_requirements(query, context=context_str)
    
    mcp_result = mcp_search_products(
        product_query=requirements.get("product_query", query),
        search_queries=requirements.get("search_queries", [query]),
        category=requirements.get("category", "general"),
        brand=requirements.get("brand"),
        budget=requirements.get("budget"),
    )
    
    # --- THE FIX: Safely check if it's a failure object ---
    if type(mcp_result).__name__ == "MCPFailureResponse":
        message = getattr(mcp_result, "message", "No products could be extracted.")
        error_msg = (
            f"🔍 **Search completed, but no prices were found.**\n\n"
            f"{message}\n\n"
            f"*(Note: Playwright attempted to scrape the platforms, but was blocked by anti-bot firewalls like Cloudflare.)*"
        )
        return {
            "final_response": error_msg,
            "agent_used": "Product Search MCP Agent",
            "products": []
        }
        
    # If successful, it's an MCPSuccessResponse, which has a .data attribute
    data = getattr(mcp_result, "data", {})
    comparison = data.get("comparison_summary", "No comparison available.")
    
    msg = f"🔍 **Product Search Results**\n\n{comparison}"
    
    flat_products = []
    for platform, items in data.get("results", {}).items():
        flat_products.extend(items)
        
    return {
        "final_response": msg,
        "agent_used": "Product Search MCP Agent",
        "products": flat_products
    }

def general_node(state: AgentState):
    """Handles basic greetings."""
    return {
        "final_response": "Hello! Paste a YouTube URL to ingest it, ask questions about a video, or ask me to search for products online.",
        "agent_used": "Orchestrator",
        "products": []
    }

# --- 3. Routing Logic ---
def route_decision(state: AgentState) -> str:
    """Decides which node to call based on the router's intent."""
    intent = state["current_intent"]
    if intent == "ingest_video": return "ingest"
    if intent == "chat_about_video": return "rag"
    if intent == "search_products": return "search"
    return "general"

# --- 4. Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("ingest", ingest_node)
workflow.add_node("rag", rag_node)
workflow.add_node("search", search_node)
workflow.add_node("general", general_node)

# Set entry point
workflow.set_entry_point("router")

# Add conditional edges
workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "ingest": "ingest",
        "rag": "rag",
        "search": "search",
        "general": "general"
    }
)

# All nodes go to END
workflow.add_edge("ingest", END)
workflow.add_edge("rag", END)
workflow.add_edge("search", END)
workflow.add_edge("general", END)

# Compile!
app_graph = workflow.compile()

async def run_agent(message: str, chat_history: list[dict]) -> dict:
    """Entry point for Streamlit UI."""
    messages = chat_history + [{"role": "user", "content": message}]
    
    # Run the graph
    final_state = await app_graph.ainvoke({"messages": messages})
    
    return {
        "message": final_state.get("final_response", "Error processing request."),
        "agent_used": final_state.get("agent_used"),
        "products": final_state.get("products", [])
    }