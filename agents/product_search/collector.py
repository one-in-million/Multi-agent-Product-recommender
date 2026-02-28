"""
Product Requirement Collector for Agent 2.
Uses LLM to extract and validate product search requirements from user input.
"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_ollama import ChatOllama
from config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from core.vector_store import search_similar

logger = logging.getLogger(__name__)

COLLECTION_PROMPT = """You are a product search assistant. Based on the user's query and any available context from video transcripts, extract structured product search parameters.

USER QUERY: {query}

CONTEXT FROM VIDEO TRANSCRIPTS:
{context}

Extract the following information (return as JSON):
- product_query: The main product to search for (be specific, include brand if mentioned)
- category: Product category (e.g., "smartphone", "TV", "spectacles", "shoes")
- brand: Brand name if specified (or null)
- budget: Budget range if mentioned (or null)
- search_queries: A list of 2-3 specific product variations to search for. 

IMPORTANT RULES FOR search_queries:
- MUST be exact product names/models only (e.g., ["Nothing Phone 3", "Nothing Phone 3 256GB"])
- NEVER include conversational phrases like "price comparison", "best deals", "buy", or "across platforms".
- Include the brand name in the queries.

Respond with ONLY valid JSON:
{{
    "product_query": "...",
    "category": "...",
    "brand": "..." or null,
    "budget": "..." or null,
    "search_queries": ["query1", "query2"]
}}

JSON:"""

def collect_product_requirements(
    query: str,
    context: str | None = None,
) -> dict:
    """
    Extract structured product search requirements from a user query.

    Args:
        query: User's natural language query.
        context: Optional context from video transcripts.

    Returns:
        Dict with product_query, category, brand, budget, search_queries.
    """
    # Get video context if not provided
    if not context:
        try:
            results = search_similar(query, k=3)
            if results:
                context = "\n".join([r["text"] for r in results])
            else:
                context = "No video context available."
        except Exception:
            context = "No video context available."

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    prompt = COLLECTION_PROMPT.format(query=query, context=context)
    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Parse JSON
    import json
    try:
        # Clean up markdown code blocks if present
        text = raw
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        requirements = json.loads(text.strip())
        logger.info(f"Collected requirements: {requirements}")
        return requirements
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse collector response: {raw[:200]}")
        # Fallback: use the query as-is
        return {
            "product_query": query,
            "category": "general",
            "brand": None,
            "budget": None,
            "search_queries": [query],
        }
