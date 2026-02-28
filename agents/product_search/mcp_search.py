"""
MCP-compliant product search wrapper.
Uses the direct async scraper to pull prices from Amazon, Flipkart, Myntra, and Croma.
"""
import sys
import os
import logging
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_ollama import ChatOllama
from a2a.models import MCPSuccessResponse, MCPFailureResponse
from search.direct_search import compare_prices
from config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

COMPARISON_PROMPT = """You are a product comparison expert. Analyze the following scraped prices from multiple e-commerce platforms and create a helpful comparison summary.

PRODUCT SEARCHED: {product_query}
CATEGORY: {category}

SEARCH RESULTS:
{search_results}

Create a comparison summary that includes:
1. Top recommendations across platforms
2. Price range observed
3. Best value recommendation

Keep it concise, useful, and well-formatted with bullet points.

COMPARISON SUMMARY:"""
import sys
import asyncio
if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

def mcp_search_products(
    product_query: str,
    search_queries: list[str],
    category: str = "general",
    brand: str | None = None,
    budget: str | None = None,
) -> MCPSuccessResponse | MCPFailureResponse:
    
    import sys
    import asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        # 1. Run your async scraper synchronously for the LangGraph node
        logger.info(f"Running direct scraper for: {product_query}")
        scraped_data = asyncio.run(compare_prices(product_query=product_query, max_results_per_store=10))

        if not scraped_data:
            return MCPFailureResponse(
                error_code="NO_RESULTS",
                message=f"No search results found for '{product_query}' across any platform.",
            )

        # 2. Format the scraped results for the LLM and the UI
        formatted_for_llm = []
        ui_results = {"results": {}}
        
        for item in scraped_data:
            store = item.get("store", "Unknown").capitalize()
            title = item.get("title", "Unknown")
            price = float(item.get("price", 0))
            url = item.get("product_url", "#")
            
            # --- SMART FILTER: Hide cheap accessories if searching for a phone ---
            if "phone" in product_query.lower() and price < 8000:
                continue
            
            # Group results by their actual platform name!
            if store not in ui_results["results"]:
                ui_results["results"][store] = []
                
            # Text for LLM
            formatted_for_llm.append(f"- {store}: {title} | ₹{price}")
            
            # Dictionary for Streamlit UI
            ui_results["results"][store].append({
                "title": f"₹{price} - {title}",
                "url": url,
                "snippet": f"Found on {store}",
                "platform": store,
            })

        # Fallback if the smart filter accidentally deletes everything
        if not formatted_for_llm:
            return MCPFailureResponse(
                error_code="NO_RESULTS",
                message="Found accessories, but no actual phones matching that query.",
            )

        results_text = "\n".join(formatted_for_llm)

        # 3. Generate comparison with LLM
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
        )

        comparison_prompt = COMPARISON_PROMPT.format(
            product_query=product_query,
            category=category,
            search_results=results_text,
        )

        logger.info("Generating LLM comparison summary...")
        comparison = llm.invoke(comparison_prompt).content.strip()

        # 4. Return success response
        return MCPSuccessResponse(
            message=f"Found {len(formatted_for_llm)} products",
            data={
                "product_query": product_query,
                "category": category,
                "results": ui_results["results"],
                "comparison_summary": comparison,
            },
        )

    except Exception as e:
        logger.error(f"MCP search failed: {e}", exc_info=True)
        return MCPFailureResponse(
            error_code="SEARCH_ERROR",
            message=f"Product search failed: {str(e)}",
        )