"""
DuckDuckGo-based product search across e-commerce platforms.
"""
import logging
from dataclasses import dataclass


from duckduckgo_search import DDGS

from config import SEARCH_PLATFORMS, MAX_SEARCH_RESULTS_PER_PLATFORM

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single product search result."""
    title: str
    url: str
    snippet: str
    platform: str
    platform_domain: str


def search_product_on_platform(
    product_query: str,
    platform_name: str,
    platform_domain: str,
    max_results: int = MAX_SEARCH_RESULTS_PER_PLATFORM,
) -> list[SearchResult]:
    """
    Search for a product on a specific platform using DuckDuckGo.

    Args:
        product_query: Product name/description to search for.
        platform_name: Display name of the platform (e.g., "Amazon India").
        platform_domain: Domain to restrict search to (e.g., "amazon.in").
        max_results: Maximum number of results per platform.

    Returns:
        List of SearchResult objects.
    """
    query = f"{product_query} site:{platform_domain}"
    results = []

    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))
            for r in search_results:
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    platform=platform_name,
                    platform_domain=platform_domain,
                ))
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for '{query}': {e}")

    return results


def search_product_across_platforms(
    product_query: str,
    platforms: list[tuple[str, str]] | None = None,
) -> dict[str, list[SearchResult]]:
    """
    Search for a product across all configured e-commerce platforms.

    Args:
        product_query: Product name/description to search.
        platforms: Optional list of (name, domain) tuples. Defaults to SEARCH_PLATFORMS.

    Returns:
        Dict mapping platform name to list of SearchResult objects.
    """
    if platforms is None:
        platforms = SEARCH_PLATFORMS

    all_results = {}
    for platform_name, platform_domain in platforms:
        logger.info(f"Searching '{product_query}' on {platform_name}...")
        results = search_product_on_platform(
            product_query, platform_name, platform_domain
        )
        if results:
            all_results[platform_name] = results
        logger.info(f"  Found {len(results)} results on {platform_name}.")

    return all_results


def format_results_for_llm(results: dict[str, list[SearchResult]]) -> str:
    """Format search results into a string for LLM processing."""
    if not results:
        return "No search results found across any platform."

    output = []
    for platform, items in results.items():
        output.append(f"\n### {platform}")
        for i, item in enumerate(items, 1):
            output.append(f"{i}. **{item.title}**")
            output.append(f"   URL: {item.url}")
            output.append(f"   {item.snippet}")

    return "\n".join(output)
