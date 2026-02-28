"""
LLM-based product extraction from video transcripts.
Uses Qwen2.5:7b via Ollama to identify and structure product information.
"""
import json
from pydantic import BaseModel
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE


class Product(BaseModel):
    """Structured product information extracted from a video."""
    name: str
    brand: str
    category: str
    features: list[str]
    price_mentioned: str | None = None
    description: str


EXTRACTION_PROMPT = """You are a product information extraction expert. Analyze the following video transcript and extract ALL products mentioned in it.

VIDEO TITLE: {title}
VIDEO CHANNEL: {channel}

TRANSCRIPT:
{transcript}

Extract each distinct product mentioned. For each product, provide:
- name: The product name/model
- brand: The brand or company
- category: Product category (e.g., "smartphone", "spectacles", "TV", "refrigerator", "shoes", etc.)
- features: Key features mentioned (list of strings)
- price_mentioned: Any price mentioned (as string, or null if not mentioned)
- description: A brief summary of what was said about this product (2-3 sentences)

IMPORTANT:
- Extract ALL products, even if only briefly mentioned
- If a brand is mentioned without a specific product, still extract it with the brand name as the product name
- Be accurate — only extract information that is actually in the transcript
- If no products are found, return an empty list

Respond ONLY with a valid JSON array of products. No other text.

Example format:
[
  {{
    "name": "Galaxy S24 Ultra",
    "brand": "Samsung",
    "category": "smartphone",
    "features": ["200MP camera", "S-Pen", "titanium frame"],
    "price_mentioned": "₹1,29,999",
    "description": "The Samsung Galaxy S24 Ultra was reviewed as a premium flagship phone with excellent camera capabilities."
  }}
]

JSON Output:"""


def extract_products(transcript: str, video_metadata: dict) -> list[Product]:
    """
    Extract product information from a video transcript using LLM.

    Args:
        transcript: The full transcript text.
        video_metadata: dict with title, channel, description, url.

    Returns:
        List of Product objects.
    """
    if not transcript or len(transcript.strip()) < 50:
        print("[ProductExtractor] Transcript too short, no products to extract.")
        return []

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    # Truncate very long transcripts to fit within context window
    max_chars = 12000
    truncated_transcript = transcript[:max_chars]
    if len(transcript) > max_chars:
        truncated_transcript += "\n... [transcript truncated]"

    prompt = EXTRACTION_PROMPT.format(
        title=video_metadata.get("title", "Unknown"),
        channel=video_metadata.get("channel", "Unknown"),
        transcript=truncated_transcript,
    )

    print("[ProductExtractor] Sending transcript to LLM for product extraction...")
    response = llm.invoke(prompt)
    raw_response = response.content.strip()

    # Parse JSON response
    products = _parse_products(raw_response)
    print(f"[ProductExtractor] Extracted {len(products)} products.")
    return products


def _parse_products(raw_response: str) -> list[Product]:
    """Parse the LLM response into Product objects."""
    # Try to extract JSON from the response
    # Sometimes LLM wraps it in markdown code blocks
    text = raw_response
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    text = text.strip()

    try:
        data = json.loads(text)
        if not isinstance(data, list):
            data = [data]

        products = []
        for item in data:
            try:
                product = Product(
                    name=item.get("name", "Unknown Product"),
                    brand=item.get("brand", "Unknown Brand"),
                    category=item.get("category", "general"),
                    features=item.get("features", []),
                    price_mentioned=item.get("price_mentioned"),
                    description=item.get("description", "No description available."),
                )
                products.append(product)
            except Exception as e:
                print(f"[ProductExtractor] Skipping malformed product entry: {e}")
                continue

        return products
    except json.JSONDecodeError as e:
        print(f"[ProductExtractor] Failed to parse LLM response as JSON: {e}")
        print(f"[ProductExtractor] Raw response: {raw_response[:500]}")
        return []
