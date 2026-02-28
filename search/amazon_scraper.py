"""
Optional Amazon.in product page scraper using BeautifulSoup.
Extracts structured product data (title, price, rating, image).
Falls back gracefully if scraping fails due to anti-bot measures.
"""
import logging
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}


def scrape_amazon_product(url: str) -> dict | None:
    """
    Scrape product details from an Amazon.in product page.

    Args:
        url: Amazon.in product URL.

    Returns:
        Dict with title, price, rating, image_url, or None if scraping fails.
    """
    if "amazon" not in url.lower():
        return None

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Amazon scrape returned status {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract product title
        title_tag = soup.find("span", {"id": "productTitle"})
        title = title_tag.get_text(strip=True) if title_tag else "Unknown Product"

        # Extract price
        price = None
        price_tag = soup.find("span", {"class": "a-price-whole"})
        if price_tag:
            price = f"₹{price_tag.get_text(strip=True)}"
        else:
            # Try alternate price selector
            price_tag = soup.find("span", {"id": "priceblock_dealprice"})
            if price_tag:
                price = price_tag.get_text(strip=True)

        # Extract rating
        rating = None
        rating_tag = soup.find("span", {"class": "a-icon-alt"})
        if rating_tag:
            rating_text = rating_tag.get_text(strip=True)
            match = re.search(r"(\d+\.?\d*)", rating_text)
            if match:
                rating = match.group(1)

        # Extract main image
        image_url = None
        img_tag = soup.find("img", {"id": "landingImage"})
        if img_tag:
            image_url = img_tag.get("src")

        return {
            "title": title,
            "price": price,
            "rating": rating,
            "image_url": image_url,
            "source_url": url,
            "platform": "Amazon India",
        }

    except Exception as e:
        logger.warning(f"Failed to scrape Amazon product at {url}: {e}")
        return None
