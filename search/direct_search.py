import asyncio
import json
import re
from urllib.parse import quote_plus
from playwright_stealth import Stealth # pyright: ignore[reportMissingImports] # <-- ADD THIS

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, BrowserContext


def parse_inr_price(*, raw_text: str) -> float | None:
    matched_value = re.search(r"(?:₹|Rs\.?\s?)\s*([\d,]+(?:\.\d{1,2})?)", raw_text)
    if not matched_value:
        matched_value = re.search(r"\b([\d]{1,3}(?:,[\d]{2,3})+(?:\.\d{1,2})?)\b", raw_text)
    if not matched_value:
        return None
    normalized_value = matched_value.group(1).replace(",", "")
    try:
        return float(normalized_value)
    except ValueError:
        return None


def clean_text(*, value: str) -> str:
    return " ".join(value.split())


def parse_json_ld_products(*, html: str, store: str, max_results: int) -> list[dict[str, str | float]]:
    soup = BeautifulSoup(html, "html.parser")
    parsed_items: list[dict[str, str | float]] = []
    scripts = soup.select("script[type='application/ld+json']")

    for script_tag in scripts:
        script_text = script_tag.string or script_tag.get_text(strip=True)
        if not script_text:
            continue
        try:
            payload = json.loads(script_text)
        except Exception:
            continue

        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = clean_text(value=str(entry.get("name", "")))
            url = str(entry.get("url", ""))
            offers = entry.get("offers")
            offer_entries = offers if isinstance(offers, list) else [offers]
            price = None
            for offer in offer_entries:
                if isinstance(offer, dict) and offer.get("price") is not None:
                    try:
                        price = float(str(offer.get("price")).replace(",", ""))
                    except ValueError:
                        price = None
                    if price is not None:
                        break
            if not name or not url or price is None:
                continue
            parsed_items.append(
                {
                    "store": store,
                    "title": name,
                    "price": price,
                    "currency": "INR",
                    "product_url": url,
                }
            )
            if len(parsed_items) >= max_results:
                return parsed_items
    return parsed_items


def parse_amazon_products(*, html: str, max_results: int) -> list[dict[str, str | float]]:
    soup = BeautifulSoup(html, "html.parser")
    parsed_items: list[dict[str, str | float]] = []
    cards = soup.select("div[data-component-type='s-search-result']")
    for card in cards:
        title_element = card.select_one("h2 span")
        price_element = (
            card.select_one("span.a-price > span.a-offscreen")
            or card.select_one("span.a-price-whole")
            or card.select_one("span.a-offscreen")
        )
        link_element = card.select_one("h2 a")
        if not title_element or not price_element or not link_element:
            continue
        parsed_price = parse_inr_price(raw_text=price_element.get_text(strip=True))
        if parsed_price is None:
            continue
        raw_href = str(link_element.get("href", ""))
        product_url = raw_href if raw_href.startswith("http") else f"https://www.amazon.in{raw_href}"
        parsed_items.append(
            {
                "store": "amazon",
                "title": clean_text(value=title_element.get_text(strip=True)),
                "price": parsed_price,
                "currency": "INR",
                "product_url": product_url,
            }
        )
        if len(parsed_items) >= max_results:
            return parsed_items
    return parsed_items


def parse_flipkart_products(*, html: str, max_results: int) -> list[dict[str, str | float]]:
    soup = BeautifulSoup(html, "html.parser")
    parsed_items: list[dict[str, str | float]] = []
    cards = soup.select("div[data-id]")
    for card in cards:
        title_element = card.select_one("a[title]") or card.select_one("div.KzDlHZ")
        price_element = card.select_one("div.Nx9bqj") or card.select_one("div._30jeq3")
        link_element = card.select_one("a[href]")
        if not title_element or not price_element or not link_element:
            continue
        parsed_price = parse_inr_price(raw_text=price_element.get_text(strip=True))
        if parsed_price is None:
            continue
        title_value = title_element.get("title") or title_element.get_text(strip=True)
        parsed_items.append(
            {
                "store": "flipkart",
                "title": clean_text(value=title_value),
                "price": parsed_price,
                "currency": "INR",
                "product_url": f"https://www.flipkart.com{link_element.get('href', '')}",
            }
        )
        if len(parsed_items) >= max_results:
            return parsed_items
    return parsed_items


def parse_myntra_products(*, html: str, max_results: int) -> list[dict[str, str | float]]:
    soup = BeautifulSoup(html, "html.parser")
    parsed_items: list[dict[str, str | float]] = []
    cards = soup.select("li.product-base")
    for card in cards:
        title_brand = card.select_one("h3.product-brand")
        title_name = card.select_one("h4.product-product")
        price_element = card.select_one("span.product-discountedPrice")
        link_element = card.select_one("a")
        if not title_name or not price_element or not link_element:
            continue
        parsed_price = parse_inr_price(raw_text=price_element.get_text(strip=True))
        if parsed_price is None:
            continue
        title_text = clean_text(
            value=f"{title_brand.get_text(strip=True) if title_brand else ''} {title_name.get_text(strip=True)}"
        )
        parsed_items.append(
            {
                "store": "myntra",
                "title": title_text,
                "price": parsed_price,
                "currency": "INR",
                "product_url": f"https://www.myntra.com/{link_element.get('href', '').lstrip('/')}",
            }
        )
        if len(parsed_items) >= max_results:
            return parsed_items
    return parsed_items


def parse_croma_products(*, html: str, max_results: int) -> list[dict[str, str | float]]:
    soup = BeautifulSoup(html, "html.parser")
    parsed_items: list[dict[str, str | float]] = []
    cards = soup.select("li.product-item")
    for card in cards:
        title_element = card.select_one("h3.product-title") or card.select_one("h3")
        price_element = card.select_one("span.amount") or card.select_one(".new-price")
        link_element = card.select_one("a[href]")
        if not title_element or not price_element or not link_element:
            continue
        parsed_price = parse_inr_price(raw_text=price_element.get_text(strip=True))
        if parsed_price is None:
            continue
        raw_href = link_element.get("href", "")
        product_url = raw_href if raw_href.startswith("http") else f"https://www.croma.com{raw_href}"
        parsed_items.append(
            {
                "store": "croma",
                "title": clean_text(value=title_element.get_text(strip=True)),
                "price": parsed_price,
                "currency": "INR",
                "product_url": product_url,
            }
        )
        if len(parsed_items) >= max_results:
            return parsed_items
    return parsed_items


async def fetch_page_with_playwright(context: BrowserContext, url: str) -> str:
    """Navigates to a URL using Playwright Stealth and extracts the HTML."""
    try:
        page = await context.new_page()
        
        # Notice: The stealth_async line is completely gone! 
        
        # Go to the page and wait for the network to settle
        # Wait for the basic DOM, then rely on our manual 3-second timeout for the JS to finish
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        
        # Add a slight human-like delay for the JS product cards to render
        await page.wait_for_timeout(5000)
        
        # Scroll down slightly to trigger lazy-loaded images/prices
        await page.mouse.wheel(0, 1000)
        await page.wait_for_timeout(2000)
        
        html = await page.content()
        await page.close()
        return html
    except Exception as e:
        print(f"[Playwright Error] Failed to fetch {url}: {e}")
        return ""


async def compare_prices(*, product_query: str, max_results_per_store: int) -> list[dict[str, str | float]]:
    if not product_query.strip():
        return []

    encoded_query = quote_plus(product_query.strip())
    amazon_url = f"https://www.amazon.in/s?k={encoded_query}"
    flipkart_url = f"https://www.flipkart.com/search?q={encoded_query}"
    myntra_url = f"https://www.myntra.com/{encoded_query}"
    croma_url = f"https://www.croma.com/searchB?q={encoded_query}%3Arelevance"

    # Launch Playwright
   # Launch Playwright with the new V2 Stealth Wrapper!
    async with Stealth().use_async(async_playwright()) as p:
        
        # TURN HEADLESS OFF. 
        browser = await p.chromium.launch(headless=False)
        
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
        )

        # Fetch all pages concurrently
        responses = await asyncio.gather(
            fetch_page_with_playwright(context, amazon_url),
            fetch_page_with_playwright(context, flipkart_url),
            fetch_page_with_playwright(context, myntra_url),
            fetch_page_with_playwright(context, croma_url),
        )

        await browser.close()

    amazon_html, flipkart_html, myntra_html, croma_html = responses

    # ... (Keep the rest of your beautifulsoup parsing logic exactly the same below this line!) ...
    
    amazon_prices = parse_amazon_products(html=amazon_html, max_results=max_results_per_store)
    if not amazon_prices:
        amazon_prices = parse_json_ld_products(
            html=amazon_html, store="amazon", max_results=max_results_per_store
        )

    flipkart_prices = parse_flipkart_products(html=flipkart_html, max_results=max_results_per_store)
    if not flipkart_prices:
        flipkart_prices = parse_json_ld_products(
            html=flipkart_html, store="flipkart", max_results=max_results_per_store
        )

    myntra_prices = parse_myntra_products(html=myntra_html, max_results=max_results_per_store)
    if not myntra_prices:
        myntra_prices = parse_json_ld_products(
            html=myntra_html, store="myntra", max_results=max_results_per_store
        )

    croma_prices = parse_croma_products(html=croma_html, max_results=max_results_per_store)
    if not croma_prices:
        croma_prices = parse_json_ld_products(
            html=croma_html, store="croma", max_results=max_results_per_store
        )

    all_prices = [
        *amazon_prices,
        *flipkart_prices,
        *myntra_prices,
        *croma_prices,
    ]
    
    unique_prices: list[dict[str, str | float]] = []
    seen_pairs: set[tuple[str, str]] = set()
    
    for item in all_prices:
        dedupe_key = (
            str(item.get("store", "")).lower(),
            str(item.get("product_url", "")).strip(),
        )
        if dedupe_key in seen_pairs:
            continue
        seen_pairs.add(dedupe_key)
        unique_prices.append(item)

    return sorted(unique_prices, key=lambda item: float(item["price"]))