import httpx
from typing import List, Dict
import logging
import asyncio

logger = logging.getLogger(__name__)


async def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search using DuckDuckGo's HTML endpoint directly.
    No library needed - just a plain HTTP request.
    This avoids all DLL and rate limit issues.
    """
    results = []
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        # DuckDuckGo HTML search - no API key, no library
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}

        # Small delay prevents DDG from rate limiting us
        # when multiple agents search simultaneously
        await asyncio.sleep(1)

        async with httpx.AsyncClient(headers=headers, timeout=10) as client:
            response = await client.post(url, data=params)

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Each result is in a div with class "result__body"
        for result in soup.select(".result__body")[:max_results]:
            title_tag = result.select_one(".result__title")
            link_tag = result.select_one(".result__url")
            snippet_tag = result.select_one(".result__snippet")

            title = title_tag.get_text(strip=True) if title_tag else ""
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            # Get the actual URL from the link
            a_tag = result.select_one("a.result__a")
            if a_tag and a_tag.get("href"):
                results.append({
                    "title": title,
                    "url": a_tag["href"],
                    "snippet": snippet,
                })

        logger.info(f"Search '{query[:50]}': {len(results)} results")

    except Exception as e:
        logger.error(f"Search failed: {e}")

    return results