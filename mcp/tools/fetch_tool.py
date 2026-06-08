import httpx
from bs4 import BeautifulSoup
import logging
import re

logger = logging.getLogger(__name__)

# We pretend to be a real browser so websites don't block us.
# Many sites return 403 Forbidden to plain Python HTTP clients.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


async def fetch_page(url: str, timeout: int = 15) -> dict:
    """
    Fetch a URL and return clean text.

    Why httpx instead of requests?
    Our whole pipeline is async (FastAPI + LangGraph).
    requests is synchronous - calling it inside async code blocks
    the entire event loop, freezing ALL other concurrent operations.
    httpx is the async-native replacement. Same API, works with await.

    Returns a dict so the URL travels with the content.
    If we returned just a string, we'd lose track of where it came from.
    """
    try:
        async with httpx.AsyncClient(
            headers=HEADERS,
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        return _parse_html(response.text, url)

    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching: {url}")
        return {"url": url, "text": "", "title": "", "success": False,
                "error": f"Timeout after {timeout}s"}

    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP {e.response.status_code} for: {url}")
        return {"url": url, "text": "", "title": "", "success": False,
                "error": f"HTTP {e.response.status_code}"}

    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return {"url": url, "text": "", "title": "", "success": False,
                "error": str(e)}


def _parse_html(html: str, url: str) -> dict:
    """
    Strip HTML tags and return clean readable text.

    Why BeautifulSoup and not regex?
    HTML is not a regular language. Regex breaks on:
    - Nested tags
    - Attributes with > inside them
    - Malformed HTML (which is most of the web)
    BeautifulSoup handles all of this correctly.

    Cleaning steps:
    1. Remove script/style/nav/footer - pure noise for research
    2. Extract text with spaces between tags
    3. Collapse multiple spaces/newlines into single spaces
    4. Hard cap at 50k chars - one huge page shouldn't dominate
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove everything that isn't useful content
    for tag in soup(["script", "style", "nav", "footer",
                     "header", "aside", "form", "noscript"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # separator=" " puts spaces between tags so words don't run together
    text = soup.get_text(separator=" ", strip=True)

    # Collapse all whitespace variations into single spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Hard limit - prevents one massive page flooding our vector store
    MAX_CHARS = 50_000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        logger.info(f"Truncated to {MAX_CHARS} chars: {url}")

    return {
        "url": url,
        "title": title,
        "text": text,
        "success": True,
        "error": "",
    }