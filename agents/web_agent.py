import asyncio
import logging
from mcp.tools.fetch_tool import fetch_page
from mcp.tools.search_tool import search_web

logger = logging.getLogger(__name__)

# How many pages to fetch per sub-question
# 3 is the sweet spot - enough diversity, not too slow
PAGES_PER_QUESTION = 3


async def web_agent_node(state: dict) -> dict:
    """
    LangGraph node: research one sub-question.

    This node receives ONE sub-question via LangGraph's Send() API.
    Multiple copies of this node run in PARALLEL, one per sub-question.

    Why parallel?
    If we have 4 sub-questions and each takes 10 seconds:
    - Sequential: 40 seconds total
    - Parallel:   10 seconds total

    Each copy returns {"raw_sources": [...], "errors": [...]}
    LangGraph merges all results using operator.add from state.py
    so everything ends up in one combined list.
    """
    question = state["question"]
    logger.info(f"Web agent researching: '{question[:80]}'")

    # Step 1: Search for URLs about this question
    try:
        search_results = await search_web(question, max_results=PAGES_PER_QUESTION)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"raw_sources": [], "errors": [f"Search failed: {e}"]}

    if not search_results:
        return {"raw_sources": [], "errors": [f"No results for: {question}"]}

    # Step 2: Fetch all pages CONCURRENTLY
    # asyncio.gather runs all fetches at the same time instead of one by one
    # 3 pages × 2 seconds each = 2 seconds total instead of 6 seconds
    urls = [r["url"] for r in search_results if r.get("url")]
    fetch_tasks = [fetch_page(url) for url in urls]

    try:
        # return_exceptions=True means one failed fetch doesn't crash all others
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    except Exception as e:
        return {"raw_sources": [], "errors": [str(e)]}

    # Step 3: Collect successful results, skip failed ones
    raw_sources = []
    errors = []

    for url, result in zip(urls, fetch_results):
        # asyncio.gather returns the exception as a value if return_exceptions=True
        if isinstance(result, Exception):
            errors.append(f"Exception fetching {url}: {result}")
            continue

        if not result.get("success"):
            errors.append(f"Failed {url}: {result.get('error', 'unknown')}")
            continue

        if not result.get("text", "").strip():
            errors.append(f"Empty content: {url}")
            continue

        raw_sources.append({
            "url": url,
            "title": result.get("title", ""),
            "text": result["text"],
            "sub_question": question,  # tag which question this answers
        })

    logger.info(
        f"Done '{question[:50]}': "
        f"{len(raw_sources)} sources, {len(errors)} errors"
    )

    return {"raw_sources": raw_sources, "errors": errors}