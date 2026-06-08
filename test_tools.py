import asyncio
from mcp.tools.search_tool import search_web
from mcp.tools.fetch_tool import fetch_page

async def main():
    # Test 1: search
    print("Testing search...")
    results = await search_web("LangGraph tutorial 2024", max_results=3)
    print(f"Search works: {len(results)} results")
    for r in results:
        print(f"  - {r['title'][:60]}")

    # Test 2: fetch one of those URLs
    if results:
        print("\nTesting fetch...")
        page = await fetch_page(results[0]['url'])
        print(f"Fetch works: success={page['success']}")
        print(f"Title: {page['title'][:60]}")
        print(f"Text preview: {page['text'][:100]}")

asyncio.run(main())