import asyncio
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

from agents.orchestrator import research_graph

async def main():
    result = await research_graph.ainvoke({
        "topic": "how does RAG work in AI systems",
        "recipient_email": "test@test.com",
        "sub_questions": [],
        "run_id": "",
        "raw_sources": [],
        "errors": [],
        "chunks_embedded": 0,
        "report_markdown": "",
        "email_sent": False,
        "email_error": None,
    })

    print("\n" + "="*50)
    print("SUB-QUESTIONS GENERATED:")
    for q in result["sub_questions"]:
        print(f"  - {q}")

    print(f"\nSOURCES GATHERED: {len(result['raw_sources'])}")
    print(f"CHUNKS EMBEDDED: {result['chunks_embedded']}")
    print(f"ERRORS: {len(result['errors'])}")
    print(f"\nREPORT PREVIEW (first 500 chars):")
    print(result["report_markdown"][:500])

asyncio.run(main())