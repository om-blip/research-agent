import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from rag.retriever import retrieve_for_prompt
from config import config

logger = logging.getLogger(__name__)

# One LLM client reused across all synthesis calls
# Creating a new client per call wastes time on connection setup
llm = ChatGroq(
    model=config.SMART_MODEL,
    groq_api_key=config.GROQ_API_KEY,
    temperature=0.3,
    max_tokens=4096,
)

SYNTHESIS_SYSTEM_PROMPT = """You are an expert research analyst writing a report.

Rules you must follow:
1. ONLY use information from the provided sources. Never add facts from memory.
2. Cite every claim using source numbers like [1], [2], [3].
3. If sources conflict with each other, mention the disagreement.
4. If sources don't have enough info, say so clearly.
5. Write in clear professional prose with markdown headers.
6. Include specific numbers, dates, and names when the sources mention them.
"""

COMBINE_SYSTEM_PROMPT = """You are a research report editor.
You receive several answers to different sub-questions on the same topic.
Combine them into one coherent well-structured report.

Rules:
1. Keep all facts and citations from the individual answers.
2. Remove duplicate information that appears in multiple answers.
3. Add an Executive Summary at the top with 3-5 bullet points.
4. Use ## for main sections, ### for subsections.
5. End with a ## Sources section listing all URLs that were cited.
6. Do NOT add any new facts. Only reorganise what is already there.
"""


async def synthesise_one_question(question: str, context: str) -> str:
    """Ask Groq to answer one sub-question given retrieved context chunks."""
    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Research question: {question}\n\n"
            f"Sources:\n{context}\n\n"
            f"Write a comprehensive answer using only the sources above."
        ))
    ]
    response = await llm.ainvoke(messages)
    return response.content.strip()


async def synthesis_node(state: dict) -> dict:
    """
    LangGraph node: turn retrieved chunks into a final report.

    Why synthesise per sub-question then combine?
    Option A (ours): retrieve per question → answer each → combine answers
    Option B: dump ALL chunks into one prompt → ask for report

    Option B fails because:
    - Too many chunks = context window overflow
    - LLMs perform worse with very long contexts (lost in the middle problem)
    - No way to track which chunk answered which question

    Option A works because each synthesis call is focused and small.
    """
    logger.info(f"Synthesis starting for run {state['run_id']}")

    # If nothing was gathered, return an error report
    if state.get("chunks_embedded", 0) == 0:
        return {
            "report_markdown": (
                f"# Research Report: {state['topic']}\n\n"
                "**Error**: No sources were successfully gathered.\n\n"
                "Errors:\n" +
                "\n".join(f"- {e}" for e in state.get("errors", []))
            )
        }

    # Step 1: Answer each sub-question using RAG
    answers = []
    for question in state["sub_questions"]:
        # Retrieve the most relevant chunks for THIS specific question
        context = retrieve_for_prompt(question, state["run_id"])

        # Ask Groq to answer with citations
        answer = await synthesise_one_question(question, context)
        answers.append(f"## {question}\n\n{answer}")
        logger.info(f"Answered: '{question[:60]}'")

    # Step 2: Combine all answers into one report
    combined = "\n\n---\n\n".join(answers)

    messages = [
        SystemMessage(content=COMBINE_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Research topic: {state['topic']}\n\n"
            f"Individual answers:\n\n{combined}\n\n"
            f"Combine these into one comprehensive research report."
        ))
    ]
    final = await llm.ainvoke(messages)
    report = final.content.strip()

    # Add a footer with run stats
    report += (
        f"\n\n---\n"
        f"*Sources gathered: {len(state.get('raw_sources', []))} · "
        f"Chunks indexed: {state.get('chunks_embedded', 0)}*"
    )

    logger.info(f"Report complete: {len(report)} chars")
    return {"report_markdown": report}