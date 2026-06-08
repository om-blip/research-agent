import uuid
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from agents.state import ResearchState
from agents.web_agent import web_agent_node
from agents.synthesis_agent import synthesis_node
from rag.chunker import chunk_multiple_sources
from rag.vector_store import upsert_documents
from mcp.tools.email_tool import send_email
from config import config

logger = logging.getLogger(__name__)

# Fast model for simple planning tasks like decomposing a topic
llm = ChatGroq(
    model=config.FAST_MODEL,
    groq_api_key=config.GROQ_API_KEY,
    temperature=0.3,
)


async def decompose_node(state: ResearchState) -> dict:
    """
    Break the research topic into focused sub-questions.

    Why decompose instead of searching the topic directly?
    Searching "quantum computing 2025" returns a mix of everything.
    Searching "what are the latest qubit error rates in 2025" returns
    targeted results. Decomposition = better search queries = better content.

    We use the FAST model here (8b) because decomposing a topic into
    questions is a simple task. No need for the expensive 70b model.
    """
    logger.info(f"Decomposing: {state['topic']}")

    messages = [
        SystemMessage(content=(
            "You are a research planning assistant. "
            "Given a research topic, generate exactly 4 focused sub-questions "
            "that together would produce a comprehensive research report. "
            "Return ONLY the questions, one per line, no numbering or bullets."
        )),
        HumanMessage(content=f"Research topic: {state['topic']}")
    ]

    response = await llm.ainvoke(messages)
    raw = response.content.strip()

    # Parse the questions - filter lines that look like questions
    sub_questions = [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in raw.split("\n")
        if line.strip() and len(line.strip()) > 10
    ][:4]  # max 4 questions

    # Fallback if parsing fails
    if not sub_questions:
        sub_questions = [state['topic']]

    logger.info(f"Generated {len(sub_questions)} sub-questions")

    return {
        "sub_questions": sub_questions,
        "run_id": f"run_{uuid.uuid4().hex[:8]}",
        "raw_sources": [],
        "errors": [],
    }


def spawn_web_agents(state: ResearchState):
    """
    Fan out to one web_agent per sub-question using LangGraph's Send() API.

    This is the parallel execution magic.
    Instead of looping through sub-questions one by one,
    Send() tells LangGraph to run web_agent_node simultaneously
    for each sub-question.

    Each Send() creates an independent task with its own input.
    All tasks run at the same time.
    All results get merged back via operator.add in state.py.
    """
    return [
        Send("web_agent", {"question": q, "run_id": state["run_id"]})
        for q in state["sub_questions"]
    ]


async def embed_node(state: ResearchState) -> dict:
    """
    Chunk and embed all gathered documents into ChromaDB.

    This runs AFTER all parallel web agents complete (fan-in).
    By this point state.raw_sources has all results from all agents merged.
    """
    logger.info(f"Embedding {len(state['raw_sources'])} sources")

    if not state["raw_sources"]:
        logger.warning("No sources to embed")
        return {"chunks_embedded": 0}

    # Filter out empty results
    valid = [s for s in state["raw_sources"] if s.get("text", "").strip()]
    logger.info(f"Valid sources: {len(valid)}/{len(state['raw_sources'])}")

    chunks = chunk_multiple_sources(valid)
    count = upsert_documents(state["run_id"], chunks)

    return {"chunks_embedded": count}


async def deliver_node(state: ResearchState) -> dict:
    """Send the final report via email."""
    subject = f"Research Report: {state['topic'][:60]}"
    result = await send_email(
        to=state["recipient_email"],
        subject=subject,
        body_markdown=state["report_markdown"],
    )
    return {
        "email_sent": result["success"],
        "email_error": result.get("error", ""),
    }


def build_research_graph():
    """
    Assemble all nodes into a compiled LangGraph.

    Graph structure:
    decompose → [web_agent × N in parallel] → embed → synthesize → deliver

    Compiling validates that:
    - All edge targets exist as registered nodes
    - There are no unreachable nodes
    - Entry point is set
    Better to catch errors at startup than mid-run.
    """
    g = StateGraph(ResearchState)

    g.add_node("decompose", decompose_node)
    g.add_node("web_agent", web_agent_node)
    g.add_node("embed", embed_node)
    g.add_node("synthesize", synthesis_node)
    g.add_node("deliver", deliver_node)

    g.set_entry_point("decompose")

    # Conditional edge: decompose → fan out to N web agents
    g.add_conditional_edges("decompose", spawn_web_agents, ["web_agent"])

    # All web agents → embed (fan-in is automatic via operator.add)
    g.add_edge("web_agent", "embed")
    g.add_edge("embed", "synthesize")
    g.add_edge("synthesize", "deliver")
    g.add_edge("deliver", END)

    graph = g.compile()
    logger.info("Graph compiled successfully")
    return graph


# Build once at import time, reuse for every research request
research_graph = build_research_graph()