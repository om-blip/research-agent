import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ResearchRequest, ResearchResponse, ResearchStatus
from agents.orchestrator import research_graph
from agents.state import ResearchState
from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# In-memory job store.
# Why not a database? For a portfolio project this is fine.
# In production you'd use Redis which gives you:
# - Persistence across server restarts
# - TTL-based automatic cleanup
# - Works across multiple server instances
# For now: dict is fast, simple, zero dependencies.
job_store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown. Good place for DB connections etc."""
    logger.info("Research Agent API starting up")
    logger.info(f"Fast model: {config.FAST_MODEL}")
    logger.info(f"Smart model: {config.SMART_MODEL}")
    yield
    logger.info("Research Agent API shutting down")


app = FastAPI(
    title="Research Agent API",
    description=(
        "Agentic RAG pipeline. Submit a topic, get a research report emailed to you. "
        "Powered by Groq + LangGraph + ChromaDB."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS allows your frontend (React, etc.) to call this API.
# allow_origins=["*"] means any domain can call it.
# In production restrict this to your frontend domain only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def run_research_job(run_id: str, topic: str, email: str):
    """
    The actual research work. Runs in the background.
    Updates job_store at each stage so status endpoint reflects progress.
    """
    start_time = time.time()

    try:
        job_store[run_id]["status"] = "running"

        # Build initial state for LangGraph
        initial_state: ResearchState = {
            "topic": topic,
            "recipient_email": email,
            "sub_questions": [],
            "run_id": run_id,
            "raw_sources": [],
            "errors": [],
            "chunks_embedded": 0,
            "report_markdown": "",
            "email_sent": False,
            "email_error": None,
        }

        # Run the full LangGraph pipeline
        # LangSmith automatically traces this if LANGCHAIN_TRACING_V2=true
        final_state = await research_graph.ainvoke(initial_state)

        duration = time.time() - start_time

        job_store[run_id].update({
            "status": "completed",
            "report_markdown": final_state.get("report_markdown", ""),
            "email_sent": final_state.get("email_sent", False),
            "sources_gathered": len(final_state.get("raw_sources", [])),
            "chunks_embedded": final_state.get("chunks_embedded", 0),
            "errors": final_state.get("errors", []),
            "duration_seconds": round(duration, 1),
        })

        logger.info(f"Job {run_id} completed in {duration:.1f}s")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Job {run_id} failed: {e}", exc_info=True)
        job_store[run_id].update({
            "status": "failed",
            "error": str(e),
            "duration_seconds": round(duration, 1),
        })


@app.post("/research", response_model=ResearchResponse, status_code=202)
async def submit_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchResponse:
    """
    Submit a research job.

    Returns HTTP 202 Accepted immediately with a run_id.
    The research runs in the background.
    Poll GET /research/{run_id}/status for updates.

    HTTP 202 vs 200:
    200 OK = "I finished the work"
    202 Accepted = "I received your request and am working on it"
    This distinction matters for API clients that check status codes.
    """
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    # Create job record immediately
    job_store[run_id] = {
        "run_id": run_id,
        "topic": request.topic,
        "email": request.recipient_email,
        "status": "queued",
    }

    # Schedule actual work as background task
    # FastAPI returns the response BEFORE the background task starts
    background_tasks.add_task(
        run_research_job,
        run_id=run_id,
        topic=request.topic,
        email=request.recipient_email,
    )

    logger.info(f"Queued job {run_id}: '{request.topic[:60]}'")

    return ResearchResponse(
        run_id=run_id,
        status="queued",
        message=f"Job queued. Poll /research/{run_id}/status for updates.",
    )


@app.get("/research/{run_id}/status", response_model=ResearchStatus)
async def get_status(run_id: str) -> ResearchStatus:
    """Check the status of a research job."""
    if run_id not in job_store:
        raise HTTPException(
            status_code=404,
            detail=f"Job {run_id} not found"
        )

    job = job_store[run_id]
    report = job.get("report_markdown", "")

    return ResearchStatus(
        run_id=run_id,
        status=job.get("status", "unknown"),
        topic=job.get("topic", ""),
        report_preview=report[:500] + "..." if len(report) > 500 else report,
        email_sent=job.get("email_sent", False),
        sources_gathered=job.get("sources_gathered", 0),
        chunks_embedded=job.get("chunks_embedded", 0),
        errors=job.get("errors", []),
        duration_seconds=job.get("duration_seconds"),
    )


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Used by Docker, Kubernetes, and load balancers to know if the
    service is alive. Should always return fast with no dependencies.
    """
    active = len([j for j in job_store.values() if j.get("status") == "running"])
    return {
        "status": "ok",
        "active_jobs": active,
        "total_jobs": len(job_store),
    }


@app.get("/")
async def root():
    return {
        "service": "Research Agent API",
        "docs": "/docs",
        "health": "/health",
    }