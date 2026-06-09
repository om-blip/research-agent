import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ResearchRequest, ResearchResponse, ResearchStatus
from agents.orchestrator import research_graph
from agents.state import ResearchState
from monitoring.metrics import (
    track_request, track_duration, track_chunks,
    track_sources, ACTIVE_RUNS, get_metrics_response
)
from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

job_store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Research Agent API starting up")
    logger.info(f"Fast model: {config.FAST_MODEL}")
    logger.info(f"Smart model: {config.SMART_MODEL}")
    yield
    logger.info("Research Agent API shutting down")


app = FastAPI(
    title="Research Agent API",
    description=(
        "Agentic RAG pipeline. Submit a topic, get a research report. "
        "Powered by Groq + LangGraph + ChromaDB."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def run_research_job(run_id: str, topic: str, email: str):
    """
    The actual research work. Runs in the background.

    Why background task and not just await in the endpoint?
    Research takes 60-120 seconds. HTTP connections time out.
    Background task returns immediately, user polls for status.
    This is exactly how OpenAI batch API and Stripe webhooks work.
    """
    start_time = time.time()
    ACTIVE_RUNS.inc()

    try:
        job_store[run_id]["status"] = "running"

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

        # Track metrics - these show up in /metrics endpoint
        track_request("success")
        track_duration(duration)
        track_chunks(final_state.get("chunks_embedded", 0))
        track_sources(len(final_state.get("raw_sources", [])))

        logger.info(f"Job {run_id} completed in {duration:.1f}s")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Job {run_id} failed: {e}", exc_info=True)

        track_request("error")

        job_store[run_id].update({
            "status": "failed",
            "error": str(e),
            "duration_seconds": round(duration, 1),
        })

    finally:
        # Gauge always decremented whether job succeeded or failed
        ACTIVE_RUNS.dec()


@app.post("/research", response_model=ResearchResponse, status_code=202)
async def submit_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchResponse:
    """
    Submit a research job. Returns immediately with a run_id.
    Poll GET /research/{run_id}/status for progress.

    HTTP 202 = "received and working on it" not "finished".
    """
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    job_store[run_id] = {
        "run_id": run_id,
        "topic": request.topic,
        "email": request.recipient_email,
        "status": "queued",
    }

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
    Docker, Kubernetes, and load balancers ping this to know
    if the service is alive. Must always return fast.
    """
    active = len([j for j in job_store.values() if j.get("status") == "running"])
    return {
        "status": "ok",
        "active_jobs": active,
        "total_jobs": len(job_store),
    }


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    Prometheus scrapes this every 15 seconds and stores the values.
    Grafana draws dashboards on top of the stored data.

    What you see here:
    - research_requests_total: total jobs by status
    - research_duration_seconds: latency histogram
    - research_active_runs: currently running jobs
    - research_chunks_embedded: chunks per run histogram
    - llm_calls_total: Groq API calls by model
    """
    return get_metrics_response()


@app.get("/")
async def root():
    return {
        "service": "Research Agent API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }