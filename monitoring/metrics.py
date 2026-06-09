"""
monitoring/metrics.py — Prometheus metrics for the research agent.

WHY PROMETHEUS?
Prometheus is the industry standard for service monitoring.
It works by SCRAPING your /metrics endpoint every 15 seconds
and storing the values as time series data.
Grafana then draws dashboards on top of that data.

The pattern at FAANG:
  Your service exposes /metrics
  Prometheus scrapes it every 15s
  Grafana shows graphs
  PagerDuty alerts you when graphs look wrong

WHAT WE TRACK AND WHY:

1. REQUEST_COUNTER
   Total research requests split by status (success/error).
   Error rate = errors / total. Alert if > 5%.
   
2. RESEARCH_DURATION
   How long each research job takes end to end.
   P95 latency = 95% of jobs finish within X seconds.
   If P95 suddenly spikes, something is slow.
   
3. ACTIVE_RUNS
   How many jobs are currently running.
   If this grows without stopping = something is hanging.
   
4. CHUNKS_EMBEDDED
   How many chunks we store per run.
   Sudden drop = scraping is broken.
   Sudden spike = we're fetching too much content.

5. LLM_CALLS
   Total Groq API calls by model.
   Maps directly to cost and rate limit usage.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response


# Counter: only goes up, never down.
# Use for: requests, errors, events.
# labelnames lets us split by dimension e.g. status="success" vs "error"
REQUEST_COUNTER = Counter(
    "research_requests_total",
    "Total research requests",
    labelnames=["status"],
)

# Histogram: tracks distribution of values.
# Use for: latencies, sizes, durations.
# buckets define the boundary points for each histogram bar.
# We expect most jobs between 20-120 seconds.
RESEARCH_DURATION = Histogram(
    "research_duration_seconds",
    "End-to-end research job duration",
    buckets=[10, 20, 30, 60, 90, 120, 180, 300],
)

# Gauge: goes up AND down.
# Use for: current state (active connections, queue size, memory).
ACTIVE_RUNS = Gauge(
    "research_active_runs",
    "Number of research jobs currently running",
)

CHUNKS_EMBEDDED = Histogram(
    "research_chunks_embedded",
    "Number of chunks embedded per research run",
    buckets=[50, 100, 200, 300, 500, 750, 1000],
)

LLM_CALLS = Counter(
    "llm_calls_total",
    "Total LLM API calls",
    labelnames=["model"],
)

SOURCES_GATHERED = Histogram(
    "research_sources_gathered",
    "Number of web sources gathered per run",
    buckets=[1, 2, 3, 5, 7, 10, 15],
)


def track_request(status: str):
    """Call this when a research job finishes."""
    REQUEST_COUNTER.labels(status=status).inc()


def track_duration(seconds: float):
    """Call this with how long the job took."""
    RESEARCH_DURATION.observe(seconds)


def track_chunks(count: int):
    """Call this with how many chunks were embedded."""
    CHUNKS_EMBEDDED.observe(count)


def track_sources(count: int):
    """Call this with how many sources were gathered."""
    SOURCES_GATHERED.observe(count)


def track_llm_call(model: str):
    """Call this every time we make a Groq API call."""
    LLM_CALLS.labels(model=model).inc()


def get_metrics_response() -> Response:
    """
    Return Prometheus metrics in text exposition format.
    This is what Prometheus scrapes at /metrics every 15 seconds.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )