from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List


class ResearchRequest(BaseModel):
    """
    What the user sends to POST /research.

    Why Pydantic here instead of a plain dict?
    Pydantic validates automatically. If someone sends an invalid email
    or a one-word topic, FastAPI returns a clear error message before
    our code even runs. No manual validation needed.

    EmailStr specifically validates email format.
    "notanemail" gets rejected instantly with a helpful message.
    """
    topic: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The research topic",
        examples=["How does RAG work in AI systems"]
    )
    recipient_email: str = Field(
        ...,
        description="Email to send the report to"
    )

    @field_validator("topic")
    @classmethod
    def topic_must_have_words(cls, v: str) -> str:
        if len(v.split()) < 3:
            raise ValueError("Topic must be at least 3 words")
        return v.strip()


class ResearchResponse(BaseModel):
    """
    What we return immediately after receiving a research request.

    We return INSTANTLY with a run_id.
    The research runs in the background.
    The user polls /research/{run_id}/status to check progress.

    Why not just wait and return the finished report?
    Research takes 60-120 seconds.
    HTTP connections time out after ~30-60 seconds in most systems.
    Background task + polling is the correct pattern for long jobs.
    This is exactly how OpenAI batch API, Stripe, etc. work.
    """
    run_id: str
    status: str
    message: str


class ResearchStatus(BaseModel):
    """What we return when the user polls for status."""
    run_id: str
    status: str           # queued / running / completed / failed
    topic: str
    report_preview: Optional[str] = None   # first 500 chars
    email_sent: bool = False
    sources_gathered: int = 0
    chunks_embedded: int = 0
    errors: List[str] = []
    duration_seconds: Optional[float] = None