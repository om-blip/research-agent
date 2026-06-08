"""
config.py

Why does this file exist?
Because without it, you end up with os.getenv("GROQ_API_KEY") scattered
across 10 different files. When a key changes, you hunt through everything.

With config.py: one import, one place to look.

Interview question you'll get: "How do you manage configuration?"
Answer: "Centralised config module loaded once at startup from environment
variables. In production I'd layer in AWS Secrets Manager or GCP Secret
Manager on top of this pattern."
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads your .env file into os.environ


class Config:
    # ── LLM ──────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Why llama-3.1-8b-instant?
    # - 8b = 8 billion parameters. Fast + cheap.
    # - 70b is better quality but 3x slower on Groq.
    # - For research synthesis we use 70b (quality matters).
    # - For decomposition/planning we use 8b (speed matters).
    FAST_MODEL: str = "llama-3.1-8b-instant"      # quick tasks
    SMART_MODEL: str = "llama-3.3-70b-versatile"  # synthesis/reasoning

    # ── Embeddings ───────────────────────────────────────────
    # Groq doesn't do embeddings. We use a free local model via
    # sentence-transformers. No API call, no cost, runs on CPU.
    # "all-MiniLM-L6-v2" is the industry standard lightweight choice:
    # 384 dimensions, fast, good quality for English text.
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── RAG chunking ─────────────────────────────────────────
    # 512 chars per chunk: empirically the best for research content.
    # Too small (128): loses context, chunks become meaningless fragments.
    # Too big (1024): retrieval is noisy, one chunk dominates.
    # 64 overlap: ensures a sentence split across chunk boundary
    #             still appears fully in at least one chunk.
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Retrieval ─────────────────────────────────────────────
    # Fetch 20 candidates from ChromaDB, then MMR re-ranks to 6.
    # MMR = Maximal Marginal Relevance.
    # Without MMR: top 6 might all be the same paragraph reworded.
    # With MMR: top 6 cover 6 DIFFERENT aspects. Much better synthesis.
    RETRIEVAL_K: int = 6
    RETRIEVAL_FETCH_K: int = 20
    MMR_LAMBDA: float = 0.7  # 0=max diversity, 1=max relevance

    # ── Storage ───────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # ── Email ─────────────────────────────────────────────────
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")

    # ── Eval thresholds ───────────────────────────────────────
    # These are the "gates" in CI. If eval scores drop below these,
    # the PR is blocked. Start conservative, tighten over time.
    EVAL_FAITHFULNESS_THRESHOLD: float = 0.70
    EVAL_RELEVANCY_THRESHOLD: float = 0.65


config = Config()