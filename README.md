# Research Agent

An end-to-end agentic RAG pipeline that researches any topic, synthesises
a report using Groq (free LLMs), and emails it to you.

## What it does

1. Takes a research topic + your email
2. Breaks it into 4 focused sub-questions (Groq llama-3.1-8b)
3. Runs 4 web agents IN PARALLEL — searches + fetches pages
4. Chunks and embeds all content into ChromaDB (local vector DB)
5. Retrieves relevant chunks using MMR search
6. Synthesises a full report with citations (Groq llama-3.1-70b)
7. Emails the report to you

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Orchestration | LangGraph | Explicit state machine, parallel fan-out |
| LLM | Groq (Llama 3.1) | Free tier, fastest inference available |
| Vector DB | ChromaDB | Local, zero infra, production-swappable |
| Embeddings | all-MiniLM-L6-v2 | Free, local, no API key needed |
| Retrieval | MMR search | Diverse results, not redundant top-k |
| API | FastAPI | Async-native, automatic docs at /docs |
| Evals | Custom LLM-as-judge | Faithfulness 0.88, Relevancy 0.86 |
| Monitoring | Prometheus | Token usage, latency, error rate |
| CI | GitHub Actions | Eval gate blocks regressions |

## Architecture
POST /research
│
▼
LangGraph StateGraph
│
decompose ──► [web_agent × 4 parallel]
│
embed (ChromaDB)
│
synthesize (RAG + Groq)
│
deliver (email)
## Eval results

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Faithfulness | 0.880 | 0.70 | ✅ PASSED |
| Answer Relevancy | 0.860 | 0.65 | ✅ PASSED |

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/research-agent
cd research-agent
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Fill in GROQ_API_KEY (free at console.groq.com)
# Fill in SMTP credentials for email delivery
```

### 3. Run

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Submit a research job

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "how does RAG work in AI", "recipient_email": "you@email.com"}'
```

### 5. Check status

```bash
curl http://localhost:8000/research/{run_id}/status
```

### 6. Run tests

```bash
pytest tests/ -v
```

### 7. Run evals

```bash
python -m evals.eval_suite
```

## Key design decisions

**Why LangGraph over a simple script?**
Parallel fan-out: 4 sub-questions researched simultaneously instead of
sequentially. 10s instead of 40s. Plus checkpointing, error recovery,
and LangSmith tracing built in.

**Why MMR over cosine similarity?**
Plain cosine returns 6 near-identical chunks. MMR returns 6 diverse
chunks covering different angles. Measurably better synthesis quality.

**Why Groq?**
Free tier. 14,400 requests/day. Fastest open-source inference available
(LPU chips). Llama 3.1 70b quality matches GPT-4o on research tasks.

**Why custom evals over RAGAS?**
RAGAS fires parallel LLM calls — breaks on Groq free tier rate limits.
Custom sequential eval with LLM-as-judge gives same signal, works reliably.

## Project structure
research-agent/
├── agents/
│   ├── state.py           # TypedDict state for LangGraph
│   ├── orchestrator.py    # LangGraph graph + all nodes
│   ├── web_agent.py       # search + fetch one sub-question
│   └── synthesis_agent.py # RAG retrieval + Groq synthesis
├── mcp/tools/
│   ├── fetch_tool.py      # httpx + BeautifulSoup scraper
│   ├── search_tool.py     # DuckDuckGo HTML search
│   └── email_tool.py      # SMTP delivery
├── rag/
│   ├── chunker.py         # RecursiveCharacterTextSplitter
│   ├── vector_store.py    # ChromaDB with dedup
│   └── retriever.py       # MMR search + prompt formatting
├── api/
│   ├── main.py            # FastAPI + background tasks
│   └── schemas.py         # Pydantic models
├── evals/
│   ├── eval_suite.py      # LLM-as-judge evaluation
│   └── gold_dataset.json  # 5 gold Q&A pairs
├── monitoring/
│   └── metrics.py         # Prometheus counters
├── tests/
│   └── test_rag_pipeline.py  # 16 unit tests
└── .github/workflows/
└── ci.yml             # test + eval gate on every PR
