"""
evals/eval_suite.py — Simple RAG quality evaluation using Groq as judge.

Why we replaced RAGAS with a custom eval:
RAGAS fires 10+ parallel LLM calls simultaneously.
Groq free tier allows ~30 requests/minute.
Result: massive rate limiting, timeouts, broken evals.

Our approach: run evaluations one at a time with a small delay.
Slower but reliable on free tier.

We still measure the same two things RAGAS measures:
1. Faithfulness - did the answer stay grounded in the retrieved context?
2. Answer relevancy - did the answer actually address the question?

We use Groq itself as the judge (LLM-as-judge pattern).
This is standard practice - GPT-4, Claude, or any strong LLM can
evaluate whether an answer is faithful to its sources.
"""

import json
import logging
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_eval_collection():
    """Build a small ChromaDB collection for evaluation."""
    from rag.chunker import chunk_text
    from rag.vector_store import upsert_documents

    collection = "eval_collection"

    knowledge_base = [
        {
            "url": "https://example.com/rag-intro",
            "text": """
            Retrieval Augmented Generation (RAG) is a technique that combines
            information retrieval with large language model generation.
            RAG retrieves relevant documents from a knowledge base and uses them
            as context when generating answers. This reduces hallucinations because
            the model is grounded in retrieved facts rather than relying purely
            on training data. RAG systems have three main components: an indexing
            pipeline that stores documents as vectors, a retrieval component that
            finds relevant chunks, and a generation component that synthesises answers.
            """
        },
        {
            "url": "https://example.com/vector-search",
            "text": """
            Vector similarity search works by converting text into high-dimensional
            numerical embeddings using embedding models. Each piece of text becomes
            a point in a high-dimensional space. Similar texts end up close together.
            At query time the query is also embedded and the database finds the
            nearest vectors using distance metrics like cosine similarity or dot product.
            This enables semantic search that understands meaning not just keywords.
            Popular vector databases include ChromaDB, Pinecone, Qdrant, and Weaviate.
            """
        },
        {
            "url": "https://example.com/chunking",
            "text": """
            Chunking strategies determine how documents are split before embedding.
            Fixed-size splitting cuts at exact character counts regardless of meaning.
            Recursive character splitting tries paragraph breaks first then sentences
            then words, respecting natural language boundaries. Semantic splitting
            uses embeddings to detect topic changes. Chunk size matters: too small
            loses context, too large makes retrieval noisy. 512 characters with
            64 character overlap is a common starting point for research content.
            """
        },
        {
            "url": "https://example.com/mmr",
            "text": """
            Maximal Marginal Relevance (MMR) is a retrieval strategy that balances
            relevance and diversity. Plain similarity search returns the top-k most
            similar chunks which are often near-duplicates of each other.
            MMR fetches a larger candidate set then greedily selects results that
            are both relevant to the query AND different from already selected results.
            The lambda parameter controls the tradeoff: 0 means maximum diversity,
            1 means maximum relevance. A value of 0.7 works well for most cases.
            """
        },
        {
            "url": "https://example.com/vector-dbs",
            "text": """
            Vector databases are purpose-built for storing and searching
            high-dimensional embedding vectors. Unlike traditional databases that
            search by exact match or range queries, vector databases use approximate
            nearest neighbour algorithms like HNSW and IVF to find similar vectors
            efficiently at scale. They power RAG systems, semantic search,
            recommendation engines, and image similarity search. ChromaDB is popular
            for local development. Pinecone and Qdrant are managed cloud options
            that scale to billions of vectors.
            """
        },
    ]

    all_chunks = []
    for item in knowledge_base:
        chunks = chunk_text(item["text"].strip(), item["url"])
        all_chunks.extend(chunks)

    count = upsert_documents(collection, all_chunks)
    logger.info(f"Built eval collection: {count} chunks")
    return collection


def judge_faithfulness(question: str, answer: str, context: str, llm) -> float:
    """
    Ask Groq: is every claim in this answer supported by the context?

    Returns a score from 0.0 to 1.0.
    1.0 = fully grounded in context
    0.0 = completely hallucinated

    This is the LLM-as-judge pattern. We ask the LLM to score itself.
    Works surprisingly well and is the basis of how RAGAS works internally.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = f"""You are evaluating a RAG system. 
    
Question: {question}

Retrieved context:
{context}

Answer given:
{answer}

Task: Score how faithful the answer is to the context.
A faithful answer only contains claims that are supported by the context.
An unfaithful answer contains claims not found in the context (hallucinations).

Respond with ONLY a number between 0.0 and 1.0.
1.0 = every claim is in the context
0.5 = some claims are in context, some are not  
0.0 = answer ignores context completely

Your score:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        score_text = response.content.strip()
        # Extract the first number we find
        import re
        numbers = re.findall(r'\d+\.?\d*', score_text)
        if numbers:
            score = float(numbers[0])
            return min(1.0, max(0.0, score))
    except Exception as e:
        logger.warning(f"Faithfulness judge failed: {e}")

    return 0.5  # default if judge fails


def judge_relevancy(question: str, answer: str, llm) -> float:
    """
    Ask Groq: does this answer actually address the question?

    Returns a score from 0.0 to 1.0.
    1.0 = directly answers the question
    0.0 = completely off-topic
    """
    from langchain_core.messages import HumanMessage

    prompt = f"""You are evaluating a question-answering system.

Question: {question}

Answer: {answer}

Task: Score how relevant the answer is to the question.
Does the answer actually address what was asked?

Respond with ONLY a number between 0.0 and 1.0.
1.0 = directly and completely answers the question
0.5 = partially answers the question
0.0 = does not answer the question at all

Your score:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        score_text = response.content.strip()
        import re
        numbers = re.findall(r'\d+\.?\d*', score_text)
        if numbers:
            score = float(numbers[0])
            return min(1.0, max(0.0, score))
    except Exception as e:
        logger.warning(f"Relevancy judge failed: {e}")

    return 0.5


def run_evals(gold_path: str = "evals/gold_dataset.json") -> dict:
    """
    Run evaluation on all gold examples one at a time.
    Sequential execution avoids rate limit issues on Groq free tier.
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    from rag.retriever import retrieve_for_prompt
    from config import config

    # Build knowledge base
    collection = build_eval_collection()

    # Load gold questions
    with open(gold_path) as f:
        gold = json.load(f)

    logger.info(f"Running evals on {len(gold)} examples")

    # One LLM for answering, one for judging
    # Both use the fast model to stay within rate limits
    answer_llm = ChatGroq(
        model=config.FAST_MODEL,
        groq_api_key=config.GROQ_API_KEY,
        temperature=0,
        max_tokens=512,
    )
    judge_llm = ChatGroq(
        model=config.FAST_MODEL,
        groq_api_key=config.GROQ_API_KEY,
        temperature=0,
        max_tokens=10,   # judge only needs to output a number
    )

    results = []

    for i, item in enumerate(gold):
        question = item["question"]
        logger.info(f"[{i+1}/{len(gold)}] Evaluating: '{question}'")

        # Step 1: Retrieve context
        context = retrieve_for_prompt(question, collection)

        # Step 2: Generate answer
        try:
            response = answer_llm.invoke([
                SystemMessage(content=(
                    "Answer using only the provided sources. "
                    "Be concise. 2-3 sentences maximum."
                )),
                HumanMessage(content=f"Question: {question}\n\nSources:\n{context}")
            ])
            answer = response.content.strip()
        except Exception as e:
            logger.warning(f"Answer generation failed: {e}")
            answer = "Could not generate answer"

        # Small delay between answer and judge calls
        time.sleep(2)

        # Step 3: Judge faithfulness
        faith_score = judge_faithfulness(question, answer, context, judge_llm)
        time.sleep(2)

        # Step 4: Judge relevancy
        rel_score = judge_relevancy(question, answer, judge_llm)
        time.sleep(2)

        results.append({
            "question": question,
            "answer": answer[:200],
            "faithfulness": faith_score,
            "answer_relevancy": rel_score,
        })

        logger.info(
            f"  faithfulness={faith_score:.2f} "
            f"relevancy={rel_score:.2f}"
        )

        # Delay between examples to avoid rate limits
        if i < len(gold) - 1:
            logger.info("  Waiting 5s before next example...")
            time.sleep(5)

    # Calculate averages
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["answer_relevancy"] for r in results) / len(results)

    scores = {
        "faithfulness": round(avg_faith, 3),
        "answer_relevancy": round(avg_rel, 3),
        "num_examples": len(results),
        "passed": (
            avg_faith >= config.EVAL_FAITHFULNESS_THRESHOLD and
            avg_rel >= config.EVAL_RELEVANCY_THRESHOLD
        )
    }

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump({**scores, "details": results}, f, indent=2)

    # Print final report
    print("\n" + "="*40)
    print("EVAL RESULTS")
    print("="*40)
    for r in results:
        print(f"\nQ: {r['question'][:60]}")
        print(f"   Faithfulness: {r['faithfulness']:.2f}  Relevancy: {r['answer_relevancy']:.2f}")
    print("\n" + "-"*40)
    print(f"AVG Faithfulness:     {scores['faithfulness']:.3f}  (threshold: {config.EVAL_FAITHFULNESS_THRESHOLD})")
    print(f"AVG Answer Relevancy: {scores['answer_relevancy']:.3f}  (threshold: {config.EVAL_RELEVANCY_THRESHOLD})")
    print(f"Examples evaluated:   {scores['num_examples']}")
    print(f"Status: {'PASSED' if scores['passed'] else 'FAILED'}")
    print("="*40)

    # Exit code 1 if thresholds not met - this is the CI gate
    if not scores["passed"]:
        sys.exit(1)

    return scores


if __name__ == "__main__":
    run_evals()