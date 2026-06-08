from langchain_core.documents import Document
from typing import List, Tuple
from rag.vector_store import get_or_create_store
from config import config
import logging

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    collection_name: str,
    k: int = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieve relevant chunks using MMR search.

    MMR = Maximal Marginal Relevance.
    Fetches 20 candidates, then picks 6 that are BOTH relevant AND diverse.
    Prevents getting 6 versions of the same paragraph back.

    Why MMR was failing before:
    The old code called max_marginal_relevance_search_with_score_by_vector
    which requires a raw embedding vector as input. ChromaDB's wrapper
    doesn't always expose that method cleanly.

    Fix: use the LangChain retriever interface directly which handles
    the MMR logic internally without needing the raw vector.
    """
    k = k or config.RETRIEVAL_K
    store = get_or_create_store(collection_name)

    try:
        # This is the correct way to call MMR through LangChain's interface
        docs = store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=config.RETRIEVAL_FETCH_K,
            lambda_mult=config.MMR_LAMBDA,
        )
        # max_marginal_relevance_search doesn't return scores
        # so we pair each doc with a dummy score of 1.0
        results = [(doc, 1.0) for doc in docs]
        logger.info(f"MMR retrieved {len(results)} chunks for: '{query[:60]}'")
        return results

    except Exception as e:
        logger.warning(f"MMR failed ({e}), falling back to similarity search")
        try:
            results = store.similarity_search_with_relevance_scores(query, k=k)
            logger.info(f"Similarity search retrieved {len(results)} chunks")
            return results
        except Exception as e2:
            logger.error(f"Both retrieval methods failed: {e2}")
            return []


def retrieve_for_prompt(query: str, collection_name: str) -> str:
    """
    Retrieve chunks and format as numbered context block for the LLM.

    Format:
        [1] Source: https://example.com
        Relevance: 0.85
        Content of chunk...

        [2] Source: https://other.com
        ...

    Why number the sources?
    So Groq can write "According to [1]..." and RAGAS can verify
    that claim actually exists in source 1 during evaluation.
    That's how faithfulness scoring works.
    """
    results = retrieve(query, collection_name)

    if not results:
        return "No relevant context found."

    parts = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(
            f"[{i}] Source: {source}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)