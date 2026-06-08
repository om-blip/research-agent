from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from chromadb.utils import embedding_functions
from typing import List
import chromadb
import logging
from config import config

logger = logging.getLogger(__name__)

_stores: dict = {}


class ChromaDBEmbeddings(Embeddings):
    """
    Wrapper that makes ChromaDB's built-in embedding function
    work with LangChain's interface.

    Why this wrapper?
    LangChain's Chroma class expects an object with .embed_documents()
    and .embed_query() methods. ChromaDB's built-in embedder has a
    different interface. This wrapper bridges the two.

    ChromaDB's default embedder uses all-MiniLM-L6-v2 under the hood
    via onnxruntime — pure Python, no sklearn, no DLL issues.
    Same model we originally wanted, just a different loading path.
    """
    def __init__(self):
        self.ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.ef(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.ef([text])[0]


def get_embedding_function():
    return ChromaDBEmbeddings()


def get_or_create_store(collection_name: str) -> Chroma:
    """
    Get or create a ChromaDB collection by name.

    One collection per research run so topics don't mix.
    Cached in _stores dict so we don't reopen the file lock repeatedly.
    """
    if collection_name not in _stores:
        logger.info(f"Creating ChromaDB collection: {collection_name}")
        _stores[collection_name] = Chroma(
            collection_name=collection_name,
            embedding_function=get_embedding_function(),
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
    return _stores[collection_name]


def upsert_documents(collection_name: str, documents: List[Document]) -> int:
    """
    Add documents to the vector store.

    Uses stable IDs built from source URL + chunk index.
    This means re-running the same research job won't create duplicates.
    ChromaDB sees the same ID and updates instead of inserting.
    """
    store = get_or_create_store(collection_name)

    if not documents:
        logger.warning("upsert_documents called with empty list")
        return 0

    # Build IDs first
    ids = [
        f"{doc.metadata.get('source', 'unknown')}__chunk_{doc.metadata.get('chunk_index', i)}"
        for i, doc in enumerate(documents)
    ]

    # Deduplicate - same URL scraped twice by different agents
    # produces identical chunk IDs. Keep only the first occurrence.
    seen = set()
    unique_docs = []
    unique_ids = []
    for doc, id_ in zip(documents, ids):
        if id_ not in seen:
            seen.add(id_)
            unique_docs.append(doc)
            unique_ids.append(id_)

    duplicates_removed = len(documents) - len(unique_docs)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate chunks")

    store.add_documents(documents=unique_docs, ids=unique_ids)
    logger.info(f"Upserted {len(documents)} chunks into '{collection_name}'")
    return len(documents)


def delete_collection(collection_name: str):
    """Delete a collection after the job is done to free disk space."""
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        client.delete_collection(collection_name)
        _stores.pop(collection_name, None)
        logger.info(f"Deleted collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to delete collection {collection_name}: {e}")