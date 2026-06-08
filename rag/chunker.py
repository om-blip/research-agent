from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from config import config
import logging

logger = logging.getLogger(__name__)


def chunk_text(text: str, source_url: str = "") -> List[Document]:
    """
    Split raw text into chunks ready for embedding.

    Why RecursiveCharacterTextSplitter?
    It tries to split on paragraph breaks first, then line breaks,
    then sentences, then words. This means chunks respect natural
    language boundaries instead of cutting mid-sentence.

    We tried 3 options:
    1. CharacterTextSplitter - splits on one character, cuts mid-sentence. Bad.
    2. TokenTextSplitter - exact token count but still cuts mid-sentence. Bad.
    3. RecursiveCharacterTextSplitter - respects language structure. Good.
    """
    if not text or not text.strip():
        logger.warning(f"Empty text from: {source_url}")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # We wrap raw text in a Document object so metadata travels with it.
    # Why? Because when Claude writes the report, it needs to cite sources.
    # Without metadata, we lose track of which chunk came from which URL.
    raw_doc = Document(
        page_content=text,
        metadata={
            "source": source_url,
            "total_chars": len(text),
        }
    )

    chunks = splitter.split_documents([raw_doc])

    # Tag each chunk with its index so we can debug retrieval later.
    # "Why did we get chunk 14 but not chunk 15?" becomes answerable.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    logger.info(f"Chunked '{source_url}': {len(text)} chars → {len(chunks)} chunks")
    return chunks


def chunk_multiple_sources(sources: List[dict]) -> List[Document]:
    """
    Chunk a list of {text, url} dicts and return all chunks combined.

    Called after ALL agents have gathered their content.
    Returns one flat list - all chunks from all sources merged.
    """
    all_chunks = []

    for source in sources:
        chunks = chunk_text(
            text=source.get("text", ""),
            source_url=source.get("url", "unknown")
        )
        all_chunks.extend(chunks)

    logger.info(f"Total chunks from {len(sources)} sources: {len(all_chunks)}")
    return all_chunks