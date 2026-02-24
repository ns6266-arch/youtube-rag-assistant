"""Embedding and ChromaDB persistence utilities.

Embeds transcript chunks with OpenAI ``text-embedding-3-small`` and stores
them in a local ChromaDB instance under ``chroma_db/``. Also exposes a
LangChain retriever for use in the RAG pipeline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "youtube_transcripts"


def _get_embeddings() -> OpenAIEmbeddings:
    """Return a configured OpenAIEmbeddings instance."""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env file and restart the app.")

    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


def _get_vectorstore() -> Chroma:
    """Create or load the persistent ChromaDB collection."""

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = _get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def index_documents(documents: List[Document]) -> None:
    """Embed and persist LangChain Documents into ChromaDB.

    Groups documents by ``video_id`` and skips re-embedding for any video that
    already exists in the collection.
    """

    if not documents:
        return

    vectordb = _get_vectorstore()

    video_ids = {
        str(doc.metadata.get("video_id", "")).strip()
        for doc in documents
        if doc.metadata.get("video_id")
    }

    if not video_ids:
        logger.warning("No video_id metadata found on documents; indexing all without deduplication.")
        vectordb.add_documents(documents)
        return

    existing_video_ids = set()
    try:
        for vid in video_ids:
            if not vid:
                continue
            result = vectordb.get(where={"video_id": vid})
            if result and result.get("ids"):
                existing_video_ids.add(vid)
    except Exception:
        logger.warning("Failed to query existing documents in Chroma; indexing all.", exc_info=True)

    videos_to_index = video_ids - existing_video_ids
    if not videos_to_index:
        logger.info("All videos in this batch are already indexed; skipping embedding.")
        return

    docs_to_add = [
        doc for doc in documents if str(doc.metadata.get("video_id", "")).strip() in videos_to_index
    ]

    if not docs_to_add:
        logger.info("No new documents to add after filtering by video_id.")
        return

    logger.info(
        "Indexing %d documents for %d new video(s) into ChromaDB.",
        len(docs_to_add),
        len(videos_to_index),
    )
    vectordb.add_documents(docs_to_add)


def get_retriever(k: int = 4):
    """Return a LangChain retriever (k results) backed by ChromaDB."""

    vectordb = _get_vectorstore()
    return vectordb.as_retriever(search_kwargs={"k": k})

