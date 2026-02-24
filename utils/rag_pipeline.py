"""LangChain LCEL RAG pipeline for multi-turn Q&A over YouTube transcripts.

- Use Chroma retriever (k=4) over embedded transcript chunks
- Use GPT-4o-mini to answer only from retrieved context
- Maintain chat memory via ConversationBufferWindowMemory (k=5)
- Format timestamp citations as clickable markdown links
- Enable LangSmith tracing via environment variables
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from utils import embedder


logger = logging.getLogger(__name__)

# Simple in-process memory store: session_id -> list of (question, answer) tuples
_MEMORY_STORE: Dict[str, List[tuple]] = {}


def ask(question: str, session_id: str) -> str:
    """Ask a question against all indexed videos and return an answer string.

    Args:
        question: User question.
        session_id: Stable ID representing a user/session for chat memory.

    Returns:
        Answer string with markdown timestamp links.
    """

    q = (question or "").strip()
    if not q:
        return "Please enter a question."

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "Missing OPENAI_API_KEY. Add it to your .env file and restart the app."

    chat_history = _get_history(session_id)
    rag_chain = _build_rag_chain()

    try:
        answer = rag_chain.invoke({"question": q, "chat_history": chat_history})
        _save_history(session_id, q, answer)
        return answer
    except Exception as exc:
        logger.exception("RAG pipeline failed.")
        return f"Sorry — I ran into an error while answering that. ({type(exc).__name__}: {exc})"


def _get_history(session_id: str) -> str:
    """Return formatted chat history for a session as a plain string."""

    sid = (session_id or "").strip() or "default"
    exchanges = _MEMORY_STORE.get(sid, [])
    # Keep last 5 exchanges only
    exchanges = exchanges[-5:]
    if not exchanges:
        return ""
    lines: List[str] = []
    for q, a in exchanges:
        lines.append(f"Human: {q}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines)


def _save_history(session_id: str, question: str, answer: str) -> None:
    """Append a question/answer exchange to session memory."""

    sid = (session_id or "").strip() or "default"
    if sid not in _MEMORY_STORE:
        _MEMORY_STORE[sid] = []
    _MEMORY_STORE[sid].append((question, answer))


def _build_rag_chain():
    """Build the LCEL RAG chain (question + chat history → retrieval → answer)."""

    retriever = embedder.get_retriever(k=4)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant answering questions about one or more YouTube videos.\n"
                "You MUST answer only using the provided CONTEXT.\n"
                "If the answer is not in the context, say: \"I don't know based on the provided video transcripts.\"\n\n"
                "CITATIONS:\n"
                "- When you reference information, include timestamp citations as clickable markdown links.\n"
                "- Format links exactly like: [MM:SS](https://www.youtube.com/watch?v=VIDEO_ID&t=SECONDS)\n"
                "- Use the chunk metadata (video_id and start_time seconds) provided in CONTEXT.\n"
                "- Prefer citing the most relevant chunk(s), and you may include multiple citations.\n",
            ),
            (
                "human",
                "CHAT HISTORY:\n{chat_history}\n\n"
                "QUESTION:\n{question}\n\n"
                "CONTEXT:\n{context}\n\n"
                "Answer:",
            ),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def retrieve_and_format(inputs: Dict[str, str]) -> str:
        docs = retriever.invoke(inputs["question"])
        return _format_docs(docs)

    chain = (
        RunnablePassthrough.assign(context=RunnableLambda(retrieve_and_format))
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def _format_docs(docs: List[Document]) -> str:
    """Format retrieved Documents into a context string with citation hints."""

    if not docs:
        return "No relevant transcript chunks were retrieved."

    parts: List[str] = []
    for i, doc in enumerate(docs, start=1):
        md = doc.metadata or {}
        video_id = str(md.get("video_id", "")).strip()
        video_title = str(md.get("video_title", "")).strip()
        start_time = int(md.get("start_time", 0) or 0)
        ts = _format_timestamp(start_time)
        # Provide the link template directly as a hint; the model may reuse it.
        link = f"https://www.youtube.com/watch?v={video_id}&t={start_time}"

        header = f"[Chunk {i}] video_title={video_title!r} video_id={video_id!r} start_time={start_time} ({ts}) link={link}"
        parts.append(header)
        parts.append(doc.page_content.strip())

    return "\n\n".join(parts).strip()


def _format_timestamp(total_seconds: int) -> str:
    """Format seconds as H:MM:SS or MM:SS."""

    s = max(0, int(total_seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"

