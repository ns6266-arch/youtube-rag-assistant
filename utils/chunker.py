"""Timestamp-aware transcript chunking.

Converts normalised transcript segments into overlapping chunks while preserving
a reliable `start_time` for each chunk so citations can link back to the correct
moment in the video.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document


def build_chunks(
    transcript: Dict[str, Any],
    target_words: int = 400,
    overlap_words: int = 50,
) -> List[Document]:
    """Chunk a transcript into LangChain Documents with timestamp metadata.

    Iterates over segments and groups them by word count (no character splitting).
    Each chunk has approximately target_words with overlap_words carried into
    the next chunk. The start_time of each chunk is the start time of the first
    segment in that chunk.

    Args:
        transcript: Dict from `utils.transcript_fetcher.fetch_transcript` with
            keys video_id, title, url, segments.
        target_words: Approximate number of words per chunk. Default 400.
        overlap_words: Approximate number of words to overlap with the next chunk.
            Default 50.

    Returns:
        List of LangChain Document objects with:
            - page_content: combined text of segments in the chunk
            - metadata: video_id, video_title, start_time (int seconds), url
    """

    video_id = transcript.get("video_id") or ""
    video_title = transcript.get("title") or "Untitled video"
    url = transcript.get("url") or ""
    segments = transcript.get("segments") or []

    if not segments:
        return []

    documents: List[Document] = []
    i = 0

    while i < len(segments):
        chunk_segments: List[Dict[str, Any]] = []
        word_count = 0

        while i + len(chunk_segments) < len(segments) and word_count < target_words:
            seg = segments[i + len(chunk_segments)]
            text = (seg.get("text") or "").strip()
            if not text:
                i += 1
                continue
            chunk_segments.append(seg)
            word_count += _word_count(text)

        if not chunk_segments:
            break

        page_content = " ".join(
            (s.get("text") or "").strip() for s in chunk_segments
        ).strip()
        start_time = int(float(chunk_segments[0].get("start", 0)))

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "video_id": video_id,
                    "video_title": video_title,
                    "start_time": start_time,
                    "url": url,
                },
            )
        )

        # Advance so next chunk starts with ~overlap_words from the end of this chunk
        overlap_count = 0
        overlap_start = len(chunk_segments) - 1
        while overlap_start >= 0 and overlap_count < overlap_words:
            overlap_count += _word_count(
                (chunk_segments[overlap_start].get("text") or "").strip()
            )
            overlap_start -= 1
        overlap_start += 1
        # Next chunk starts at this segment (global index i + overlap_start)
        i += overlap_start if overlap_start > 0 else max(len(chunk_segments) - 1, 1)

    return documents


def _word_count(text: str) -> int:
    """Return the number of space-separated words in text."""
    return len(text.split()) if text else 0
