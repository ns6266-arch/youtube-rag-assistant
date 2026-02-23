"""Transcript fetching utilities (single path: yt-dlp + OpenAI Whisper API).

Downloads audio from YouTube via yt-dlp, transcribes with the OpenAI Whisper API,
and normalises output to timestamped segments:
    [{"text": "...", "start": 12.4, "duration": 3.1}, ...]
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import parse_qs, urlparse

from openai import OpenAI
from yt_dlp import YoutubeDL

logger = logging.getLogger(__name__)

# Repo root: utils/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data"


class TranscriptSegment(TypedDict):
    text: str
    start: float
    duration: float


class TranscriptResult(TypedDict):
    video_id: str
    title: str
    url: str
    segments: List[TranscriptSegment]
    source: str  # always "whisper"


def fetch_transcript(youtube_url: str) -> Dict[str, Any]:
    """Fetch or generate a timestamped transcript for a YouTube URL.

    Checks cache first; if missing, downloads audio via yt-dlp, transcribes
    with the OpenAI Whisper API, then caches and returns the result.

    Args:
        youtube_url: Any common YouTube URL format.

    Returns:
        Dict with keys:
            - video_id: str
            - title: str
            - url: str
            - segments: list[dict[str, Any]]
            - source: str (always "whisper")

    Raises:
        RuntimeError: If transcript fetching fails.
    """

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    video_id = extract_video_id(youtube_url)
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    cached = _load_cached(video_id)
    if cached is not None:
        return cached

    title = _fetch_video_title(canonical_url)

    audio_path: Optional[Path] = None
    try:
        audio_path = _download_audio(canonical_url, video_id)
        segments = _transcribe_with_whisper(audio_path)
        result: TranscriptResult = {
            "video_id": video_id,
            "title": title,
            "url": canonical_url,
            "segments": segments,
            "source": "whisper",
        }
        _save_cached(video_id, result)
        return result
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch transcript. Reason: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to delete temp audio file: %s", audio_path, exc_info=True)


def extract_video_id(youtube_url: str) -> str:
    """Extract a YouTube video ID from common URL formats.

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - Raw video id input (11-char id)
    """

    value = youtube_url.strip()
    if not value:
        raise ValueError("YouTube URL is empty.")

    # Allow raw video IDs.
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    try:
        parsed = urlparse(value)
    except Exception as exc:
        raise ValueError(f"Invalid YouTube URL: {youtube_url}") from exc

    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    # youtu.be/<id>
    if "youtu.be" in host:
        candidate = path.lstrip("/").split("/")[0]
        return _validate_video_id(candidate, youtube_url)

    # youtube.com/watch?v=<id>
    if "youtube.com" in host or "m.youtube.com" in host or "www.youtube.com" in host:
        if path == "/watch":
            q = parse_qs(parsed.query)
            candidate = (q.get("v") or [""])[0]
            return _validate_video_id(candidate, youtube_url)

        # youtube.com/shorts/<id>
        m = re.match(r"^/shorts/([A-Za-z0-9_-]{11})", path)
        if m:
            return m.group(1)

        # youtube.com/embed/<id>
        m = re.match(r"^/embed/([A-Za-z0-9_-]{11})", path)
        if m:
            return m.group(1)

    # Fallback: look for a v= param anywhere
    q = parse_qs(parsed.query)
    if "v" in q and q["v"]:
        return _validate_video_id(q["v"][0], youtube_url)

    raise ValueError(
        "Could not extract a video ID from the provided URL. "
        "Please use a standard YouTube URL (watch?v=..., youtu.be/..., or /shorts/...)."
    )


def _validate_video_id(candidate: str, original: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate or ""):
        return candidate
    raise ValueError(f"Could not extract a valid YouTube video ID from: {original}")


def _cache_path(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}_transcript.json"


def _load_cached(video_id: str) -> Optional[TranscriptResult]:
    path = _cache_path(video_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "segments" not in data:
            return None
        return data  # type: ignore[return-value]
    except Exception:
        logger.warning("Failed to read cache file: %s", path, exc_info=True)
        return None


def _save_cached(video_id: str, payload: TranscriptResult) -> None:
    path = _cache_path(video_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _fetch_video_title(canonical_url: str) -> str:
    """Fetch video metadata (title) via yt-dlp in metadata-only mode."""

    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(canonical_url, download=False)
        title = (info or {}).get("title")
        return title if isinstance(title, str) and title.strip() else "Untitled video"
    except Exception as exc:
        logger.warning("Failed to fetch video metadata: %s", canonical_url, exc_info=True)
        return "Untitled video"


def _download_audio(canonical_url: str, video_id: str) -> Path:
    """Download audio-only media to the data folder using yt-dlp.

    We avoid requiring ffmpeg by downloading the best audio format directly.
    The Whisper API supports many containers including m4a and webm.
    """

    outtmpl = str(DATA_DIR / f"{video_id}.%(ext)s")
    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(canonical_url, download=True)
        filename = ydl.prepare_filename(info)

    path = Path(filename)
    if not path.exists():
        raise RuntimeError("Audio download succeeded but the file was not found on disk.")
    return path


def _transcribe_with_whisper(audio_path: Path) -> List[TranscriptSegment]:
    """Transcribe audio via OpenAI Whisper API and return normalised segments.

    Uses `response_format="verbose_json"` so we can preserve timestamps.
    """

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env file and restart the app.")

    client = OpenAI(api_key=api_key)

    with audio_path.open("rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )

    data: Dict[str, Any]
    if hasattr(transcript, "model_dump"):
        data = transcript.model_dump()  # type: ignore[assignment]
    else:
        data = dict(transcript)  # type: ignore[arg-type]

    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise RuntimeError("Whisper transcription returned no segments.")

    segments: List[TranscriptSegment] = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text", "")).strip()
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        duration = max(0.0, end - start)
        if not text:
            continue
        segments.append({"text": text, "start": start, "duration": duration})

    if not segments:
        raise RuntimeError("Whisper transcription returned segments but none had usable text.")
    return segments
