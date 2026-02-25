"""Streamlit UI entry point for the YouTube RAG assistant.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

from typing import Dict, List
from uuid import uuid4
from pathlib import Path
import sys
import time

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chunker, rag_pipeline, transcript_fetcher, embedder


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tuned · Your YouTube RAG Assistant",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _load_css() -> None:
    css_path = Path(__file__).with_name("styles.css")
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ── Global CSS ─────────────────────────────────────────────────────────────────
_load_css()


# ── Session state ───────────────────────────────────────────────────────────────
def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid4())
    if "videos" not in st.session_state:
        st.session_state["videos"]: List[Dict[str, str]] = []
    if "messages" not in st.session_state:
        st.session_state["messages"]: List[Dict[str, str]] = []


# ── Sidebar ─────────────────────────────────────────────────────────────────────
def _sidebar() -> None:
    with st.sidebar:
        st.markdown("""
        <div class="tuned-logo">
            <span class="play-icon">▶</span> Tuned
        </div>
        <div class="tuned-tagline">Your YouTube RAG Assistant</div>
        """, unsafe_allow_html=True)

        st.markdown("LOAD A VIDEO", unsafe_allow_html=False)
        youtube_url = st.text_input(
            label="url",
            label_visibility="collapsed",
            placeholder="https://www.youtube.com/watch?v=...",
            key="url_input",
        )

        if st.button("▶  Analyse Video", key="load_btn"):
            if not youtube_url:
                st.error("Please enter a YouTube URL.")
            else:
                _ingest_video(youtube_url)

        # Loaded videos list
        videos: List[Dict[str, str]] = st.session_state["videos"]
        if videos:
            st.markdown("---")
            st.markdown("LOADED VIDEOS")
            for v in videos:
                st.markdown(f"""
                <div class="video-card">
                    <div class="video-card-icon">▶</div>
                    <div>
                        <div class="video-card-title">{v.get('title', 'Untitled')}</div>
                        <div class="video-card-url">{v.get('url', '')[:45]}...</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.72rem; color:#7a7395; line-height:2; letter-spacing:0.04em;">
            Powered by<br>
            <span style="color:#ff4d4d;">GPT-4o-mini</span> · 
            <span style="color:#ff4d4d;">Whisper</span> · 
            <span style="color:#ff4d4d;">ChromaDB</span> · 
            <span style="color:#ff4d4d;">LangChain</span>
        </div>
        """, unsafe_allow_html=True)


def _ingest_video(youtube_url: str) -> None:
    """Run the full ingestion pipeline with step-by-step progress UI."""

    steps = [
        ("📥", "Transcribing audio via Whisper..."),
        ("✂️", "Chunking transcript with timestamp metadata..."),
        ("🧠", "Embedding chunks into ChromaDB..."),
    ]

    progress_placeholder = st.sidebar.empty()

    def render_steps(current: int, done: bool = False):
        html = '<div style="margin-top:0.5rem;">'
        for i, (icon, label) in enumerate(steps):
            if done and i < current:
                cls = "done"
                prefix = "✓"
            elif i == current and not done:
                cls = "active"
                prefix = "→"
            elif i < current:
                cls = "done"
                prefix = "✓"
            else:
                cls = ""
                prefix = "·"
            html += f'<div class="loading-step {cls}"><div class="step-dot"></div>{prefix} {label}</div>'
        html += "</div>"
        progress_placeholder.markdown(html, unsafe_allow_html=True)

    try:
        render_steps(0)
        transcript = transcript_fetcher.fetch_transcript(youtube_url)

        render_steps(1)
        docs = chunker.build_chunks(transcript)

        render_steps(2)
        embedder.index_documents(docs)

        render_steps(2, done=True)
        time.sleep(0.3)
        progress_placeholder.empty()

        video_info = {
            "video_id": str(transcript.get("video_id", "")),
            "title": str(transcript.get("title", "Untitled video")),
            "url": str(transcript.get("url", "")),
            "source": str(transcript.get("source", "whisper")),
        }

        videos: List[Dict[str, str]] = st.session_state["videos"]
        if not any(v.get("video_id") == video_info["video_id"] for v in videos):
            videos.append(video_info)

        st.sidebar.success(f"✓ Ready — \"{video_info['title']}\"")

    except Exception as exc:
        progress_placeholder.empty()
        st.sidebar.error(f"Failed: {exc}")


# ── Main area ───────────────────────────────────────────────────────────────────
def _main_area() -> None:
    videos: List[Dict[str, str]] = st.session_state["videos"]
    messages: List[Dict[str, str]] = st.session_state["messages"]

    # Hero always visible
    st.markdown("""
    <div class="hero-title">Chat with your<br><em>YouTube video.</em></div>
    <div class="hero-sub">TUNED · GROUNDED IN YOUR VIDEO · TIMESTAMP CITATIONS INCLUDED</div>
    """, unsafe_allow_html=True)

    if not videos and not messages:
        # Empty state shown below hero
        st.markdown("""
        <div class="empty-state">
            <span class="empty-state-icon">▶</span>
            <div class="empty-state-text">No video loaded yet</div>
            <div class="empty-state-sub">Paste a YouTube URL in the sidebar to get started</div>
        </div>
        """, unsafe_allow_html=True)
        st.chat_input("Load a video first to start chatting...", disabled=True)
        return

    # Summarize button
    if videos:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.container():
                st.markdown('<div class="summarize-btn">', unsafe_allow_html=True)
                if st.button("✦  Summarize all loaded videos", key="summarize_btn"):
                    _run_summary()
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Chat history
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if videos:
        user_input = st.chat_input("Ask a question about the video...")
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            session_id = st.session_state["session_id"]
            with st.chat_message("assistant"):
                with st.spinner(""):
                    answer = rag_pipeline.ask(user_input, session_id=session_id)
                    st.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.rerun()
    else:
        st.chat_input("Load a video first to start chatting...", disabled=True)


def _run_summary() -> None:
    session_id = st.session_state["session_id"]
    prompt = (
        "Provide a clear, structured summary of the key ideas from all loaded videos. "
        "Highlight major sections, important arguments, and any concrete examples. "
        "Include timestamp links where helpful."
    )
    with st.spinner("Generating summary..."):
        answer = rag_pipeline.ask(prompt, session_id=session_id)
    st.session_state["messages"].append({"role": "user", "content": "[Auto] Summarize loaded videos"})
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.rerun()


# ── Entry point ─────────────────────────────────────────────────────────────────
def main() -> None:
    _init_session_state()
    _sidebar()
    _main_area()


if __name__ == "__main__":
    main()