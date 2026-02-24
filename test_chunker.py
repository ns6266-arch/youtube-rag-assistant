from dotenv import load_dotenv
load_dotenv()

from utils.transcript_fetcher import fetch_transcript
from utils.chunker import build_chunks

# Use cached transcript - no API call needed
print("Loading cached transcript...")
transcript = fetch_transcript("https://www.youtube.com/watch?v=jNQXAC9IVRw")
print(f"  Segments: {len(transcript['segments'])}")

# Test with default settings
print("\nBuilding chunks (default 400 words)...")
docs = build_chunks(transcript)
print(f"  Chunks produced: {len(docs)}")
for i, doc in enumerate(docs):
    words = len(doc.page_content.split())
    print(f"  Chunk {i+1}: {words} words, starts at {doc.metadata['start_time']}s")
    print(f"    Preview: {doc.page_content[:80]}...")
    print(f"    Metadata: {doc.metadata}")

# Test with smaller chunk size to verify overlap works
print("\nBuilding chunks (small 20 word chunks to test overlap)...")
small_docs = build_chunks(transcript, target_words=20, overlap_words=5)
print(f"  Chunks produced: {len(small_docs)}")
for i, doc in enumerate(small_docs):
    words = len(doc.page_content.split())
    print(f"  Chunk {i+1}: {words} words, starts at {doc.metadata['start_time']}s")
    print(f"    Content: {doc.page_content}")

# Test edge case: empty transcript
print("\nTesting empty transcript edge case...")
empty_result = build_chunks({"video_id": "test", "title": "test", "url": "", "segments": []})
print(f"  Empty segments returned: {empty_result} (should be [])")

# Test that metadata is correct
print("\nVerifying metadata structure...")
if docs:
    meta = docs[0].metadata
    assert "video_id" in meta, "Missing video_id"
    assert "video_title" in meta, "Missing video_title"
    assert "start_time" in meta, "Missing start_time"
    assert "url" in meta, "Missing url"
    assert isinstance(meta["start_time"], int), "start_time should be int"
    print("  All metadata keys present and correctly typed")