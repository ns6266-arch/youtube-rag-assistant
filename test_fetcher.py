from dotenv import load_dotenv
load_dotenv()

from utils.transcript_fetcher import fetch_transcript, extract_video_id

# Test 1: URL parsing
print("Testing URL extraction...")
test_urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "dQw4w9WgXcQ",
]
for url in test_urls:
    vid_id = extract_video_id(url)
    print(f"  {url[:45]:<45} -> {vid_id}")

# Test 2: Full transcript fetch (hits Whisper API)
# Use a short video to minimise cost during testing
print("\nFetching transcript (this will take 30-60 seconds)...")
result = fetch_transcript("https://www.youtube.com/watch?v=jNQXAC9IVRw")  # 18 second video
print(f"  Title: {result['title']}")
print(f"  Source: {result['source']}")
print(f"  Segments: {len(result['segments'])}")
print(f"  First segment: {result['segments'][0]}")

# Test 3: Cache hit
print("\nTesting cache (should be instant)...")
result2 = fetch_transcript("https://www.youtube.com/watch?v=jNQXAC9IVRw")
print(f"  Title: {result2['title']} - returned instantly from cache")

# Test 4: Bad URL error handling
print("\nTesting error handling...")
try:
    extract_video_id("https://www.google.com")
except ValueError as e:
    print(f"  Caught expected error: {e}")