from dotenv import load_dotenv
load_dotenv()

from utils.transcript_fetcher import fetch_transcript
from utils.chunker import build_chunks
from utils.embedder import index_documents, get_retriever

# Use cached transcript - no Whisper API call
print("Loading cached transcript...")
transcript = fetch_transcript("https://www.youtube.com/watch?v=jNQXAC9IVRw")
docs = build_chunks(transcript)
print(f"  Chunks to embed: {len(docs)}")

# Test 1: Index documents (hits OpenAI embeddings API)
print("\nIndexing documents into ChromaDB...")
index_documents(docs)
print("  Done - check your /chroma_db folder, it should now have files in it")

# Test 2: Deduplication - run again, should skip
print("\nRe-indexing same video (should skip)...")
index_documents(docs)
print("  Done - if no errors, deduplication worked")

# Test 3: Retrieval
print("\nTesting retrieval...")
retriever = get_retriever(k=2)
results = retriever.invoke("elephants")
print(f"  Results returned: {len(results)}")
for i, doc in enumerate(results):
    print(f"  Result {i+1}: {doc.page_content[:80]}...")
    print(f"  Metadata: {doc.metadata}")

# Test 4: Retrieval with a question
print("\nTesting retrieval with a natural language question...")
results2 = retriever.invoke("what does the speaker say about trunks?")
print(f"  Results returned: {len(results2)}")
for i, doc in enumerate(results2):
    print(f"  Result {i+1}: {doc.page_content[:80]}...")