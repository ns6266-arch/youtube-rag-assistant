from dotenv import load_dotenv
load_dotenv()

from utils.rag_pipeline import ask

# Test 1: Basic question
print("Test 1: Basic question...")
answer = ask("What does the speaker say about elephants?", session_id="test_session")
print(f"  Answer: {answer}")

# Test 2: Follow-up question (tests memory)
print("\nTest 2: Follow-up question (tests memory)...")
answer2 = ask("What did he say was cool about them?", session_id="test_session")
print(f"  Answer: {answer2}")

# Test 3: Question not in context (tests I don't know behavior)
print("\nTest 3: Out of context question...")
answer3 = ask("What is the capital of France?", session_id="test_session")
print(f"  Answer: {answer3}")

# Test 4: Different session (tests memory isolation)
print("\nTest 4: Different session (should have no memory of previous questions)...")
answer4 = ask("What did I just ask you about?", session_id="different_session")
print(f"  Answer: {answer4}")

# Test 5: Empty question
print("\nTest 5: Empty question...")
answer5 = ask("", session_id="test_session")
print(f"  Answer: {answer5}")