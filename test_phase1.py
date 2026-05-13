from app.main import ingest_url, chat

TEST_URL = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

print("\n--- INGESTING DOCUMENT ---\n")

ingest_url(TEST_URL)

print("\n--- ASKING QUESTIONS ---\n")

chat("What is Retrieval-Augmented Generation?")
