from app.main import ingest_urls, chat, clear_data

# 1. CLEAR OLD DATA (To get rid of the stale 'Retrieval-augmented generation' page)
clear_data()

TEST_URLS = [
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://www.ibm.com/topics/artificial-intelligence",
    "https://docs.python.org/3/tutorial/introduction.html"
]

print("\n--- INGESTING 3 DOMAINS ---\n")
ingest_urls(TEST_URLS)

print("\n--- TEST 1: Wikipedia Domain ---")
chat("What are LLMs?")

print("\n--- TEST 2: IBM Domain ---")
chat("How does IBM explain AI?")

print("\n--- TEST 3: Python Domain ---")
chat("What can Python do as a calculator?")

print("\n--- TEST 4: Cross-Document Synthesis ---")
chat("Based on all sources, how is the relationship between AI and LLMs described?")