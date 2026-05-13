from dotenv import load_dotenv
import os

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-small-en"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

VECTOR_DB_PATH = "vectorstore/faiss_index"

TOP_K_RESULTS = 5

BM25_TOP_K = 5
VECTOR_TOP_K = 5

FINAL_TOP_K = 3

GROQ_API_KEY = os.getenv("GROQ_API_KEY")