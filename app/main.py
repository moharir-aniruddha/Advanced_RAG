from app.ingestion.loader import WebDocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.vectordb.vector_store import VectorStoreManager
from app.generation.generator import ResponseGenerator
from app.memory.conversation_memory import ConversationMemory # Add this
from app.llm.llm_provider import get_llm

# Initialize these ONCE at the module level for persistence
llm = get_llm()
memory = ConversationMemory()
generator = ResponseGenerator(llm, memory) # Pass the persistent memory here

def ingest_urls(urls: list):
    docs = WebDocumentLoader.load_multiple_urls(urls)
    if not docs:
        return
    chunks = DocumentChunker.split_documents(docs)
    VectorStoreManager.create_vector_store(chunks)

def chat(query, session_id="default_session"):
    # Use the persistent generator
    result = generator.generate_response(query, session_id=session_id)
    return result

def clear_data():
    VectorStoreManager.clear_database()