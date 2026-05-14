from app.ingestion.loader import WebDocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.vectordb.vector_store import VectorStoreManager
from app.generation.generator import ResponseGenerator
from app.memory.conversation_memory import ConversationMemory
from app.llm.llm_provider import get_llm

# Placeholders for global objects
llm = None
memory = None
generator = None


def initialize_system():
    """
    Initializes or re-boots the RAG components.
    This is called on startup and after every database wipe.
    """
    global llm, memory, generator

    print("Initializing RAG System components...")
    llm = get_llm()
    memory = ConversationMemory()

    # We pass the memory object to the generator so it persists across chats
    generator = ResponseGenerator(llm, memory)


# Initial boot-up of the system
initialize_system()


def ingest_urls(urls: list):
    """Processes multiple URLs and updates the system."""
    # Assuming you updated your loader to handle multiple URLs as discussed earlier
    # If not, you can loop through ingest_url(url)
    docs = WebDocumentLoader.load_multiple_urls(urls)

    if not docs:
        print("No documents were loaded.")
        return

    chunks = DocumentChunker.split_documents(docs)

    if not chunks:
        print("No chunks were created.")
        return

    # Save to disk
    VectorStoreManager.create_vector_store(chunks)

    # Re-initialize to ensure the generator's retriever loads the new data
    initialize_system()
    print(f"Ingestion Completed and system re-initialized for {len(urls)} URLs.")


def chat(query, session_id="default_session"):
    """
    Routes the query to the generator.
    Returns a generator object for streaming to the UI.
    """
    return generator.generate_response(query, session_id=session_id)


def clear_data():
    """
    The 'Nuke' function:
    1. Physically deletes FAISS index and pickle files.
    2. Resets the in-memory LLM, Memory, and Generator.
    """
    print("Starting full system wipe...")
    # 1. Clear files on disk
    VectorStoreManager.clear_database()

    # 2. Reset RAM (Memory and Generator)
    initialize_system()
    print("System wipe and re-initialization complete.")