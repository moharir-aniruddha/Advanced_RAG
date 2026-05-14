from app.ingestion.loader import WebDocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.vectordb.vector_store import VectorStoreManager
from app.generation.generator import ResponseGenerator
from app.llm.llm_provider import get_llm


def ingest_urls(urls: list):
    """Processes multiple URLs and appends them to the store."""
    docs = WebDocumentLoader.load_multiple_urls(urls)
    if not docs:
        print("No documents loaded.")
        return

    chunks = DocumentChunker.split_documents(docs)
    VectorStoreManager.create_vector_store(chunks)
    print(f"Successfully ingested {len(urls)} URLs.")


def chat(query):
    """Freshly loads the generator to avoid using a stale index."""
    llm = get_llm()
    # Instantiate inside the function so it loads the NEWEST index from disk
    generator = ResponseGenerator(llm)

    result = generator.generate_response(query)

    print("\n\nSources Used:")
    for source in result["sources"]:
        print(f"- {source}")


def clear_data():
    """Exposes the clear function to the test script."""
    VectorStoreManager.clear_database()