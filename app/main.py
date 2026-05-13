from app.ingestion.loader import WebDocumentLoader
from app.ingestion.chunker import DocumentChunker
from app.vectordb.vector_store import VectorStoreManager
from app.generation.generator import ResponseGenerator
from app.llm.llm_provider import get_llm


def ingest_url(url):

    docs = WebDocumentLoader.load_url(url)
    if not docs:
        print("No documents loaded.")
        return

    chunks = DocumentChunker.split_documents(docs)
    if not chunks:
        print("No chunks created.")
        return

    VectorStoreManager.create_vector_store(chunks)

    print("Ingestion Completed")


def chat(query):

    llm = get_llm()

    generator = ResponseGenerator(llm)

    result = generator.generate_response(query)

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nSources:\n")

    for source in result["sources"]:
        print(source)