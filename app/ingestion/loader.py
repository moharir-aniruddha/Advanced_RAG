import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from typing import List

class WebDocumentLoader:

    @staticmethod
    def load_url(url: str) -> List[Document]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text(separator=" ", strip=True)
            if not text:
                return []

            return [
                Document(
                    page_content=text,
                    metadata={"source": url} # Metadata is crucial for cross-document Q&A
                )
            ]
        except Exception as e:
            print(f"\nLoader Error for {url}: {e}")
            return []

    @staticmethod
    def load_multiple_urls(urls: List[str]) -> List[Document]:
        """New feature: Processes a list of URLs and aggregates the documents."""
        all_documents = []
        for url in urls:
            print(f"Loading: {url}")
            docs = WebDocumentLoader.load_url(url)
            all_documents.extend(docs)
        return all_documents