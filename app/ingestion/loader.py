import requests

from bs4 import BeautifulSoup

from langchain_core.documents import Document

class WebDocumentLoader:


    @staticmethod
    def load_url(url: str):

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=15
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text(
                separator=" ",
                strip=True
            )

            if not text:
                raise Exception("No text content extracted")

            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": url
                    }
                )
            ]

        except Exception as e:

            print(f"\nLoader Error: {e}")

            return []

