import re
from typing import List

class QueryRewriter:

    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query: str, chat_history: str) -> List[str]:
        """
        Decomposes a user query into multiple sub-queries to ensure 
        diverse retrieval across multiple documents.
        """
        prompt = f"""
You are a retrieval query optimizer for a Multi-Document RAG system.

Your task:
1. Analyze the User Query and Conversation History.
2. If the question is complex or asks for a comparison/relationship, break it into 1-3 distinct sub-queries.
3. If the question is simple, provide one optimized version of it.

Rules:
- Resolve all pronouns (it, they, this) using history.
- Ensure each sub-query focuses on a specific entity or concept mentioned in the user's intent.
- Format: Return each sub-query on a new line starting with a dash (-).
- Do NOT include any intro or outro text.

Example for "How does IBM define AI compared to LLMs?":
- IBM official definition of Artificial Intelligence
- Characteristics and definition of Large Language Models (LLM)
- Relationship between AI and LLMs

Conversation History:
{chat_history}

User Query:
{query}

Rewritten Sub-Queries:
"""

        response = self.llm.invoke(prompt)
        raw_output = response.content.strip()

        # Parse the dash-separated list into a Python list
        queries = []
        for line in raw_output.split('\n'):
            clean_line = re.sub(r'^[-\d.]+\s*', '', line).strip()
            if clean_line:
                queries.append(clean_line)

        # Fallback: if parsing fails, return the original query as a list
        if not queries:
            return [query]

        return queries