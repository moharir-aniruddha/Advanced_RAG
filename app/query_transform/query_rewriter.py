class QueryRewriter:

    def __init__(self, llm):

        self.llm = llm

    def rewrite(self, query, chat_history):

        prompt = f"""
You are a retrieval query optimizer for a RAG system.

Your task is to rewrite user questions into
clear and retrieval-friendly search queries.

Use conversation history when resolving:
- vague references
- pronouns
- follow-up questions

Rules:
- Preserve original meaning
- Do NOT ask clarification questions
- Convert vague references into explicit topics
- Keep important entities from chat history
- Return ONLY the rewritten query
- The output must look like a search query

Conversation History:
{chat_history}

User Query:
{query}
"""

        response = self.llm.invoke(prompt)

        rewritten_query = response.content.strip()

        return rewritten_query