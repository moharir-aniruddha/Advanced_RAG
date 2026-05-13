class QueryRewriter:

    def __init__(self, llm):

        self.llm = llm

    def rewrite(self, query):
        prompt = f"""
        You are a retrieval query optimizer for a RAG system.

        Your job is to rewrite user questions into
        clear, specific, retrieval-friendly search queries.

        Rules:
        - Preserve the original meaning
        - Do NOT ask follow-up questions
        - Do NOT ask for clarification
        - Convert vague references into explicit topics
        - Expand short queries into searchable queries
        - Keep the rewritten query concise
        - Return ONLY the rewritten query
        - The rewritten query must look like a search query,
          NOT like a chatbot response

        User Query:
        {query}
        """

        response = self.llm.invoke(prompt)

        rewritten_query = response.content.strip()

        return rewritten_query