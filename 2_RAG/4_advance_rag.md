# Advance RAG
- Advance RAG focus on enhancing retrieval quality. It employs following strategies
  - Pre-Retrieval: Improvement of the indexing structure and user's query
  - Post-Retrieval: Combine pre-retrieval with the original query
    - Re-Ranking to highlight the most important content
- **Query Expansion**: Generate potential answers to the query using an LLM and to get
  relevant context. It helps in information Retrieval.<br>
  - **Hallucinated Query**: A Query -> LLM -> hallucinated Answer from LLM + Query -> Vector DB -> Query Results -> LLM -> Answer
  - **Multiple Queries**: A Query -> LLM ->  Multiple similar queries from LLM + Query -> Vector DB -> Multiple Query Results -> LLM -> Answer