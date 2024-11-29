# Advance RAG
- Advance RAG focus on enhancing retrieval quality. It employs following strategies
  - Pre-Retrieval: Improvement ofthe indexing structure and user's query
  - Post-Retrieval: Combine pre-retrieval with the original query
    - Re-Ranking to highlight the most important content
- *Query Expansion*: Generate potential answers to the query using an LLM and to get
  relevant context. <br>
  Query -> LLM -> hallucinated Answer + Query -> Vector DB -> Query Results -> LLM -> Answer