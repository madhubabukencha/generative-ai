# Drawbacks of Naive RAG
- **Limited Contextual Understanding**: Let's say you have documents related to weather
  change which does not talk about human's role in weather change. Now, if you ask any questions about "Human role in weather change" then it only answer from weather related
  information but not human role on weather change. It happening because basic RAG works on
  key word matching or basic semantic similarity which can lead to irrelevant or partially 
  relevant documents.
- **Poor Ranking**: Naive RAG model wouldn't be able to rank documents properly which can
  lead to poor input the generative model.
- Poor integration between retrieval and generative componenets which can leads to poor
  response
- Because of inefficient retrieval machanism and bad indexing naive RAG models can fail
  on handling large-scale datasets.
- Without significant human intervention naive RAG model can fail to handle complex
  queries