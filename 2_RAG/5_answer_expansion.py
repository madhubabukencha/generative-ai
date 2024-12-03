import os
from pypdf import PdfReader
from openai import OpenAI
import numpy as np
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Defining Global variable
openai_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# STEP-1: Loading the data
reader = PdfReader("data_for_adv_rag\\microsoft-annual-report.pdf")
pdf_texts: list = [p.extract_text().strip() for p in reader.pages]
# Filter the empty strings in the list
pdf_texts = [text for text in pdf_texts if text]


# STEP-2: Creating chunks and token per chunk
# It usage in explanation_about_function.txt file
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

# Tokens split
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)


# STEP-3: Creating embedding function object (defulat model: "all-MiniLM-L6-v2)
# Embedding automatically created when you chroma_collection.add
embedding_function = SentenceTransformerEmbeddingFunction()


# STEP-4: Inserting into Chroma
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()


# STEP-5: hallucinated Answer + Query
def augment_query_generated(query, model="gpt-3.5-turbo"):
    prompt = """
    You are a helpful expert financial research assistant. 
    Provide an example answer to the given question, that might be
    found in a document like an annual report.
    """
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content


original_query = "What was the total profit for the year, and how does it \
                  compare to the previous year?"
hypothetical_answer = augment_query_generated(original_query)
joint_query = f"{original_query} {hypothetical_answer}"

# STEP-6: Generating final result 
results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"][0]

