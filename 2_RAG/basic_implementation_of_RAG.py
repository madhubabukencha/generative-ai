import os
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
from tqdm import tqdm
import pickle


def load_documents_from_directory(dir_path: str) -> list[dict]:
    """
    Function which reads the text from documents and
    stores in the list of dictionaries

    :param dir_path: Directory of text documents
    :type  dir_path: str

    :returns : List of directories which holds extracted text
    :rtype: list[dict]
    """
    print("Loading documents...")
    doc_with_content = []
    count = 0
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            with open(file=os.path.join(dir_path, filename),
                      mode="r",
                      encoding="utf-8") as file:
                doc_with_content.append({"id": filename, "text": file.read()})
        count += 1
    print(f"Loading Completed. Loaded {count} documents")
    return doc_with_content


def split_text(text: str,
               chunk_size: int = 1000,
               chunk_overlap: int = 20) -> list[str]:
    """
    Function which creates a chunks of given text

    :param text: Text
    :type  text: str
    :param chunk_size: Large text divided into provide chunk size
    :type  chunk_size: int
    :param chunk_overlap: Which is the text overlap between chunk
                          to next chunk. Helps in keep good context
    :type  chunk_overlap: int

    :returns: List of string of specified chunk_size
    :rtype: list[str]
    """
    chunks_list = []
    start = 0
    while start <= len(text):
        end = start + chunk_size
        chunks_list.append(text[start: end])
        start = end - chunk_overlap
    return chunks_list


def create_chunks(documents: list) -> list[dict]:
    """
    Takes list of dictionaries with extracted text and creates the chunks
    of it

    :param document: List of dictionaries with extracted text
    :param type: list

    :returns: List with chunks of text
    :rtype: list[dict]
    """
    chunked_documents = []
    print("Creating chunks data...")
    for doc in documents:
        chunks = split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}",
                                      "text": chunk})
    print(f"Creating chunks data is completed. \
           Total chunks: {len(chunked_documents)}")
    return chunked_documents


def get_openai_embedding(text: str):
    """
    Function to generate embeddings using OpenAI API by taking
    text as input
    """
    response = client.embeddings.create(input=text,
                                        model="text-embedding-3-small")
    embedding = response.data[0].embedding
    return embedding


if __name__ == "__main__":
    # Setting up OpenAI Environment
    openai_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_key)

    # Setting up OpenAI Embeddings
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small"
                )

    # Setting up Chromadb
    chromadb_client = chromadb.PersistentClient(path="chroma_persistant_path")
    COLLECTION_NAME = "document_qa_collection"
    collection = chromadb_client.get_or_create_collection(
                                name=COLLECTION_NAME,
                                embedding_function=openai_ef)

    # Load documents from the directory
    DIR_PATH = "./news_articles"
    documents = load_documents_from_directory(DIR_PATH)

    # Generate embeddings for the document chunks
    chunked_docs = create_chunks(documents=documents)
    # for doc in tqdm(chunked_docs):
    #     doc["embedding"] = get_openai_embedding(doc["text"])

    with open("gpt-embedding.pkl", "rb") as file:
        chunked_docs = pickle.load(file)

    print("Upsert documents with embeddings into Chroma")
    for doc in tqdm(chunked_docs):
        collection.upsert(
            ids=[doc["id"]],
            documents=[doc["text"]],
            embeddings=[doc["embedding"]]
        )
    print("Upsert documents with embeddings into Chroma")
