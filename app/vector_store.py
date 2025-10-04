import os
from sentence_transformers import SentenceTransformer
import chromadb


# ----------------------------
# Initialize embedding model
# ----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------
# Initialize Chroma vector DB
# ----------------------------
chroma_client = chromadb.PersistentClient(path="app/chroma_db")


# ----------------------------
# Helper: Get or create user-specific collection
# ----------------------------
def get_user_collection(user_id: str):
    """
    Returns a Chroma collection specific to the given user.
    Each user has a separate collection to isolate their documents.
    """
    collection_name = f"documents_{user_id}"
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for search
    )


# ----------------------------
# Embed text
# ----------------------------
def embed_text(text: str):
    """
    Generate embeddings for the input text using local model
    """
    return model.encode(text).tolist()


# ----------------------------
# Add document for a user
# ----------------------------
def add_document(user_id: str, doc_id: str, text_chunks: list[str], source: str):
    """
    Add document chunks to the user's Chroma collection with embeddings.
    """
    collection = get_user_collection(user_id)
    embeddings = [embed_text(chunk) for chunk in text_chunks]

    collection.add(
        ids=[f"{doc_id}_{i}" for i in range(len(text_chunks))],
        documents=text_chunks,
        embeddings=embeddings,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(text_chunks))]
    )


# ----------------------------
# Search documents for a user
# ----------------------------
def search(user_id: str, query: str, top_k: int = 3):
    """
    Search in the user's Chroma collection and return top_k most relevant chunks along with metadata and distances.
    """
    collection = get_user_collection(user_id)
    q_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results





# # ----------------------------
# # Example usage
# # ----------------------------
# if __name__ == "__main__":
#     user_id = "user123"
#     doc_id = "doc1"
#     text_chunks = [
#         "This is the first chunk of text.",
#         "This is the second chunk of text."
#     ]
#     source = "example.pdf"

#     # Add document for user
#     add_document(user_id, doc_id, text_chunks, source)

#     # Search for user
#     query = "first chunk"
#     search_results = search(user_id, query, top_k=2)
#     print(search_results)




# import os
# from sentence_transformers import SentenceTransformer  # For Solution 2
# # OR from openai import OpenAI  # For Solution 1
# import chromadb

# # Initialize embedding model (Solution 2)
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Initialize Chroma vector DB with cosine similarity
# chroma_client = chromadb.PersistentClient(path="app/chroma_db")

# collection = chroma_client.get_or_create_collection(
#     name="documents",
#     metadata={"hnsw:space": "cosine"}  # Use cosine similarity for better text matching :cite[9]:cite[10]
# )

# def embed_text(text: str):
#     """
#     Generate embeddings for the input text using local model
#     """
#     return model.encode(text).tolist()

# def add_document(doc_id: str, text_chunks: list[str], source: str):
#     """
#     Add document chunks to the Chroma vector DB with embeddings.
#     """
#     embeddings = [embed_text(chunk) for chunk in text_chunks]
#     collection.add(
#         ids=[f"{doc_id}_{i}" for i in range(len(text_chunks))],
#         documents=text_chunks,
#         embeddings=embeddings,
#         metadatas=[{"source": source, "chunk_index": i} for i in range(len(text_chunks))]  # Enhanced metadata
#     )

# def search(query: str, top_k: int = 3):
#     """
#     Search in the Chroma DB and return top_k most relevant chunks along with source.
#     """
#     q_embedding = embed_text(query)
#     results = collection.query(
#         query_embeddings=[q_embedding], 
#         n_results=top_k,
#         include=["documents", "metadatas", "distances"]  # Return distances for relevance scoring
#     )
#     return results