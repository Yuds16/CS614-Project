from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core import Document

import uuid

db_location = "./chroma_db"

# Placing it here so it is initialized only once
# To prevent hitting any rate limits from the Ollama API
EMBEDDING_FUNCTIONS = {
    "nomic-embed-text": OllamaEmbeddings(model="nomic-embed-text"),
    "mxbai-embed-large": OllamaEmbeddings(model="mxbai-embed-large"),
    "bge-m3": OllamaEmbeddings(model="bge-m3"),
    "snowflake-arctic-embed": OllamaEmbeddings(model="snowflake-arctic-embed"),
    "snowflake-arctic-embed2": OllamaEmbeddings(model="snowflake-arctic-embed2"),
    "all-minilm": OllamaEmbeddings(model="all-minilm"),
}

def get_collection_names_from_dim(dim_size, custom_suffix=None) -> str:
    '''
    Returns the name of the collection based on the dimension size and an optional custom suffix.
    '''
    return "dim_{}_collection".format(dim_size) + ("" if custom_suffix is None else "_{}".format(custom_suffix))

def get(query: str, embedding_func: str, custom_suffix: str = None, n_results: int = 10) -> list[Document]:
    '''
    Queries the embedding database for the given query embedding.
    Will search for the collection with the same dim, can be differentiated using a custom suffix.
    ''' 
        
    if query is None:
        raise ValueError("Query cannot be None.")
    
    if embedding_func not in EMBEDDING_FUNCTIONS:
        raise ValueError(f"Embedding function '{embedding_func}' is not supported. Available functions: {list(EMBEDDING_FUNCTIONS.keys())}")
    
    collection_name = get_collection_names_from_dim(len(query[0]), custom_suffix)
    client = Chroma(
        persist_directory=db_location,
        embedding_function=EMBEDDING_FUNCTIONS.get(embedding_func),
        collection_name=collection_name,
    )
    
    documents = client.search(query=query, search_type="similarrity")
    
    return documents

def insert(data: str, embedding_func: str, document=None, custom_suffix=None, metadata=None) -> None:
    '''
    Adds an embedding to the vector database.
    '''
    if data is None:
        raise ValueError("Data cannot be None.")
    
    collection_name = get_collection_names_from_dim(len(embeddings[0]), custom_suffix)
    client = Chroma(
        persist_directory=db_location,
        embedding_function=EMBEDDING_FUNCTIONS.get(embedding_func),
        collection_name=collection_name,
    )

def list_collections() -> list:
    '''
    Lists all collections in the vector database.
    '''
    collections = CLIENT.list_collections()
    collection_names = [collection.name for collection in collections]
    return collection_names

def delete_collection(collection_name: str) -> None:
    '''
    Deletes a collection from the vector database.
    '''
    CLIENT.delete_collection(collection_name)
    print(f"Collection '{collection_name}' deleted.")
    
def cleanup() -> None:
    '''
    Cleans up the vector database by deleting all collections.
    To keep the database clean, this function deletes collections that have "example" or "test" in their names.
    '''
    collections = CLIENT.list_collections()
    for collection in collections:
        if "example" in collection.name or "test" in collection.name:
            print(f"Deleting collection: {collection.name}")
            CLIENT.delete_collection(collection.name)
    
if __name__ == "__main__":
    embeddings = [[0.1 for _ in range(768)] for _ in range(5)]
    document = "This is a sample document."
    custom_suffix = "example"
    
    print(embeddings)
    
    add_embedding(
        embeddings=embeddings,
        document=document,
        custom_suffix=custom_suffix,
        metadata=None
    )
    
    query_result = query_embedding(embeddings, custom_suffix)
    print(query_result)
    
    print(CLIENT.list_collections())
    
    cleanup()