from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

import os

# Define project root relative to this file's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
db_location = os.path.join(PROJECT_ROOT, "data", "chroma_db")

# Placing it here so it is initialized only once
# Ensure that the Ollama server is running and accessible
# The models should be available in the Ollama server
EMBEDDING_FUNCTIONS = {
    "nomic-embed-text": OllamaEmbeddings(model="nomic-embed-text"),
    "mxbai-embed-large": OllamaEmbeddings(model="mxbai-embed-large"),
    "bge-m3": OllamaEmbeddings(model="bge-m3"),
    # "snowflake-arctic-embed": OllamaEmbeddings(model="snowflake-arctic-embed"),
    # "snowflake-arctic-embed2": OllamaEmbeddings(model="snowflake-arctic-embed2"),
    # "all-minilm": OllamaEmbeddings(model="all-minilm"),
}

def get_collection_names_from_dim(embedding_func, custom_suffix=None) -> str:
    '''
    Returns the name of the collection based on the dimension size and an optional custom suffix.
    '''
    return "{}_collection".format(embedding_func) + ("" if custom_suffix is None else "_{}".format(custom_suffix))

def get(query: str, embedding_func: str, custom_suffix: str = None, n_results: int = 10) -> list[Document]:
    '''
    Queries the embedding database for the given query embedding.
    Will search for the collection with the same dim, can be differentiated using a custom suffix.
    ''' 
    if query is None:
        raise ValueError("Query cannot be None.")
    
    if embedding_func not in EMBEDDING_FUNCTIONS:
        raise ValueError(f"Embedding function '{embedding_func}' is not supported. Available functions: {list(EMBEDDING_FUNCTIONS.keys())}")
    
    collection_name = get_collection_names_from_dim(embedding_func, custom_suffix)
    client = Chroma(
        persist_directory=db_location,
        embedding_function=EMBEDDING_FUNCTIONS.get(embedding_func),
        collection_name=collection_name,
    )
    
    documents = client.search(query=query, search_type="similarity")
    
    return documents

def insert(data: list[str], embedding_func: str, custom_suffix=None, metadata=None) -> None:
    '''
    Adds an embedding to the vector database.
    '''
    if data is None:
        raise ValueError("Data cannot be None.")
    
    collection_name = get_collection_names_from_dim(embedding_func, custom_suffix)
    client = Chroma(
        persist_directory=db_location,
        embedding_function=EMBEDDING_FUNCTIONS.get(embedding_func),
        collection_name=collection_name,
    )
    
    ids = client.add_texts(
        texts=data,
        metadatas=[metadata] if metadata else [None],
    )
    
    print(f"Inserted data with IDs: {ids} into collection '{collection_name}'.")

def list_collections() -> list:
    '''
    Lists all collections in the vector database.
    '''
    collections = Chroma(
        persist_directory=db_location
    )._client.list_collections()
    
    return [collection.name for collection in collections]

def delete_collection(collection_name: str) -> None:
    '''
    Deletes a collection from the vector database.
    '''
    Chroma(
        persist_directory=db_location
    )._client.delete_collection(collection_name)
    print(f"Collection '{collection_name}' deleted.")
    
def cleanup() -> None:
    '''
    Cleans up the vector database by deleting all collections.
    To keep the database clean, this function deletes collections that have "example" or "test" in their names.
    '''
    collections = list_collections()
    for collection in collections:
        if "example" in collection or "test" in collection:
            print(f"Deleting collection: {collection}")
            delete_collection(collection)
    
if __name__ == "__main__":
    # Example usage and testing
    custom_suffix = "example"
    
    ids = insert(
        data=["This is an example input."],
        embedding_func="nomic-embed-text",
        custom_suffix=custom_suffix,
        metadata=None
    )
    
    print(ids)
    
    documents = get(
        query="example",
        embedding_func="nomic-embed-text",
        custom_suffix=custom_suffix
    )
    print(documents)
    
    print(documents[0].page_content)
    
    # Seems like a collection called "langchain" is created by default
    print(list_collections())
    
    cleanup()