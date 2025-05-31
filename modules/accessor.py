import chromadb
from chromadb.api.types import QueryResult

import uuid

CLIENT = chromadb.PersistentClient(path="./chroma_db")

def get_collection_names_from_dim(dim_size, custom_suffix=None) -> str:
    '''
    Returns the name of the collection based on the dimension size and an optional custom suffix.
    '''
    return "dim_{}_collection".format(dim_size) + ("" if custom_suffix is None else "_{}".format(custom_suffix))

def query_embedding(query, custom_suffix=None, n_results=10, metadata_query=None) -> QueryResult:
    '''
    Queries the embedding database for the given query embedding.
    Will search for the collection with the same dim, can be differentiated using a custom suffix.
    ''' 
        
    if query is None or not isinstance(query[0], list) or not isinstance(query[0][0], float):
        raise ValueError("Query embedding must be a list of lists.")
    
    collection_name = get_collection_names_from_dim(len(query[0]), custom_suffix)
    collection = CLIENT.get_or_create_collection(collection_name)
    
    results = collection.query(
        query_embeddings=query,
        n_results=n_results,
        where=metadata_query,
        include=["embeddings", "documents", "metadatas"]
    )
    
    return results

def add_embedding(embeddings, document=None, custom_suffix=None, metadata=None) -> None:
    '''
    Adds an embedding to the vector database.
    '''
    if embeddings is None or not isinstance(embeddings[0], list) or not isinstance(embeddings[0][0], float):
        raise ValueError("Embeddings must be a list of lists.")
    
    entry_count = len(embeddings)
    
    collection_name = get_collection_names_from_dim(len(embeddings[0]), custom_suffix)
    collection = CLIENT.get_or_create_collection(collection_name)
    
     # document structuring
    documents = None
    if document:
        if isinstance(document, str):
            documents = [document for _ in range(entry_count)]
        
    # populate metadata
    if custom_suffix is not None or document is not None:
        if metadata is None:
            metadata = {} # create an empty metadata dictionary if none is provided
            
        # carry metadata with entry
        if custom_suffix:
            metadata['custom_suffix'] = custom_suffix
        if document:
            metadata['document'] = document
        
        metadata = [metadata for _ in range(entry_count)]        
    
    
    try:
        # this function is basically a bulk insert, hence all embeddings, documents, and metadata should be lists of the same length
        collection.add(
            embeddings=embeddings,
            documents=documents if documents else None,
            metadatas=metadata if metadata else None,
            ids=[str(uuid.uuid4()) for _ in range(len(embeddings))]
        )   
    except Exception as e:
        print(f"Error adding embedding: {e}")
        print("Dimension of embeddings:", len(embeddings[0]) if embeddings else "None")
        raise e

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