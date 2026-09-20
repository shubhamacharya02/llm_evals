import os
import shutil
from typing import List
from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.settings import settings


def save_uploaded_file(file: UploadFile, destination_dir: str = settings.DATA_DIR) -> str:
    """
    Saves an uploaded FastAPI UploadFile to the specified destination directory.
    """
    os.makedirs(destination_dir, exist_ok=True)
    file_path = os.path.join(destination_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return file_path


def load_document(file_path: str) -> List[Document]:
    """
    Loads a document using appropriate LangChain document loader based on file extension.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext in [".txt", ".md", ".log"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        loader = PyMuPDFLoader(file_path)
        
    return loader.load()


def split_documents(
    documents: List[Document],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> List[Document]:
    """
    Splits loaded documents into smaller chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


def load_and_split_document(
    file_path: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> List[Document]:
    """
    Loads a document and splits it into chunks.
    """
    docs = load_document(file_path)
    return split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


_EMBEDDING_MODEL = None
_VECTOR_STORE = None


def get_embedding_model():
    """
    Initializes and returns the HuggingFace embedding model (cached singleton).
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _EMBEDDING_MODEL


def get_chroma_vector_store(
    embeddings=None,
    persist_directory: str = settings.CHROMA_DIR,
    collection_name: str = settings.COLLECTION_NAME,
    force_new: bool = False
):
    """
    Initializes and connects to the persistent Chroma DB vector store (cached singleton).
    """
    global _VECTOR_STORE
    if _VECTOR_STORE is not None and not force_new and embeddings is None:
        return _VECTOR_STORE

    if embeddings is None:
        embeddings = get_embedding_model()
        
    os.makedirs(persist_directory, exist_ok=True)
    
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    if embeddings is None or embeddings == _EMBEDDING_MODEL:
        _VECTOR_STORE = store
    return store


def create_vector_store(
    documents: List[Document],
    embeddings=None,
    persist_directory: str = settings.CHROMA_DIR,
    collection_name: str = settings.COLLECTION_NAME
):
    """
    Creates and populates a Chroma vector store from document chunks.
    """
    if embeddings is None:
        embeddings = get_embedding_model()
        
    os.makedirs(persist_directory, exist_ok=True)
    
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )


def get_retriever(
    vector_store=None,
    top_k: int = settings.TOP_K,
    search_type: str = settings.SEARCH_TYPE
):
    """
    Creates a LangChain retriever from a Chroma vector store.
    
    Args:
        vector_store: Chroma vector store instance. Defaults to get_chroma_vector_store().
        top_k (int): Number of relevant documents to retrieve. Defaults to settings.TOP_K.
        search_type (str): Search type ('similarity' or 'mmr'). Defaults to settings.SEARCH_TYPE.
        
    Returns:
        VectorStoreRetriever: Configured retriever instance.
    """
    if vector_store is None:
        vector_store = get_chroma_vector_store()
        
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": top_k}
    )


def semantic_search(
    query: str,
    vector_store=None,
    top_k: int = settings.TOP_K
) -> List[Document]:
    """
    Performs semantic similarity search against the Chroma vector store for a given query.
    
    Args:
        query (str): User question / query string.
        vector_store: Chroma instance. Defaults to get_chroma_vector_store().
        top_k (int): Number of top similar chunks to return. Defaults to settings.TOP_K.
        
    Returns:
        List[Document]: Top matching document chunks.
    """
    if vector_store is None:
        vector_store = get_chroma_vector_store()
    return vector_store.similarity_search(query=query, k=top_k)


def semantic_search_with_score(
    query: str,
    vector_store=None,
    top_k: int = settings.TOP_K
) -> List[tuple]:
    """
    Performs semantic search returning both matching document chunks and their similarity scores.
    
    Args:
        query (str): User question / query string.
        vector_store: Chroma instance. Defaults to get_chroma_vector_store().
        top_k (int): Number of top similar chunks to return. Defaults to settings.TOP_K.
        
    Returns:
        List[Tuple[Document, float]]: List of (Document, score) pairs.
    """
    if vector_store is None:
        vector_store = get_chroma_vector_store()
    return vector_store.similarity_search_with_score(query=query, k=top_k)


def clear_vector_store(
    persist_directory: str = settings.CHROMA_DIR,
    collection_name: str = settings.COLLECTION_NAME
) -> None:
    """
    Safely resets the Chroma collection and clears persistent cache.
    """
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        try:
            _VECTOR_STORE.delete_collection()
        except Exception:
            pass
        _VECTOR_STORE = None


def process_and_index_document(file_path: str, reset: bool = False) -> int:
    """
    Loads, splits, and persists document chunks into Chroma DB.
    
    Args:
        file_path (str): Path to the uploaded document.
        reset (bool): If True, wipes existing Chroma DB before indexing. Defaults to False.
        
    Returns:
        int: Number of chunked documents indexed into Chroma DB.
    """
    if reset:
        clear_vector_store()
        
    chunks = load_and_split_document(file_path)
    vector_store = get_chroma_vector_store(force_new=reset)
    vector_store.add_documents(chunks)
    return len(chunks)


def retrieve_relevant_chunks(
    query: str,
    top_k: int = None
) -> List[dict]:
    """
    Retrieves the most relevant chunks from Chroma DB for a query.
    
    Args:
        query (str): Search query string.
        top_k (int, optional): Number of results to retrieve. Defaults to settings.TOP_K.
        
    Returns:
        List[dict]: Formatted document results with content, metadata, and score.
    """
    k = top_k if top_k is not None and top_k > 0 else settings.TOP_K
    vector_store = get_chroma_vector_store()
    
    try:
        results_with_scores = vector_store.similarity_search_with_score(query=query, k=k)
        formatted_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score) if score is not None else None
            }
            for doc, score in results_with_scores
        ]
    except Exception:
        docs = vector_store.similarity_search(query=query, k=k)
        formatted_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": None
            }
            for doc in docs
        ]
        
    return formatted_results