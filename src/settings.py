import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


@dataclass
class Settings:
    # Document Processing / Chunking Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Retriever & Search Settings
    TOP_K: int = int(os.getenv("TOP_K", 4))
    SEARCH_TYPE: str = os.getenv("SEARCH_TYPE", "similarity")  # similarity or mmr
    
    # LLM Model Settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.0))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 1024))
    
    # Embedding Model Settings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Directory & Storage Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    CHROMA_DIR: str = os.path.join(DATA_DIR, "chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "evals_collection")
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    HUGGINGFACEHUB_API_TOKEN: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")


# Global settings instance
settings = Settings()
