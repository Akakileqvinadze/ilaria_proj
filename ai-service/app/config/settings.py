from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Keys
    google_api_key: str
    huggingface_api_key: Optional[str] = None

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "development"

    # Models
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_store: str = "faiss"

    # RAG Config
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 4
    temperature: float = 0.7
    max_tokens: int = 1024

    # Paths
    data_path: str = "app/data"
    index_path: str = "faiss_index"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
