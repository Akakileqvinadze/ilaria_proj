from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingService:
    def __init__(self, model_name: str):
        print(f"📥 იტვირთება embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Embedding model მზადაა (dimension: {self.dimension})")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        print(f"🔄 Embedding {len(texts)} დოკუმენტი...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"✅ Embeddings generated: {embeddings.shape}")
        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding

    def get_dimension(self) -> int:
        return self.dimension
