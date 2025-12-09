import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional


class FAISSVectorStore:
    def __init__(self, dimension: int, index_path: str = "faiss_index"):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)

        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []

        self._load_index()

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None
    ):
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        self.documents.extend(texts)

        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(texts))

        self._save_index()
        print(f"✅ დაემატა {len(texts)} დოკუმენტი. სულ: {self.index.ntotal}")

    def similarity_search(
        self, query_embedding: np.ndarray, k: int = 4
    ) -> List[Dict]:
        if self.index.ntotal == 0:
            print("⚠️  Vector store ცარიელია!")
            return []

        query_embedding = query_embedding.astype('float32')
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    "content": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(distance),
                    "index": int(idx)
                })

        return results

    def _save_index(self):
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))

        with open(self.index_path / "documents.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadatas": self.metadatas
            }, f)

        print(f"💾 Index შენახულია: {self.index_path}")

    def _load_index(self):
        index_file = self.index_path / "index.faiss"
        docs_file = self.index_path / "documents.pkl"

        if index_file.exists() and docs_file.exists():
            self.index = faiss.read_index(str(index_file))

            with open(docs_file, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadatas = data["metadatas"]

            print(f"📂 ჩაიტვირთა არსებული index: {self.index.ntotal} დოკუმენტი")
        else:
            print(f"📁 ახალი index შეიქმნება")

    @property
    def count(self) -> int:
        return self.index.ntotal
