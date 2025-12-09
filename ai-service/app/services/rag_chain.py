import google.generativeai as genai
from typing import List, Dict
from app.services.document_loader import DocumentLoader
from app.services.embeddings import EmbeddingService
from app.services.vector_store import FAISSVectorStore
from app.config.settings import settings
from app.config.constants import SYSTEM_PROMPT_TEMPLATE


class RAGService:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.doc_loader = DocumentLoader(settings.data_path)

    async def initialize(self):
        print("=" * 60)
        print("🚀 RAG Service ინიციალიზაცია...")
        print("=" * 60)

        # 1. Gemini
        genai.configure(api_key=settings.google_api_key)
        self.llm = genai.GenerativeModel(settings.llm_model)
        print(f"✅ Gemini: {settings.llm_model}")

        # 2. Embeddings
        self.embeddings = EmbeddingService(settings.embedding_model)

        # 3. Vector Store
        self.vector_store = FAISSVectorStore(
            dimension=self.embeddings.get_dimension(),
            index_path=settings.index_path
        )

        # 4. Load documents
        if self.vector_store.count == 0:
            await self._load_documents()
        else:
            print(f"✅ Vector store უკვე შევსებულია: {self.vector_store.count} დოკუმენტი")

        print("=" * 60)
        print("🎉 RAG Service მზადაა!")
        print("=" * 60)

    async def _load_documents(self):
        print("📄 ilaria-instructions.md იტვირთება...")

        try:
            chunks = self.doc_loader.load_and_split(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )

            texts = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]

            print("🔄 Embeddings იქმნება...")
            embeddings = self.embeddings.embed_documents(texts)

            self.vector_store.add_documents(texts, embeddings, metadatas)
            print(f"✅ დაინდექსირდა {len(chunks)} chunk")

        except FileNotFoundError as e:
            print(f"❌ შეცდომა: {e}")
            raise

    async def generate_response(
        self, query: str, history: List[Dict] = None
    ) -> Dict:
        if not query.strip():
            raise ValueError("Query არ შეიძლება იყოს ცარიელი")

        try:
            # 1. Embed query
            query_embedding = self.embeddings.embed_query(query)

            # 2. Search similar docs
            similar_docs = self.vector_store.similarity_search(
                query_embedding, k=settings.top_k_results
            )

            if not similar_docs:
                return {
                    "answer": "უკაცრავად, ვერ ვიპოვე რელევანტური ინფორმაცია.",
                    "sources": [],
                    "metadata": {"retrieved_docs": 0}
                }

            # 3. Prepare context
            context = "\n\n".join([
                f"[დოკუმენტი {i+1}]:\n{doc['content']}"
                for i, doc in enumerate(similar_docs)
            ])

            # 4. Format history
            history_text = ""
            if history:
                history_text = "\n".join([
                    f"{'მომხმარებელი' if h.get('role') == 'user' else 'ასისტენტი'}: {h.get('content', '')}"
                    for h in history[-5:]
                ])

            # 5. Create prompt
            prompt = SYSTEM_PROMPT_TEMPLATE.format(
                context=context,
                history=history_text if history_text else "არ არის",
                query=query
            )

            # 6. Generate with Gemini
            response = self.llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=settings.temperature,
                    max_output_tokens=settings.max_tokens,
                )
            )

            answer = response.text.strip()

            # 7. Format sources
            sources = [
                {
                    "id": i + 1,
                    "content": doc["content"][:200] + "...",
                    "section": doc["metadata"].get("section", "Unknown")
                }
                for i, doc in enumerate(similar_docs[:3])
            ]

            return {
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "model": settings.llm_model,
                    "retrieved_docs": len(similar_docs)
                }
            }

        except Exception as e:
            print(f"❌ შეცდომა: {str(e)}")
            raise
