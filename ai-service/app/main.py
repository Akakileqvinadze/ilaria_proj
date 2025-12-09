import os
from typing import List, Dict, Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai
from dotenv import load_dotenv  # 🆕 ეს დავამატოთ

load_dotenv()  # 🆕 .env ფაილიდან გარემოს ცვლადების ჩატვირთვა


# -----------------------------
# შეცდომის მესიჯები
# -----------------------------
ERROR_MESSAGES = {
    "empty_query": "გთხოვთ დაწეროთ თქვენი კითხვა",
    "service_unavailable": "სერვისი დროებით მიუწვდომელია",
    "processing_error": "შეცდომა დამუშავებისას",
}


# -----------------------------
# სქემები (request / response)
# -----------------------------
class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ConversationMessage] = []


class Source(BaseModel):
    id: int
    content: str
    section: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    metadata: Dict[str, Any]


# -----------------------------
# RagService – რეალური AI პასუხი (Gemini)
# -----------------------------
class RagService:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY არ არის განსაზღვრული .env ფაილში")

        genai.configure(api_key=api_key)

        # შეგიძლია შეცვალო მოდელის სახელი
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    async def generate_response(
        self,
        query: str,
        history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        history სტრუქტურა:
        [
          {"role": "user" | "assistant", "content": "..."},
          ...
        ]
        """

        # ვაშენებთ prompt-ს: მთელი ისტორია + ახალი კითხვა
        history_text_lines: List[str] = []
        for msg in history:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            history_text_lines.append(f"{prefix}: {msg['content']}")

        history_text = "\n".join(history_text_lines) if history_text_lines else "—"

        system_instruction = (
            "შენ ხარ დამხმარე AI ასისტენტი Subconscious აპისთვის. "
            "უპასუხე მკაფიოდ, სტრუქტურირებულად და მაქსიმალურად სასარგებლოდ. "
            "თუ კითხვა არ არის ნათელი, სთხოვი მომხმარებელს დაზუსტებას."
        )

        full_prompt = (
            f"{system_instruction}\n\n"
            f"საუბრის ისტორია:\n{history_text}\n\n"
            f"ახლა მომხმარებლის ახალი კითხვა:\nUser: {query}\n\n"
            "გთხოვ დეტალური და გასაგები პასუხი ქართულად."
        )

        try:
            # Gemini-ის გამოძახება (სინქრონულია, მაგრამ FastAPI ამას ისევე ითმენს
            # თუ გინდა, შემდგომში ThreadPoolExecutor-ით გადავიტანთ)
            response = self.model.generate_content(full_prompt)

            if not response or not response.text:
                answer_text = "ვერ შევძელი ამ კითხვის დამუშავება, სცადე სხვა ფორმულირება."
            else:
                answer_text = response.text

            # აქ შეგიძლია დაამატო რეალური წყაროების ლოგიკა (RAG + vector DB),
            # ჯერჯერობით ვაბრუნებთ დეფოლტს
            sources = [
                {
                    "content": "პასუხი გენერირებულია Gemini მოდელით მოცემული ისტორიის და კითხვის საფუძველზე.",
                    "section": "model: gemini-2.5-flash",
                }
            ]

            metadata = {
                "model": "gemini-2.5-flash",
                "has_history": bool(history),
            }

            return {
                "answer": answer_text,
                "source_documents": sources,
                "metadata": metadata,
            }

        except Exception as e:
            # ეს შეცდომა ავა მაღლა და დაიჭერს /api/chat
            raise RuntimeError(f"Gemini error: {e}") from e


# ვცდილობთ ავაგოთ RagService
try:
    rag_service: RagService | None = RagService()
except Exception as e:
    print(f"❌ RagService init error: {e}")
    rag_service = None


# -----------------------------
# FastAPI აპი
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production-ში შეგიძლია შეცვალო კონკრეტული დომენით
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "rag_service_ready": rag_service is not None}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 1) ვამოწმებთ, დაინიციალდა თუ არა RagService
    if not rag_service:
        raise HTTPException(
            status_code=503,
            detail=ERROR_MESSAGES["service_unavailable"],
        )

    # 2) ცარიელი მესიჯი
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES["empty_query"],
        )

    try:
        # 3) ისტორიის მომზადება RagService-სთვის
        history_payload = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_history
        ]

        # 4) რეალური პასუხის გენერაცია
        raw = await rag_service.generate_response(
            query=request.message,
            history=history_payload,
        )

        # 5) გარდაქმნა ChatResponse სქემაში
        sources: List[Source] = []
        for i, doc in enumerate(raw.get("source_documents", [])):
            sources.append(
                Source(
                    id=i,
                    content=doc.get("content", ""),
                    section=doc.get("section", ""),
                )
            )

        response = ChatResponse(
            answer=raw.get("answer", ""),
            sources=sources,
            metadata=raw.get("metadata", {}),
        )
        return response

    except HTTPException:
        raise

    except Exception as e:
        print(f"❌ Error in /api/chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=ERROR_MESSAGES["processing_error"],
        )


# სურვილის შემთხვევაში, რომ /chat-იც მუშაობდეს (ფრონტიდან თუ ეგ გზა მოდის)
@app.post("/chat", response_model=ChatResponse)
async def chat_alias(request: ChatRequest):
    return await chat(request)
