from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import ChatRequest, ChatResponse
from .rag import RecipeRAG

app = FastAPI(title="Recipe R&D Assistant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_instance: RecipeRAG | None = None


@app.on_event("startup")
def setup_rag() -> None:
    global rag_instance
    try:
        rag_instance = RecipeRAG()
    except Exception as exc:
        print(f"[WARN] RAG init failed: {exc}")
        rag_instance = None


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model": settings.gemini_model,
        "index_loaded": rag_instance is not None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG index not ready, run data pipeline first")

    answer, contexts = rag_instance.generate(req.query)
    return ChatResponse(answer=answer, contexts=contexts, model=settings.gemini_model)
