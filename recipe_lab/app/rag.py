from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import google.generativeai as genai

from .config import settings


class RecipeRAG:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

        index_dir = Path(settings.index_dir)
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        with (index_dir / "chunks.json").open("r", encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)

    def _embed(self, text: str) -> np.ndarray:
        embedding = genai.embed_content(
            model=settings.embed_model,
            content=text,
            task_type="retrieval_query",
        )
        vector = np.array(embedding["embedding"], dtype=np.float32)
        return vector.reshape(1, -1)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or settings.top_k
        q = self._embed(query)
        _, indices = self.index.search(q, top_k)

        results = []
        for idx in indices[0].tolist():
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(self.chunks[idx])
        return results

    def generate(self, query: str) -> tuple[str, list[str]]:
        docs = self.retrieve(query)
        contexts = [d["content"] for d in docs]

        prompt = (
            "你是一名专业菜谱研发助手。请基于给定资料回答。\n"
            "要求：\n"
            "1) 给出可执行步骤；\n"
            "2) 给出替代食材建议；\n"
            "3) 给出关键失败点与规避方法；\n"
            "4) 如果资料不足，明确指出并给出保守建议。\n\n"
            f"用户需求：{query}\n\n"
            "资料片段：\n"
            + "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
        )

        resp = self.model.generate_content(prompt)
        return resp.text, contexts
