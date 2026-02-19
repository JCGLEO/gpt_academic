from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import google.generativeai as genai
from pypdf import PdfReader

from .config import settings


def load_documents(data_dir: Path) -> Iterable[dict]:
    for fp in sorted(data_dir.glob("**/*")):
        if not fp.is_file():
            continue

        suffix = fp.suffix.lower()

        if suffix in {".txt", ".md"}:
            content = fp.read_text(encoding="utf-8", errors="ignore")
            yield {"title": fp.stem, "content": content, "source": str(fp), "tags": []}

        elif suffix == ".jsonl":
            for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                yield {
                    "title": row.get("title", fp.stem),
                    "content": row.get("content", ""),
                    "source": str(fp),
                    "tags": row.get("tags", []),
                }

        elif suffix == ".pdf":
            reader = PdfReader(str(fp))
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
            yield {
                "title": fp.stem,
                "content": "\n".join(pages),
                "source": str(fp),
                "tags": ["pdf"],
            }


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def embed_batch(texts: list[str]) -> np.ndarray:
    vectors = []
    for t in texts:
        emb = genai.embed_content(
            model=settings.embed_model,
            content=t,
            task_type="retrieval_document",
        )
        vectors.append(np.array(emb["embedding"], dtype=np.float32))
    return np.vstack(vectors)


def main() -> None:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    genai.configure(api_key=settings.gemini_api_key)

    data_dir = Path(settings.data_dir)
    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    chunk_texts: list[str] = []

    for doc in load_documents(data_dir):
        for i, c in enumerate(chunk_text(doc["content"])):
            payload = {
                "title": doc.get("title", "untitled"),
                "content": c,
                "source": doc.get("source", ""),
                "chunk_id": i,
                "tags": doc.get("tags", []),
            }
            all_chunks.append(payload)
            chunk_texts.append(c)

    if not chunk_texts:
        raise ValueError(f"No valid documents found in {data_dir}")

    matrix = embed_batch(chunk_texts)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-12, None)
    index.add(matrix)

    faiss.write_index(index, str(index_dir / "faiss.index"))
    with (index_dir / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Indexed chunks: {len(all_chunks)}")
    print(f"Saved to: {index_dir}")


if __name__ == "__main__":
    main()
