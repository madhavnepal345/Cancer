import pickle
import os
from typing import List, Optional,Tuple
from pathlib import Path
from .qa_types import RetrievedChunk
import numpy as np
from ..embeddings import EmbeddingGenerator

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

class KnowledgeBase:
    def __init__(self, faiss_pkl_path: str = str(DATA_DIR / "cancer_chunks.pkl"), top_k: int = None):
        from .. import config  # local import to avoid circular dependency
        self.top_k = top_k if top_k is not None else config.TOP_K
        # Fallback static entries
        self.entries: List[Tuple[List[str], str]] = [
            (["emergency", "urgent"], "Call local emergency number immediately."),
            (["screening", "mammogram"], "Breast cancer screening info..."),
            (["hotline", "helpline"], "Contact your national cancer helpline."),
            (["disclaimer"], "Always consult a qualified healthcare professional."),
        ]

        # Load FAISS index + chunks
        self.index = None
        self.chunks = []
        if os.path.exists(faiss_pkl_path):
            with open(faiss_pkl_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "index" in data and "chunks" in data:
                self.index = data["index"]
                self.chunks = data["chunks"]
            elif isinstance(data, list):
                self.chunks = data
        self.embedder = EmbeddingGenerator()
        self.dimension = self.embedder.dimension

    def maybe_answer(self, question: str) -> Optional[str]:
        q = question.lower()
        for keywords, reply in self.entries:
            if all(k in q for k in keywords):
                return reply
        return None

    def retrieve(self, question: str, top_k: int = None) -> List[RetrievedChunk]:
        top_k = top_k or self.top_k
        if self.index is None or not self.chunks:
            return []
        # Embed the query
        q_emb = self.embedder.embed_texts([question]).astype(np.float32)
        # Search FAISS
        distances, indices = self.index.search(q_emb, top_k)
        # Convert to RetrievedChunk
        results = []
        for i, idx in enumerate(indices[0]):
            chunk = self.chunks[idx]
            results.append(RetrievedChunk(
                id=chunk.get("id", str(idx)),
                text=chunk["text"],
                score=float(distances[0][i]),
                metadata=chunk.get("metadata", {})
            ))
        return results
