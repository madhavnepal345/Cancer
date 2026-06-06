import pickle
import numpy as np
from pathlib import Path
from .qa_types import RetrievedChunk
from ..embeddings import EmbeddingGenerator
import faiss

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

class Retriever:
    def __init__(self, faiss_index_path: str = str(DATA_DIR / "cancer_index_checkpoint.faiss"),
                 chunks_pkl_path: str = str(DATA_DIR / "cancer_chunks.pkl"),
                 top_k: int = None):
        from .. import config  # lazy import to avoid cycles in some envs
        self.top_k = top_k if top_k is not None else config.TOP_K
        self.embedder = EmbeddingGenerator()
        self.dimension = self.embedder.model.get_sentence_embedding_dimension()

        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)
        if not isinstance(self.index, faiss.Index):
            raise TypeError(f"Loaded index is not a FAISS Index object, got {type(self.index)}")

        # Load chunks
        with open(chunks_pkl_path, "rb") as f:
            self.chunks = pickle.load(f)
        if not isinstance(self.chunks, list):
            raise TypeError(f"Chunks must be a list, got {type(self.chunks)}")

    def fetch(self, query_text: str, top_k: int = None):
        top_k = top_k or self.top_k

        if self.index.ntotal == 0:
            return []

        q_emb = self.embedder.embed_texts([query_text]).astype(np.float32)
        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            sim_score = float(distances[0][i])
            results.append(RetrievedChunk(
                id=chunk.get("id", str(idx)),
                text=chunk.get("text", ""),
                score=sim_score,
                metadata=chunk.get("metadata", {})
            ))
        return results
