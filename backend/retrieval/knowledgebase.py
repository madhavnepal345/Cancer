import pickle
from typing import List, Optional,Tuple
from retrieval.qa_types import RetrievedChunk
import numpy as np
from embeddings import EmbeddingGenerator

class KnowledgeBase:
    def __init__(self, faiss_pkl_path: str = "data/cancer_chunks.pkl", top_k: int = None):
        from backend import config  # local import to avoid circular dependency
        self.top_k = top_k if top_k is not None else config.TOP_K
        # Fallback static entries
        self.entries: List[Tuple[List[str], str]] = [
            (["emergency", "urgent"], "Call local emergency number immediately."),
            (["screening", "mammogram"], "Breast cancer screening info..."),
            (["hotline", "helpline"], "Contact your national cancer helpline."),
            (["disclaimer"], "Always consult a qualified healthcare professional."),
        ]

        # Load FAISS index + chunks
        with open(faiss_pkl_path, "rb") as f:
            data = pickle.load(f)
        self.index = data[0]
        self.chunks = data[1]
        self.embedder = EmbeddingGenerator()
        self.dimension = self.embedder.model.get_sentence_embedding_dimension()

    def maybe_answer(self, question: str) -> Optional[str]:
        q = question.lower()
        for keywords, reply in self.entries:
            if all(k in q for k in keywords):
                return reply
        return None

    def retrieve(self, question: str, top_k: int = None) -> List[RetrievedChunk]:
        top_k = top_k or self.top_k
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
