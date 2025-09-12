import pickle
import numpy as np
from retrieval.qa_types import RetrievedChunk
from embeddings import EmbeddingGenerator

class Retriever:
    def __init__(self, faiss_pkl_path: str = "data/cancer_chunks.pkl", top_k: int = 5):
        self.top_k = top_k
        self.embedder = EmbeddingGenerator()
        self.dimension = self.embedder.model.get_sentence_embedding_dimension()

        with open(faiss_pkl_path, "rb") as f:
            data = pickle.load(f)

        self.index = data["index"]      # FAISS index
        self.chunks = data["chunks"]    # List of dicts with 'id', 'text', 'metadata'

    def fetch(self, query_text: str):
        if self.index.ntotal == 0:
            return []

        q_emb = self.embedder.embed_texts([query_text]).astype(np.float32)
        distances, indices = self.index.search(q_emb, self.top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            chunk = self.chunks[idx]
            sim_score = float(distances[0][i])
            results.append(RetrievedChunk(
                id=chunk.get("id", str(idx)),
                text=chunk["text"],
                score=sim_score,
                metadata=chunk.get("metadata", {})
            ))
        return results
