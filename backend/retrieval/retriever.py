import pickle
import numpy as np
from retrieval.qa_types import RetrievedChunk
from embeddings import EmbeddingGenerator
import faiss

class Retriever:
    def __init__(self, faiss_index_path: str = "data/cancer_index_checkpoint.faiss",
                 chunks_pkl_path: str = "data/cancer_chunks.pkl",
                 top_k: int = 5):
        self.top_k = top_k
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
