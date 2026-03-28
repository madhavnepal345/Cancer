import pickle
import logging
from typing import List, Optional
import numpy as np
from embeddings import EmbeddingGenerator  

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RetrievedChunk:
   
    def __init__(self, text: str, metadata: Optional[dict] = None, score: float = 0.0):
        self.text = text
        self.metadata = metadata or {}
        self.score = score


class FaissRetriever:
    
    def __init__(self, pkl_path: str, embedder: EmbeddingGenerator):
     
        self.embedder = embedder
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self.index = data["index"]
            self.chunks = data["chunks"]
            logger.info(f"Loaded FAISS index with {len(self.chunks)} chunks from {pkl_path}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise e

    def fetch(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        from backend import config  # avoid top-level circular import
        top_k = top_k if top_k is not None else config.TOP_K
       
        try:
            # 1. Embed query
            query_emb = self.embedder.embed_texts([query])

            # 2. Search in FAISS
            distances, indices = self.index.search(query_emb, top_k)

            results = []
            for idx, score in zip(indices[0], distances[0]):
                # Handle invalid index (sometimes FAISS returns -1)
                if idx < 0 or idx >= len(self.chunks):
                    continue

                chunk_data = self.chunks[idx]
                results.append(
                    RetrievedChunk(
                        text=chunk_data.get("text", ""),
                        metadata=chunk_data.get("metadata", {}),
                        score=float(score)
                    )
                )
            return results

        except Exception as e:
            logger.error(f"FAISS retrieval failed: {e}")
            return []
