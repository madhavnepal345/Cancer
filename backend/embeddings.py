from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts):
        """
        Generate embeddings for a list of texts.
        Returns numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True  # good for cosine similarity
        )
        return embeddings.astype('float32')

    def build_faiss_index(self, embeddings, index_type="FlatL2"):
        """
        Build a FAISS index from embeddings.
        index_type: 'FlatL2' (exact L2 search) or 'FlatIP' (cosine similarity)
        """
        if index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError("Unsupported index type. Use 'FlatL2' or 'FlatIP'.")
        
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def save_index(self, index_path):
        """Save FAISS index to disk."""
        if self.index is None:
            raise ValueError("Index is not built yet.")
        faiss.write_index(self.index, index_path)
        print(f"FAISS index saved at {index_path}")

    def load_index(self, index_path):
        """Load FAISS index from disk."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No index found at {index_path}")
        self.index = faiss.read_index(index_path)
        print(f"FAISS index loaded from {index_path}, total vectors: {self.index.ntotal}")

    def search(self, query_embedding, top_k=5):
        """
        Search FAISS index for nearest neighbors.
        Returns distances and indices of top_k results.
        """
        if self.index is None:
            raise ValueError("Index is not built or loaded.")
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        return distances[0], indices[0]

