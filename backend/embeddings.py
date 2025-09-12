# embedding.py (build_faiss_index.py style)
import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts):
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings.astype("float32")


def build_and_save_faiss(json_path: str, output_pkl: str, index_type="FlatIP"):
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_texts(texts)

    if index_type == "FlatIP":
        index = faiss.IndexFlatIP(embedder.dimension)
    else:
        raise ValueError("Unsupported index type")

    index.add(embeddings)

    data_to_save = {"index": index, "chunks": chunks}
    with open(output_pkl, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Saved FAISS index + chunks to {output_pkl}")



if __name__ == "__main__":
    build_and_save_faiss(
        json_path="data/Combined_Cancer_Chunks.json",   
        output_pkl="data/cancer_chunks.pkl",
        index_type="FlatIP"  
    )
