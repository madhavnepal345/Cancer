import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import time
import torch
from pathlib import Path

try:
    from . import config  # pragma: no cover
except ImportError:  # pragma: no cover
    import config

os.environ["OMP_NUM_THREADS"] = "4"  
os.environ["MKL_NUM_THREADS"] = "4"

device = 0 if torch.cuda.is_available() else -1  
print("Using device:", device)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

class EmbeddingGenerator:
    def __init__(self, model_name: str = None):
        self.model = SentenceTransformer(model_name or config.EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts, batch_size=16):
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=batch_size
            

        )
        return embeddings.astype("float32")


def build_and_save_faiss_checkpoint(
    json_path: str,
    output_index_path: str,
    output_chunks_path: str,
    index_type="FlatIP",
    batch_size=16,
    sleep_sec=2
):

    # Load chunks
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedder = EmbeddingGenerator()

    # Initialize or load FAISS index
    if os.path.exists(output_index_path):
        print("Resuming from existing FAISS index...")
        index = faiss.read_index(output_index_path)
        start_idx = index.ntotal
        print(f"Starting from chunk {start_idx}")
    else:
        print("Creating new FAISS index...")
        if index_type == "FlatIP":
            index = faiss.IndexFlatIP(embedder.dimension)
        else:
            raise ValueError("Unsupported index type")
        start_idx = 0

    # Loop over batches
    for i in range(start_idx, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch_chunks]

        embeddings = embedder.embed_texts(texts, batch_size=batch_size)
        index.add(embeddings)

        # Save after each batch
        faiss.write_index(index, output_index_path)
        with open(output_chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        print(f"Processed chunks {i} -> {i+len(batch_chunks)} / {len(chunks)}")
        time.sleep(sleep_sec)  

    print(f"FAISS index and chunks saved. Total chunks embedded: {len(chunks)}")


if __name__ == "__main__":
    build_and_save_faiss_checkpoint(
        json_path=str(DATA_DIR / "Combined_Cancer_Chunks.json"),
        output_index_path=str(DATA_DIR / "cancer_index_checkpoint.faiss"),
        output_chunks_path=str(DATA_DIR / "cancer_chunks.pkl"),
        index_type="FlatIP",
        batch_size=16  
        
    )
