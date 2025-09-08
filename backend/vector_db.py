import json
import chromadb
from embeddings import EmbeddingGenerator
from retrieval.qa_types import RetrievedChunk
from typing import List, Dict, Any
import os
import glob
import uuid
import re
import traceback

# Example keywords for extraction (you can expand this list)
CANCER_TYPES = [
    "glioma", "melanoma", "lung cancer", "kidney cancer",
    "squamous cell carcinoma", "breast cancer", "leukemia", "lymphoma"
]
ORGANS = [
    "brain", "skin", "lung", "kidney", "breast", "liver", "colon"
]
TUMOR_CHARACTERISTICS = [
    "high-grade", "low-grade", "infiltrating", "benign", "malignant"
]
TREATMENTS = [
    "chemotherapy", "radiation therapy", "surgery", "immunotherapy"
]

def extract_metadata(text: str) -> Dict[str, List[str]]:
    """Extract cancer types, organs, tumor characteristics, treatments from chunk text."""
    text_lower = text.lower()
    return {
        "cancer_types": [c for c in CANCER_TYPES if c.lower() in text_lower],
        "organs_affected": [o for o in ORGANS if o.lower() in text_lower],
        "tumor_characteristics": [t for t in TUMOR_CHARACTERISTICS if t.lower() in text_lower],
        "treatments": [tr for tr in TREATMENTS if tr.lower() in text_lower]
    }

class VectorDB:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "cancer_awareness", auto_load: bool = True):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        print("Initialized VectorDB with collection:", collection_name)
        self.debug_info()

        if auto_load:
            if self.collection.count() == 0:
                print("Collection is empty, attempting auto-load")
                self.auto_load_documents()
            else:
                print("Collection already has documents, skipping auto-load")

    def load_chunks_from_json(self, json_path: str) -> List[Dict[str, str]]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def add_documents(self, json_path: str):
        try:
            print("Loading documents from:", json_path)
            chunks = self.load_chunks_from_json(json_path)
            print("Loaded", len(chunks), "chunks from JSON")

            texts = []
            ids = []
            metadatas = []

            for chunk in chunks:
                if "text" in chunk and "id" in chunk:
                    texts.append(chunk["text"])
                    ids.append(str(chunk["id"]))
                    metadata = extract_metadata(chunk["text"])
                    metadata["chunk_id"] = str(chunk["id"])
                    metadatas.append(metadata)

            if not texts:
                print("No valid chunks found in JSON file")
                return

            print("Embedding", len(texts), "texts")
            embeddings = self.embedder.embed_texts(texts)

            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
            print("Added", len(texts), "documents to ChromaDB")
            print("Total documents now:", self.collection.count())

        except FileNotFoundError:
            print("JSON file not found:", json_path)
        except Exception as e:
            print("Error adding documents:", e)
            traceback.print_exc()

    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            if count == 0:
                print("Collection is empty")
                return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

            query_embedding = self.embedder.embed_texts([query_text])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(n_results, count)
            )

            print("Retrieved", len(results.get("documents", [[]])[0]), "results")
            return results
        except Exception as e:
            print("Error querying ChromaDB:", e)
            traceback.print_exc()
            return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

    def query_as_chunks(self, query_text: str, n_results: int = 3) -> List[RetrievedChunk]:
        results = self.query(query_text, n_results)
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        chunks = []
        for i, doc in enumerate(docs):
            if i < len(ids) and i < len(dists):
                chunks.append(RetrievedChunk(
                    id=str(ids[i]),
                    text=doc,
                    score=1.0 - dists[i],
                    metadata=metadatas[i] if i < len(metadatas) else {}
                ))
        return chunks

    def debug_info(self):
        try:
            count = self.collection.count()
            print("Collection", self.collection.name, "has", count, "documents")
            if count > 0:
                sample = self.collection.peek(limit=1)
                print("Sample document IDs:", sample.get("ids", []))
                sample_text = sample.get("documents", [[]])[0]
                if sample_text:
                    print("Sample text first 100 chars:", sample_text[0][:100])
            else:
                print("Database is empty")
        except Exception as e:
            print("Debug error:", e)
            traceback.print_exc()

    def auto_load_documents(self, search_paths=None):
        if search_paths is None:
            search_paths = ["./data/"]

        json_files = []
        for path in search_paths:
            if os.path.exists(path):
                found_files = glob.glob(os.path.join(path, "**", "*.json"), recursive=True)
                json_files.extend(found_files)
                print("Found", len(found_files), "JSON files in", path)

        if not json_files:
            print("No JSON files found in search paths")
            return False

        print("JSON files to load:", json_files)
        for json_file in json_files:
            try:
                self.add_documents(json_file)
            except Exception as e:
                print("Failed to load", json_file, ":", e)
                traceback.print_exc()

        return True

# Test block
if __name__ == "__main__":
    print("Running VectorDB standalone test")
    db = VectorDB(auto_load=True)
    db.debug_info()
