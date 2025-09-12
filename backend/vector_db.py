# import json
# import os
# import glob
# import traceback
# import faiss
# import numpy as np
# from embeddings import EmbeddingGenerator
# from retrieval.qa_types import RetrievedChunk
# from typing import List, Dict, Any, Optional

# # Example keywords for metadata extraction
# CANCER_TYPES = [
#     "glioma", "melanoma", "lung cancer", "kidney cancer",
#     "squamous cell carcinoma", "breast cancer", "leukemia", "lymphoma"
# ]
# ORGANS = ["brain", "skin", "lung", "kidney", "breast", "liver", "colon"]
# TUMOR_CHARACTERISTICS = ["high-grade", "low-grade", "infiltrating", "benign", "malignant"]
# TREATMENTS = ["chemotherapy", "radiation therapy", "surgery", "immunotherapy"]


# def extract_metadata(text: str) -> Dict[str, Optional[str]]:
#     text_lower = text.lower()

#     def join_or_none(matches: List[str]) -> Optional[str]:
#         return ", ".join(matches) if matches else ""

#     return {
#         "cancer_types": join_or_none([c for c in CANCER_TYPES if c.lower() in text_lower]),
#         "organs_affected": join_or_none([o for o in ORGANS if o.lower() in text_lower]),
#         "tumor_characteristics": join_or_none([t for t in TUMOR_CHARACTERISTICS if t.lower() in text_lower]),
#         "treatments": join_or_none([tr for tr in TREATMENTS if tr.lower() in text_lower])
#     }


# class VectorDBFAISS:
#     def __init__(self, index_path: str = "./faiss_index.index", auto_load: bool = True):
#         self.index_path = index_path
#         self.embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
#         self.dimension = self.embedder.model.get_sentence_embedding_dimension()
#         self.index = faiss.IndexFlatIP(self.dimension)  # cosine similarity
#         self.metadata_store: List[Dict[str, Any]] = []
#         print("Initialized VectorDBFAISS with dimension:", self.dimension)

#         if auto_load and os.path.exists(self.index_path) and os.path.exists(self._metadata_path()):
#             self.load_index()
#             print("Loaded existing FAISS index and metadata")
#         else:
#             print("No existing index found, starting fresh")

#     def _metadata_path(self):
#         return self.index_path.replace(".index", "_metadata.json")

#     def add_documents(self, json_path: str, batch_size: int = 500):
#         try:
#             print("Loading documents from:", json_path)
#             with open(json_path, "r", encoding="utf-8") as f:
#                 chunks = json.load(f)
#             print("Loaded", len(chunks), "chunks from JSON")

#             texts, metadatas, ids = [], [], []
#             for chunk in chunks:
#                 if "text" in chunk and "id" in chunk:
#                     texts.append(chunk["text"])
#                     metadata = extract_metadata(chunk["text"])
#                     metadata["chunk_id"] = str(chunk["id"])
#                     metadatas.append(metadata)
#                     ids.append(str(chunk["id"]))

#             if not texts:
#                 print("No valid chunks to add")
#                 return

#             # Batch embeddings and add to FAISS
#             for start in range(0, len(texts), batch_size):
#                 batch_texts = texts[start:start+batch_size]
#                 batch_embeddings = self.embedder.embed_texts(batch_texts)
#                 self.index.add(batch_embeddings)
#                 self.metadata_store.extend(metadatas[start:start+batch_size])
#                 print(f"Added batch {start}-{start+len(batch_texts)} to FAISS")

#             self.save_index()
#             print(f"Total vectors in FAISS index: {self.index.ntotal}")

#         except Exception as e:
#             print("Error adding documents:", e)
#             traceback.print_exc()

#     def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
#         try:
#             if self.index.ntotal == 0:
#                 print("FAISS index is empty")
#                 return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

#             query_embedding = self.embedder.embed_texts([query_text])[0].reshape(1, -1)
#             distances, indices = self.index.search(query_embedding, min(n_results, self.index.ntotal))

#             results = {"ids": [], "documents": [], "distances": [], "metadatas": []}
#             for idx_list, dist_list in zip(indices, distances):
#                 ids, docs, dists, metas = [], [], [], []
#                 for i, idx in enumerate(idx_list):
#                     if idx < len(self.metadata_store):
#                         meta = self.metadata_store[idx]
#                         ids.append(meta.get("chunk_id", ""))
#                         docs.append(meta.get("text", ""))
#                         dists.append(float(dist_list[i]))
#                         metas.append(meta)
#                 results = {"ids": [ids], "documents": [docs], "distances": [dists], "metadatas": [metas]}
#             return results

#         except Exception as e:
#             print("Error querying FAISS:", e)
#             traceback.print_exc()
#             return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

#     def query_as_chunks(self, query_text: str, n_results: int = 3) -> List[RetrievedChunk]:
#         results = self.query(query_text, n_results)
#         ids = results.get("ids", [[]])[0]
#         docs = results.get("documents", [[]])[0]
#         dists = results.get("distances", [[]])[0]
#         metadatas = results.get("metadatas", [[]])[0]

#         chunks = []
#         for i, doc in enumerate(docs):
#             chunks.append(RetrievedChunk(
#                 id=str(ids[i]),
#                 text=doc,
#                 score=dists[i],  # already cosine similarity
#                 metadata=metadatas[i] if i < len(metadatas) else {}
#             ))
#         return chunks

#     def debug_info(self):
#         print("FAISS index total vectors:", self.index.ntotal)
#         if self.index.ntotal > 0:
#             print("Sample metadata:", self.metadata_store[0])

#     def save_index(self):
#         faiss.write_index(self.index, self.index_path)
#         with open(self._metadata_path(), "w", encoding="utf-8") as f:
#             json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)
#         print("FAISS index and metadata saved to disk")

#     def load_index(self):
#         self.index = faiss.read_index(self.index_path)
#         with open(self._metadata_path(), "r", encoding="utf-8") as f:
#             self.metadata_store = json.load(f)
#         print("FAISS index and metadata loaded from disk")

#     def auto_load_documents(self, search_paths=None):
#         if search_paths is None:
#             search_paths = ["./data/"]
#         json_files = []
#         for path in search_paths:
#             if os.path.exists(path):
#                 json_files.extend(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
#         for json_file in json_files:
#             self.add_documents(json_file)


# if __name__ == "__main__":
#     print("Running VectorDBFAISS standalone test")
#     db = VectorDBFAISS(auto_load=True)
#     db.debug_info()
#     # query_results = db.query_as_chunks("What is treatment for lung cancer?", n_results=3)
#     # for chunk in query_results:
#     #     print(chunk.id, chunk.score, chunk.metadata.get("cancer_types"), chunk.text[:100])
