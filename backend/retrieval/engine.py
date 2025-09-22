import re
import traceback
import logging
from typing import List, Optional

from .knowledgebase import KnowledgeBase
from .qa_types import QAResult, RetrievedChunk
from embeddings import EmbeddingGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

logger = logging.getLogger(__name__)

CANCER_TYPES = ["lung cancer", "breast cancer", "glioma", "melanoma", "anal cancer", "kidney cancer"]
ORGANS = ["lung", "brain", "breast", "skin", "anus", "kidney"]
TREATMENTS = ["surgery", "chemotherapy", "radiotherapy", "immunotherapy", "targeted therapy"]
TUMOR_CHARACTERISTICS = ["malignant", "benign", "infiltrating", "metastatic", "high-grade", "low-grade"]

ALL_KEYWORDS = [*CANCER_TYPES, *ORGANS, *TREATMENTS, *TUMOR_CHARACTERISTICS]

def map_to_canonical(word: str) -> str:
    word = word.lower().rstrip("s")  
    for kw in ALL_KEYWORDS:
        if word in kw.lower():  
            return kw
    return word

class CancerQAEngine:
    def __init__(
        self,
        vectordb,
        retriever_k: int = 5,
        llama_model: str = "adityak74/medfit-llm-3B",
        max_new_tokens: int = 500,
        device: Optional[int] = None,
        quantization: Optional[str] = "8bit",
        checkpoint_dir: Optional[str] = None,
    ):
        try:
            # Knowledge base
            self.kb = KnowledgeBase()

            # Retriever
            self.retriever = vectordb
            self.retriever_k = retriever_k

            # Device
            self.device = device if device is not None else 0 if torch.cuda.is_available() else -1

            # Load tokenizer
            if checkpoint_dir:
                self.llama_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            else:
                self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)

            # Load model with optional quantization
            model_kwargs = {"device_map": "auto"} if self.device != -1 else {}

            if quantization == "8bit":
                model_kwargs["load_in_8bit"] = True
            elif quantization == "4bit":
                model_kwargs["load_in_4bit"] = True

            if checkpoint_dir:
                self.llama_model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, **model_kwargs)
            else:
                self.llama_model = AutoModelForCausalLM.from_pretrained(llama_model, **model_kwargs)

            # Pipeline
            self.llama_pipeline = pipeline(
                "text-generation",
                model=self.llama_model,
                tokenizer=self.llama_tokenizer,
                max_length=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )

            logger.info(f"CancerQAEngine initialized with top_k={retriever_k}, model={llama_model}, quant={quantization}")
        except Exception as e:
            logger.error("Failed to initialize CancerQAEngine: %s", e)
            traceback.print_exc()
            raise e

    # --- Keyword extraction ---
    def _extract_keywords_from_question(self, question: str) -> List[str]:
        try:
            words = re.findall(r'\b\w+\b', question.lower())
            normalized = [map_to_canonical(w) for w in words]
            return list(set(normalized))
        except Exception as e:
            logger.error("Keyword extraction failed: %s", e)
            return []

    # --- Filter chunks using metadata ---
    def _filter_chunks_by_metadata(self, chunks: List[RetrievedChunk], question: str) -> List[RetrievedChunk]:
        try:
            keywords = set(self._extract_keywords_from_question(question))
            scored_chunks = []

            for c in chunks:
                meta = c.metadata or {}
                score = 0

                cancer_types = meta.get("cancer_types") or []
                organs_affected = meta.get("organs_affected") or []

                if isinstance(cancer_types, str):
                    cancer_types = [cancer_types]
                if isinstance(organs_affected, str):
                    organs_affected = [organs_affected]

                cancer_types_lower = [map_to_canonical(ct) for ct in cancer_types]
                organs_lower = [map_to_canonical(o) for o in organs_affected]
                text_lower = c.text.lower()

                if any(k in cancer_types_lower for k in keywords):
                    score += 5
                if any(k in organs_lower for k in keywords):
                    score += 4
                if any(k in text_lower for k in keywords):
                    score += 2

                if score > 0:
                    scored_chunks.append((score, c))

            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            filtered_chunks = [c for score, c in scored_chunks]

            return filtered_chunks[:3] if filtered_chunks else chunks[:3]
        except Exception as e:
            logger.error("Chunk filtering failed: %s", e)
            traceback.print_exc()
            return chunks[:3]

    # --- Concatenate context ---
    def _concat_context(self, chunks: List[RetrievedChunk], limit_chars: int = 2000) -> str:
        try:
            out = []
            total = 0

            for c in chunks:
                meta_text = []
                meta = c.metadata or {}

                for key, vocab_list in [
                    ("cancer_types", CANCER_TYPES),
                    ("organs_affected", ORGANS),
                    ("tumor_characteristics", TUMOR_CHARACTERISTICS),
                    ("treatments", TREATMENTS)
                ]:
                    val = meta.get(key)
                    if val:
                        if isinstance(val, str):
                            val = [val]
                        val_canonical = [map_to_canonical(v) for v in val]
                        meta_text.append(f"{key.replace('_',' ').title()}: {', '.join(val_canonical)}")

                full_chunk = "\n".join(meta_text + [c.text])
                if total + len(full_chunk) > limit_chars:
                    remaining = max(0, limit_chars - total)
                    if remaining > 0:
                        out.append(full_chunk[:remaining])
                    break

                out.append(full_chunk)
                total += len(full_chunk)

            return "\n\n".join(out)
        except Exception as e:
            logger.error("Context concatenation failed: %s", e)
            traceback.print_exc()
            return ""

    # --- Main ask method ---
    def ask(self, question: str) -> QAResult:
        try:
            kb_ans = self.kb.maybe_answer(question)
            if kb_ans:
                return QAResult(answer=kb_ans, confidence=0.95, used_chunks=[], method="kb")

            chunks = self.retriever.fetch(question, top_k=self.retriever_k)
            if not chunks:
                return QAResult(
                    answer="I don't have enough information to answer that.",
                    confidence=0.2,
                    used_chunks=[],
                    method="fallback"
                )

            chunks = self._filter_chunks_by_metadata(chunks, question)
            context_text = self._concat_context(chunks)

            if not context_text.strip():
                return QAResult(
                    answer="I don't have enough information to answer that.",
                    confidence=0.2,
                    used_chunks=chunks,
                    method="fallback"
                )

            prompt = (
                f"You are a careful cancer-awareness assistant. Answer the question using ONLY the provided context.\n\n"
                f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
            )

            output = self.llama_pipeline(prompt)
            llama_answer = output[0].get("generated_text", "").replace(prompt, "").strip()

            max_sim = max([getattr(c, "score", 0.0) for c in chunks], default=0.0)
            conf = float(max_sim)

            return QAResult(
                answer=llama_answer if llama_answer else "I don't have enough information to answer that.",
                confidence=conf,
                used_chunks=chunks,
                method="medfit_llm_3b_rag"
            )
        except Exception as e:
            logger.error("QA engine failed: %s", e)
            traceback.print_exc()
            return QAResult(
                answer="An error occurred while processing your question.",
                confidence=0.0,
                used_chunks=[],
                method="error"
            )
