import re
import traceback
import logging
from typing import List, Optional

from .knowledgebase import KnowledgeBase
from .qa_types import QAResult, RetrievedChunk
from ..embeddings import EmbeddingGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

try:
    from .. import config  # pragma: no cover
except ImportError:  # pragma: no cover
    import config

logger = logging.getLogger(__name__)

CANCER_TYPES = [
    "lung cancer", "breast cancer", "glioma", "melanoma", "anal cancer", "kidney cancer"
]
ORGANS = ["lung", "brain", "breast", "skin", "anus", "kidney"]
TREATMENTS = ["surgery", "chemotherapy", "radiotherapy", "immunotherapy", "targeted therapy"]
TUMOR_CHARACTERISTICS = ["malignant", "benign", "infiltrating", "metastatic", "high-grade", "low-grade"]

ALL_KEYWORDS = [*CANCER_TYPES, *ORGANS, *TREATMENTS, *TUMOR_CHARACTERISTICS]

DIAGNOSTIC_KEYWORDS = [
    "do i have", "am i sick", "am i at risk", "could i have",
    "is it cancer", "do i have cancer", "should i be worried", "do i have tumor"
]


def map_to_canonical(word: str) -> str:
    """Map a word to a canonical keyword if possible."""
    word = word.lower().rstrip("s")
    for kw in ALL_KEYWORDS:
        if word in kw.lower():
            return kw
    return word


class CancerQAEngine:
    def __init__(
        self,
        vectordb,
        retriever_k: int = None,
        llama_model: Optional[str] = None,
        max_new_tokens: int = 500,
        device: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        try:
            self.kb = KnowledgeBase()
            self.retriever = vectordb
            self.retriever_k = retriever_k if retriever_k is not None else config.TOP_K
            self.device = device if device is not None else 0 if torch.cuda.is_available() else -1

            model_location = checkpoint_dir or llama_model or config.LLAMA_CHECKPOINT_DIR or config.LLAMA_MODEL
            temperature = temperature if temperature is not None else config.LLAMA_TEMPERATURE

            self.llama_tokenizer = AutoTokenizer.from_pretrained(model_location)

            model_kwargs = {"device_map": "auto"} if self.device != -1 else {}

            self.llama_model = AutoModelForCausalLM.from_pretrained(model_location, **model_kwargs)

            self.llama_pipeline = pipeline(
                "text-generation",
                model=self.llama_model,
                tokenizer=self.llama_tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature
            )

            logger.info(
                f"CancerQAEngine initialized with top_k={retriever_k}, model={model_location}"
            )
        except Exception as e:
            logger.error("Failed to initialize CancerQAEngine: %s", e)
            traceback.print_exc()
            raise e

    def _extract_keywords_from_question(self, question: str) -> List[str]:
        try:
            words = re.findall(r'\b\w+\b', question.lower())
            normalized = [map_to_canonical(w) for w in words]
            return list(set(normalized))
        except Exception as e:
            logger.error("Keyword extraction failed: %s", e)
            return []

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

    def _concat_context(self, chunks: List[RetrievedChunk], limit_tokens: int = 2000) -> str:
        try:
            out = []
            total_tokens = 0

            for c in chunks:
                meta_text = []
                meta = c.metadata or {}

                for key in ["cancer_types", "organs_affected", "tumor_characteristics", "treatments"]:
                    val = meta.get(key)
                    if val:
                        if isinstance(val, str):
                            val = [val]
                        val_canonical = [map_to_canonical(v) for v in val]
                        meta_text.append(f"{key.replace('_',' ').title()}: {', '.join(val_canonical)}")

                full_chunk = "\n".join(meta_text + [c.text])
                tokens = self.llama_tokenizer(full_chunk, return_tensors="pt")["input_ids"].shape[1]

                if total_tokens + tokens > limit_tokens:
                    break

                out.append(full_chunk)
                total_tokens += tokens

            return "\n\n".join(out)
        except Exception as e:
            logger.error("Context concatenation failed: %s", e)
            traceback.print_exc()
            return ""

    def _is_diagnostic_question(self, question: str) -> bool:
        q_lower = question.lower()
        return any(keyword in q_lower for keyword in DIAGNOSTIC_KEYWORDS)

    def ask(self, question: str, biobert_model=None, rag_model=None) -> QAResult:
        """
        Hybrid QA pipeline:
        1. BioBERT (optional) - extractive
        2. RAG (optional) - generative synthesis
        3. LLaMA (compulsory) - final reasoning / fluent answer
        """
        try:
            if self._is_diagnostic_question(question):
                return QAResult(
                    answer=(
                        "I am not a medical professional and cannot provide a diagnosis. "
                        "Changes in your health, symptoms, or lab results may have multiple causes. "
                        "Please consult a qualified physician for proper evaluation."
                    ),
                    confidence=0.0,
                    method="safe-response",
                    used_chunks=[]
                )

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

            biobert_answer, biobert_score = "", 0.0
            if biobert_model:
                res = biobert_model.answer(question, context_text)
                biobert_answer, biobert_score = res.get("answer", ""), res.get("score", 0.0)

            rag_answer = ""
            if rag_model and (not biobert_answer.strip() or biobert_score < 0.8):
                rag_answer = rag_model.generate(question, chunks)

            final_context = context_text
            if biobert_answer:
                final_context += f"\n\nBioBERT Answer: {biobert_answer}"
            if rag_answer:
                final_context += f"\n\nRAG Answer: {rag_answer}"

            prompt = (
                f"You are a careful cancer-awareness assistant. Answer the question using ONLY the provided context.\n\n"
                f"Context:\n{final_context}\n\nQuestion: {question}\nAnswer:"
            )

            output = self.llama_pipeline(prompt)
            llama_answer = output[0].get("generated_text", "").replace(prompt, "").strip()

            max_sim = max([getattr(c, "score", 0.0) for c in chunks], default=0.0)
            conf = float(max_sim)

            return QAResult(
                answer=llama_answer if llama_answer else "I don't have enough information to answer that.",
                confidence=conf,
                used_chunks=chunks,
                method="BioBERT→RAG→LLaMA"
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
