import re
import traceback
import logging
from typing import List, Tuple, Optional, Dict, Any

from .knowledgebase import KnowledgeBase
from .qa_types import QAResult, RetrievedChunk
from .biobert_qa import BioBertQA
from .confidence import ConfidenceScorer
from vector_db import VectorDBFAISS
from .retriever import Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

logger = logging.getLogger(__name__)


class CancerQAEngine:
    """
    Cancer QA Engine: combines KnowledgeBase, BioBERT, and LLaMA/Flan-T5 RAG pipeline.
    Retrieval is done via FAISS vector DB.
    """

    def __init__(
        self,
        vectordb: VectorDBFAISS,
        retriever_k: int = 5,
        biobert_model: str = "dmis-lab/biobert-base-cased-v1.1-squad",
        llama_model: str = "google/flan-t5-base",
        max_new_tokens: int = 256,
        device: Optional[int] = None,
    ):
        try:
            # Knowledge base
            self.kb = KnowledgeBase()

            # VectorDB retriever
            self.retriever = Retriever(vectordb, top_k=retriever_k)

            # BioBERT QA model
            self.biobert = BioBertQA(model_name=biobert_model)

            # Confidence scorer
            self.conf = ConfidenceScorer()

            # Device selection
            self.device = device if device is not None else -1  # CPU default

            # LLaMA / Flan-T5 RAG setup
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)
            seq2seq_keywords = ["t5", "flan", "bart", "mbart", "pegasus"]
            if any(k in llama_model.lower() for k in seq2seq_keywords):
                self.llama_model = AutoModelForSeq2SeqLM.from_pretrained(llama_model)
                self.llama_pipeline = pipeline(
                    "text2text-generation",
                    model=self.llama_model,
                    tokenizer=self.llama_tokenizer,
                    max_length=max_new_tokens,
                    do_sample=True,
                    temperature=0.7
                )
            else:
                self.llama_model = AutoModelForCausalLM.from_pretrained(llama_model)
                self.llama_pipeline = pipeline(
                    "text-generation",
                    model=self.llama_model,
                    tokenizer=self.llama_tokenizer,
                    max_length=max_new_tokens,
                    do_sample=True,
                    temperature=0.7
                )

            logger.info(f"CancerQAEngine initialized with top_k={retriever_k}")
        except Exception as e:
            logger.error("Failed to initialize CancerQAEngine: %s", e)
            traceback.print_exc()
            raise e

    # ----------------------
    # Internal helpers
    # ----------------------

    def _extract_keywords_from_question(self, question: str) -> List[str]:
        try:
            return re.findall(r'\b\w+\b', question.lower())
        except Exception as e:
            logger.error("Keyword extraction failed: %s", e)
            return []

    def _filter_chunks_by_metadata(self, chunks: List[RetrievedChunk], question: str) -> List[RetrievedChunk]:
        """
        Filter chunks based on metadata relevance to question keywords.
        """
        try:
            keywords = set(self._extract_keywords_from_question(question))
            scored_chunks = []

            for c in chunks:
                meta = c.metadata or {}
                score = 0

                # Metadata lists
                cancer_types = meta.get("cancer_types", [])
                if isinstance(cancer_types, str):
                    cancer_types = [cancer_types]
                organs_affected = meta.get("organs_affected", [])
                if isinstance(organs_affected, str):
                    organs_affected = [organs_affected]

                # Lowercase matching
                cancer_types_lower = [ct.lower() for ct in cancer_types]
                organs_lower = [o.lower() for o in organs_affected]
                text_lower = c.text.lower()

                if any(k in cancer_types_lower for k in keywords):
                    score += 3
                if any(k in organs_lower for k in keywords):
                    score += 2
                if any(k in text_lower for k in keywords):
                    score += 1

                if score > 0:
                    scored_chunks.append((score, c))

            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            filtered_chunks = [c for score, c in scored_chunks]
            return filtered_chunks if filtered_chunks else chunks

        except Exception as e:
            logger.error("Chunk filtering failed: %s", e)
            traceback.print_exc()
            return chunks

    def _concat_context(self, chunks: List[RetrievedChunk], limit_chars: int = 3500) -> str:
        try:
            out = []
            total = 0
            for c in chunks:
                meta_text = []
                meta = c.metadata or {}
                for key in ["cancer_types", "organs_affected", "tumor_characteristics", "treatments"]:
                    val = meta.get(key)
                    if val:
                        if isinstance(val, str):
                            val = [val]
                        meta_text.append(f"{key.replace('_',' ').title()}: {', '.join(val)}")
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

    def _rag_generate(self, answer: str, context: str, question: str) -> str:
        """
        RAG generation using LLaMA / Flan-T5.
        """
        try:
            prompt = (
                f"You are a careful cancer-awareness assistant. Answer the question using the context.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\nRAG Answer:"
            )
            output = self.llama_pipeline(prompt)
            generated_text = output[0].get("generated_text", "")
            generated_text = generated_text.replace(prompt, "").strip()
            return generated_text if generated_text else answer
        except Exception as e:
            logger.error("RAG generation failed: %s", e)
            traceback.print_exc()
            return answer

    # ----------------------
    # Public method
    # ----------------------

    def ask(self, question: str) -> QAResult:
        """
        Ask a question using KB, BioBERT, and RAG (LLaMA/Flan-T5) in parallel.
        Returns QAResult.
        """
        try:
            # 1. Knowledge base
            kb_ans = self.kb.maybe_answer(question)
            if kb_ans:
                return QAResult(answer=kb_ans, confidence=0.95, used_chunks=[], method="kb")

            # 2. Retrieve chunks
            chunks = self.retriever.fetch(question)
            chunks = self._filter_chunks_by_metadata(chunks, question)
            context_text = self._concat_context(chunks)
            max_sim = max([c.score for c in chunks], default=0.0)

            if not context_text.strip() or not chunks:
                return QAResult(
                    answer="I don't have enough information to answer that.",
                    confidence=0.2,
                    used_chunks=chunks,
                    method="fallback"
                )

            # 3. BioBERT answer
            qa_result = self.biobert.answer(question, context_text)
            biobert_answer = qa_result.get("answer", "")
            biobert_score = qa_result.get("score", 0.0)

            # 4. RAG answer (always parallel)
            rag_answer = self._rag_generate(biobert_answer, context_text, question)

            # 5. Confidence calculation with metadata boost
            first_meta = chunks[0].metadata or {}
            first_cancers = first_meta.get("cancer_types", [])
            if isinstance(first_cancers, str):
                first_cancers = [first_cancers]
            metadata_boost = 0.05 if any(k in [c.lower() for c in first_cancers] for k in self._extract_keywords_from_question(question)) else 0.0
            conf = self.conf(max_similarity=max_sim, qa_score=biobert_score, context=context_text) + metadata_boost

            # 6. Return combined QAResult
            return QAResult(
                answer=rag_answer,
                confidence=conf,
                used_chunks=chunks,
                method="biobert+rag",
                extra={"biobert_answer": biobert_answer, "biobert_score": biobert_score}
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
