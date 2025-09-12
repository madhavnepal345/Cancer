import re
import traceback
import logging
from typing import List, Tuple, Optional, Dict, Any

from .knowledgebase import KnowledgeBase
from .qa_types import QAResult, RetrievedChunk
from .biobert_qa import BioBertQA
from .confidence import ConfidenceScorer
from vector_db import VectorDB
from .retriever import Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

logger = logging.getLogger(__name__)


class CancerQAEngine:
    def __init__(
        self,
        vectordb: VectorDB,
        retriever_k: int = 5,
        biobert_model: str = "dmis-lab/biobert-base-cased-v1.1-squad",
        llama_model: str = "google/flan-t5-base",
        use_fast: bool = False,
    ):
        try:
            self.kb = KnowledgeBase()
            self.retriever = Retriever(vectordb, top_k=retriever_k)
            self.biobert = BioBertQA(model_name=biobert_model)
            self.conf = ConfidenceScorer()
            self.use_fast = use_fast

            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)
            seq2seq_keywords = ["t5", "flan", "bart", "mbart", "pegasus"]
            if any(k in llama_model.lower() for k in seq2seq_keywords):
                self.llama_model = AutoModelForSeq2SeqLM.from_pretrained(
                    llama_model, device_map="auto", torch_dtype="auto"
                )
                self.llama_pipeline = pipeline(
                    "text2text-generation",
                    model=self.llama_model,
                    tokenizer=self.llama_tokenizer,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7
                )
            else:
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_model, device_map="auto", torch_dtype="auto"
                )
                self.llama_pipeline = pipeline(
                    "text-generation",
                    model=self.llama_model,
                    tokenizer=self.llama_tokenizer,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7
                )

            logger.info(f"CancerQAEngine initialized with top_k={retriever_k}")
        except Exception as e:
            logger.error("Failed to initialize CancerQAEngine: %s", e)
            traceback.print_exc()
            raise e

    def _extract_keywords_from_question(self, question: str) -> List[str]:
        try:
            return re.findall(r'\b\w+\b', question.lower())
        except Exception as e:
            logger.error("Keyword extraction failed: %s", e)
            return []

    def _filter_chunks_by_metadata(self, chunks: List[RetrievedChunk], question: str) -> List[RetrievedChunk]:
        try:
            keywords = self._extract_keywords_from_question(question)
            filtered = []
            for c in chunks:
                meta = c.metadata or {}
                text_match = any(k in c.text.lower() for k in keywords)
                cancer_types = meta.get("cancer_types", [])
                if isinstance(cancer_types, str):
                    cancer_types = [cancer_types]
                organs_affected = meta.get("organs_affected", [])
                if isinstance(organs_affected, str):
                    organs_affected = [organs_affected]

                cancer_match = any(k in [ct.lower() for ct in cancer_types] for k in keywords)
                organ_match = any(k in [o.lower() for o in organs_affected] for k in keywords)

                if text_match or cancer_match or organ_match:
                    filtered.append(c)
            return filtered if filtered else chunks
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
                        meta_text.append(f"{key.replace('_', ' ').title()}: {', '.join(val)}")

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

    def _rewrite_with_llama(self, answer: str, context: str, question: str) -> str:
        try:
            prompt = f"Rewrite the following answer to make it more clear and comprehensive based on the context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer: {answer}\n\nRewritten Answer:"
            output = self.llama_pipeline(prompt)
            rewritten_answer = output[0].get("generated_text", "")
            rewritten_answer = rewritten_answer.replace(prompt, "").strip()
            return rewritten_answer if rewritten_answer else answer
        except Exception as e:
            logger.error("LLaMA rewriting failed: %s", e)
            traceback.print_exc()
            return answer

    def ask(self, question: str, method_order: Tuple[str, ...] = ("kb", "biobert", "llama")) -> QAResult:
        try:
            logger.debug(f"Received question: {question}")

            # 1. Knowledge Base
            if "kb" in method_order:
                kb_ans = self.kb.maybe_answer(question)
                if kb_ans:
                    return QAResult(answer=kb_ans, confidence=0.95, used_chunks=[], method="kb")

            # 2. Retrieve relevant chunks
            chunks = self.retriever.fetch(question)
            logger.debug(f"Retrieved {len(chunks)} chunks")
            chunks = self._filter_chunks_by_metadata(chunks, question)
            context_text = self._concat_context(chunks)
            max_sim = max([c.score for c in chunks], default=0.0)

            # 3. BioBERT extract answer
            if "biobert" in method_order:
                if not context_text.strip() or not chunks:
                    return QAResult(
                        answer="I don't have enough information to answer that.",
                        confidence=0.2,
                        used_chunks=chunks,
                        method="fallback"
                    )

                qa_result = self.biobert.answer(question, context_text)
                biobert_answer = qa_result.get("answer", "")
                biobert_score = qa_result.get("score", 0.0)
                if not biobert_answer:
                    return QAResult(
                        answer="I don't have enough information to answer that.",
                        confidence=0.2,
                        used_chunks=chunks,
                        method="fallback"
                    )

                # Metadata boost
                
                first_chunk_cancers = chunks[0].metadata.get("cancer_types", [])
                if isinstance(first_chunk_cancers, str):
                    first_chunk_cancers = [first_chunk_cancers]
                metadata_boost = 0.05 if any(k in [c.lower() for c in first_chunk_cancers] for k in self._extract_keywords_from_question(question)) else 0.0

                conf = self.conf(max_similarity=max_sim, qa_score=biobert_score, context=context_text) + metadata_boost

                rewritten_answer = biobert_answer
                if "llama" in method_order:
                    try:
                        rewritten_answer = self._rewrite_with_llama(biobert_answer, context_text, question)
                    except Exception as e:
                        logger.error("LLaMA rewriting failed, using BioBERT answer: %s", e)

                return QAResult(
                    answer=rewritten_answer,
                    confidence=conf,
                    used_chunks=chunks,
                    method="biobert+llama" if "llama" in method_order else "biobert",
                    extra={"qa_score": biobert_score}
                )

            # Fallback
            return QAResult(
                answer="I don't have enough information to answer that.",
                confidence=0.2,
                used_chunks=chunks,
                method="fallback"
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
