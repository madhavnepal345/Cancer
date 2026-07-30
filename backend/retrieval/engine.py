import re
import traceback
import logging
import os
from collections import defaultdict
from typing import List, Optional

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

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

INTERVENTION_TERMS = [
    "treatment", "therapy", "surgery", "procedure", "medication", "medicine",
    "drug", "chemotherapy", "radiotherapy", "radiation", "immunotherapy",
    "targeted therapy", "hormone therapy", "endocrine therapy",
]
ADVERSE_QUERY_TERMS = [
    "risk", "risks", "side effect", "side effects", "complication",
    "complications", "adverse effect", "adverse effects", "toxicity",
    "toxicities", "harm", "danger",
]
ADVERSE_EVIDENCE_TERMS = [
    "side effect", "complication", "adverse", "toxicity", "toxic",
    "lymphedema", "cardiotoxicity", "infection", "bleeding", "fibrosis",
    "pneumonitis", "secondary cancer", "second primary",
]
INTENT_EXPANSIONS = {
    "symptoms": {
        "triggers": ["symptom", "symptoms", "sign", "signs", "presentation"],
        "terms": ["symptoms", "signs", "clinical presentation", "manifestations"],
    },
    "diagnosis": {
        "triggers": ["diagnosis", "diagnose", "diagnosed", "test", "tests", "screening"],
        "terms": ["diagnosis", "diagnostic tests", "screening", "evaluation"],
    },
    "treatment": {
        "triggers": INTERVENTION_TERMS,
        "terms": ["treatment", "management", "therapy", "surgery", "medication"],
    },
    "prevention": {
        "triggers": ["prevent", "prevention", "reduce the risk", "protect"],
        "terms": ["prevention", "risk reduction", "preventive measures"],
    },
    "prognosis": {
        "triggers": ["prognosis", "outlook", "survival", "recovery", "outcome"],
        "terms": ["prognosis", "outcomes", "survival", "recurrence"],
    },
    "risk_factors": {
        "triggers": ["risk factor", "risk factors", "risk of developing", "cause", "causes"],
        "terms": ["risk factors", "predisposition", "causes", "likelihood of developing"],
    },
}
QUERY_STOPWORDS = {
    "a", "after", "an", "and", "are", "be", "can", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "the", "this", "to",
    "what", "when", "where", "which", "who", "why", "with",
}
GENERIC_INTENT_WORDS = {
    "adverse", "cancer", "cause", "causes", "clinical", "complication",
    "complications", "danger", "developing", "diagnose", "diagnosed",
    "diagnosis", "diagnostic", "disease", "effect", "effects", "evaluation",
    "factor", "factors", "harm", "likelihood", "management", "manifestations",
    "chemotherapy", "drug", "endocrine", "hormone", "immunotherapy",
    "medication", "medicine", "occur", "occurs", "outcome", "outcomes",
    "outlook", "predisposition",
    "presentation", "prevent", "prevention", "preventive", "procedure",
    "prognosis", "protect", "recovery", "recurrence", "reduce", "reduction",
    "radiation", "radiotherapy", "risk", "risks", "screening", "side", "sign",
    "signs", "surgery", "survival", "targeted",
    "symptom", "symptoms", "test", "tests", "therapy", "toxic", "toxicities",
    "toxicity", "treatment",
}

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
        max_new_tokens: int = 180,
        device: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        try:
            self.kb = KnowledgeBase()
            self.retriever = vectordb
            self.retriever_k = retriever_k if retriever_k is not None else config.TOP_K
            self.device = device if device is not None else 0 if torch.cuda.is_available() else -1
            self.llama_pipeline = None

            model_location = checkpoint_dir or llama_model or config.LLAMA_CHECKPOINT_DIR or config.LLAMA_MODEL
            temperature = temperature if temperature is not None else config.LLAMA_TEMPERATURE

            try:
                self.llama_tokenizer = AutoTokenizer.from_pretrained(model_location)
                model_kwargs = {"device_map": "auto"} if self.device != -1 else {}
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    model_location,
                    **model_kwargs,
                )
                self.llama_pipeline = pipeline(
                    "text-generation",
                    model=self.llama_model,
                    tokenizer=self.llama_tokenizer,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=temperature
                )
            except Exception as model_error:
                logger.warning("LLM unavailable, using context-only fallback: %s", model_error)
                self.llama_tokenizer = None
                self.llama_model = None

            logger.info(
                f"CancerQAEngine initialized with top_k={retriever_k}, model={model_location}"
            )
        except Exception as e:
            logger.error("Failed to initialize CancerQAEngine: %s", e)
            traceback.print_exc()
            raise e

    def _extract_keywords_from_question(self, question: str) -> List[str]:
        try:
            question_lower = question.lower()
            return list({
                keyword for keyword in ALL_KEYWORDS
                if keyword in question_lower
            })
        except Exception as e:
            logger.error("Keyword extraction failed: %s", e)
            return []

    def _is_adverse_effect_question(self, question: str) -> bool:
        question_lower = question.lower()
        has_intervention = any(term in question_lower for term in INTERVENTION_TERMS)
        has_adverse_term = any(term in question_lower for term in ADVERSE_QUERY_TERMS)
        has_explicit_adverse_term = any(
            term in question_lower
            for term in ADVERSE_QUERY_TERMS
            if term not in {"risk", "risks"}
        )
        return has_explicit_adverse_term or (has_intervention and has_adverse_term)

    def _matched_intents(self, question: str) -> List[str]:
        question_lower = question.lower()
        intents = [
            name for name, config in INTENT_EXPANSIONS.items()
            if any(trigger in question_lower for trigger in config["triggers"])
        ]
        if self._is_adverse_effect_question(question):
            intents.append("adverse_effects")
        return list(dict.fromkeys(intents))

    def _query_scope_terms(self, question: str) -> set:
        question_terms = {
            term for term in re.findall(r"\b[\w-]+\b", question.lower())
            if len(term) > 2 and term not in QUERY_STOPWORDS
        }
        return question_terms - GENERIC_INTENT_WORDS

    def _expand_retrieval_query(self, question: str) -> str:
        intents = self._matched_intents(question)
        if not intents:
            return question

        expansion_terms = []
        for intent in intents:
            if intent == "adverse_effects":
                expansion_terms.extend([
                    "side effects", "complications", "adverse effects", "toxicity",
                    "treatment-related harm",
                ])
            else:
                expansion_terms.extend(INTENT_EXPANSIONS[intent]["terms"])

        scope_terms = sorted(self._query_scope_terms(question))
        named_interventions = [
            term for term in INTERVENTION_TERMS
            if term in question.lower()
        ]
        focus_terms = [*scope_terms, *named_interventions, *expansion_terms]
        return " ".join(dict.fromkeys(focus_terms))

    def _retrieve_candidates(self, question: str) -> List[RetrievedChunk]:
        expanded_query = self._expand_retrieval_query(question)
        candidate_k = max(self.retriever_k, 20) if expanded_query != question else self.retriever_k
        queries = [question]
        if expanded_query != question:
            queries.append(expanded_query)
        if self._is_adverse_effect_question(question):
            named_interventions = [
                term for term in INTERVENTION_TERMS
                if term in question.lower()
            ]
            if named_interventions:
                queries.append(
                    " ".join([
                        *named_interventions,
                        "side effects complications adverse effects toxicity",
                    ])
                )

        merged = {}
        for query in queries:
            for chunk in self.retriever.fetch(query, top_k=candidate_k):
                existing = merged.get(chunk.id)
                if existing is None or chunk.score > existing.score:
                    merged[chunk.id] = chunk

        return sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)

    def _filter_chunks_by_metadata(self, chunks: List[RetrievedChunk], question: str) -> List[RetrievedChunk]:
        try:
            question_lower = question.lower()
            question_terms = {
                term for term in re.findall(r"\b[\w-]+\b", question_lower)
                if len(term) > 2 and term not in QUERY_STOPWORDS
            }
            scope_terms = self._query_scope_terms(question)
            matched_intents = self._matched_intents(question)
            adverse_effect_question = "adverse_effects" in matched_intents
            scored_chunks = []

            for c in chunks:
                meta = c.metadata or {}
                score = float(c.score)
                text_lower = c.text.lower()
                section_lower = str(meta.get("section_title") or "").lower()
                searchable_text = f"{section_lower} {text_lower}"

                metadata_values = []
                for key in ("cancer_types", "organs_affected", "treatments"):
                    values = meta.get(key) or []
                    if isinstance(values, str):
                        values = [values]
                    metadata_values.extend(str(value).lower() for value in values)

                score += sum(
                    value in question_lower for value in set(metadata_values)
                ) * 4
                lexical_overlap = sum(term in searchable_text for term in question_terms)
                score += min(lexical_overlap, 8) * 0.5
                scope_overlap = sum(term in searchable_text for term in scope_terms)
                score += scope_overlap * 4
                if scope_terms and scope_overlap == 0:
                    score -= 6

                for intent in matched_intents:
                    if intent == "adverse_effects":
                        continue
                    intent_terms = INTENT_EXPANSIONS[intent]["terms"]
                    score += min(
                        sum(term in searchable_text for term in intent_terms),
                        3,
                    )

                if adverse_effect_question:
                    evidence_matches = sum(
                        term in searchable_text for term in ADVERSE_EVIDENCE_TERMS
                    )
                    treatment_matches = sum(
                        term in searchable_text for term in INTERVENTION_TERMS
                    )
                    score += min(evidence_matches, 4) * 3
                    score += min(treatment_matches, 3)
                    if evidence_matches == 0:
                        score -= 8

                    describes_disease_risk = (
                        "risk factor" in text_lower or "risk factor" in section_lower
                    )
                    if describes_disease_risk and evidence_matches == 0:
                        score -= 8

                scored_chunks.append((score, c))

            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            filtered_chunks = [c for score, c in scored_chunks]

            result_limit = 5 if adverse_effect_question else 3
            return filtered_chunks[:result_limit] if filtered_chunks else chunks[:result_limit]
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
                tokens = len(re.findall(r"\b\w+\b", full_chunk))

                if total_tokens + tokens > limit_tokens:
                    break

                out.append(full_chunk)
                total_tokens += tokens

            return "\n\n".join(out)
        except Exception as e:
            logger.error("Context concatenation failed: %s", e)
            traceback.print_exc()
            return ""

    def _context_only_answer(self, question: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "I don't have enough information to answer that."

        best = chunks[0]
        snippet = best.text.strip()
        if len(snippet) > 500:
            snippet = snippet[:500].rsplit(" ", 1)[0] + "..."

        return (
            "I don't have a local language model available, but the most relevant source excerpt is:\n\n"
            f"{snippet}"
        )

    def _expand_hierarchical_chunks(self, chunks: List[RetrievedChunk], sibling_span: int = 1) -> List[RetrievedChunk]:
        try:
            if not chunks:
                return []

            grouped = defaultdict(list)
            for chunk in self.retriever.chunks:
                parent_id = (chunk.get("metadata") or {}).get("parent_id")
                if parent_id:
                    grouped[parent_id].append(chunk)

            for parent_id in grouped:
                grouped[parent_id].sort(
                    key=lambda c: (
                        (c.get("metadata") or {}).get("parent_chunk_index") is None,
                        (c.get("metadata") or {}).get("parent_chunk_index", 0),
                    )
                )

            expanded = []
            seen_ids = set()
            for chunk in chunks:
                if chunk.id not in seen_ids:
                    expanded.append(chunk)
                    seen_ids.add(chunk.id)

                parent_id = (chunk.metadata or {}).get("parent_id")
                if not parent_id or parent_id not in grouped:
                    continue

                siblings = grouped[parent_id]
                try:
                    current_index = next(
                        i for i, sibling in enumerate(siblings) if sibling.get("id") == chunk.id
                    )
                except StopIteration:
                    continue

                start = max(0, current_index - sibling_span)
                end = min(len(siblings), current_index + sibling_span + 1)
                for sibling in siblings[start:end]:
                    sibling_id = sibling.get("id")
                    if sibling_id in seen_ids:
                        continue
                    expanded.append(
                        RetrievedChunk(
                            id=sibling_id,
                            text=sibling.get("text", ""),
                            score=float(chunk.score),
                            metadata=sibling.get("metadata", {}),
                        )
                    )
                    seen_ids.add(sibling_id)

            return expanded
        except Exception as e:
            logger.error("Hierarchical expansion failed: %s", e)
            traceback.print_exc()
            return chunks

    def _response_guidance(self, question: str) -> str:
        question_lower = question.lower().strip()
        intents = self._matched_intents(question)

        asks_for_list = (
            "symptoms" in intents
            or "risk_factors" in intents
            or "adverse_effects" in intents
            or any(
                term in question_lower
                for term in (" options", " types", " stages", " causes")
            )
        )
        if asks_for_list:
            return (
                "Provide a complete, concise list of the relevant items supported "
                "by the excerpts."
            )
        if re.match(r"^(what\s+(?:is|are)|define|explain)\b", question_lower):
            return (
                "For this definition question, explain what it is, how it works, "
                "and its purpose when those details are supported."
            )
        if re.match(r"^(can|could|is|are|does|do|should|will)\b", question_lower):
            return (
                "For this yes-or-no question, begin with a qualified answer, then "
                "give the supporting conditions, limitations, or examples."
            )
        if "diagnosis" in intents:
            return (
                "Explain the supported diagnostic tests or process in a clear sequence."
            )
        if "treatment" in intents:
            return (
                "Explain the treatment's role, how it is used, and important supported "
                "limitations or risks."
            )
        if "prevention" in intents or "prognosis" in intents:
            return (
                "Give a qualified explanation and include the important supported "
                "conditions or limitations."
            )
        return "Give enough supported detail to answer the question completely."

    def _build_llama_prompt(self, question: str, final_context: str) -> str:
        response_guidance = self._response_guidance(question)
        return (
            "You are a cancer-awareness assistant.\n"
            "Answer using only the provided source excerpts.\n"
            "Do not use outside knowledge, do not infer missing facts, and do not diagnose.\n"
            "If the answer is not explicitly supported by the source excerpts, reply exactly:\n"
            "\"I don't have enough information in the provided materials.\"\n"
            "Be concise, factual, and complete; do not reduce a supported explanation to a fragment.\n"
            "Use 2-4 sentences for explanations or a concise list when the question asks for items.\n"
            f"{response_guidance}\n"
            "Answer exactly one question, then stop immediately.\n"
            "Do not generate follow-up questions, Question/Answer labels, or notes about these instructions.\n"
            "Do not mention chain-of-thought, hidden reasoning, or uncertainty beyond the required fallback.\n"
            "Return only the answer text.\n\n"
            f"Context:\n{final_context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    @staticmethod
    def _clean_generated_answer(answer: str) -> str:
        if not answer:
            return ""

        cleaned = answer.strip()
        cleaned = re.sub(r"^\s*Answer\s*:\s*", "", cleaned, flags=re.IGNORECASE)

        stop_patterns = [
            r"\n\s*(?:Question|Q)\s*:",
            r"\n\s*Context\s*:",
            r"\n\s*Note\s*:\s*(?:the\s+)?(?:provided\s+)?answer\b",
        ]
        stop_positions = []
        for pattern in stop_patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                stop_positions.append(match.start())

        if stop_positions:
            cleaned = cleaned[:min(stop_positions)]

        return cleaned.strip()

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
                    used_chunks=[],
                    extra={
                        "retrieval_similarity": 0.0,
                        "biobert_score": 0.0,
                        "rag_used": False,
                    },
                )

            kb_ans = self.kb.maybe_answer(question)
            if kb_ans:
                return QAResult(
                    answer=kb_ans,
                    confidence=0.95,
                    used_chunks=[],
                    method="kb",
                    extra={
                        "retrieval_similarity": 0.0,
                        "biobert_score": 0.0,
                        "rag_used": False,
                    },
                )

            chunks = self._retrieve_candidates(question)
            if not chunks:
                return QAResult(
                    answer="I don't have enough information to answer that.",
                    confidence=0.2,
                    used_chunks=[],
                    method="fallback",
                    extra={
                        "retrieval_similarity": 0.0,
                        "biobert_score": 0.0,
                        "rag_used": False,
                    },
                )

            chunks = self._filter_chunks_by_metadata(chunks, question)
            chunks = self._expand_hierarchical_chunks(chunks)
            context_text = self._concat_context(chunks)

            if not context_text.strip():
                return QAResult(
                    answer="I don't have enough information to answer that.",
                    confidence=0.2,
                    used_chunks=chunks,
                    method="fallback",
                    extra={
                        "retrieval_similarity": max([getattr(c, "score", 0.0) for c in chunks], default=0.0),
                        "biobert_score": 0.0,
                        "rag_used": False,
                    },
                )

            biobert_answer, biobert_score = "", 0.0
            if biobert_model:
                res = biobert_model.answer(question, context_text)
                biobert_answer, biobert_score = res.get("answer", ""), res.get("score", 0.0)

            rag_answer = ""
            if rag_model and (not biobert_answer.strip() or biobert_score < 0.8):
                rag_answer = rag_model.generate(question, chunks)

            # Keep the final LLaMA prompt grounded only in retrieved source text.
            # BioBERT and RAG outputs can still inform confidence, but they are not
            # fed back into the final prompt to avoid self-referential amplification.
            final_context = context_text

            prompt = self._build_llama_prompt(question, final_context)

            if self.llama_pipeline is not None:
                output = self.llama_pipeline(prompt, return_full_text=False)
                generated_text = output[0].get("generated_text", "")
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):]
                llama_answer = self._clean_generated_answer(generated_text)
            else:
                llama_answer = self._context_only_answer(question, chunks)

            max_sim = max([getattr(c, "score", 0.0) for c in chunks], default=0.0)
            if biobert_answer.strip():
                conf = min(1.0, (0.6 * float(max_sim)) + (0.4 * float(biobert_score)))
            else:
                conf = float(max_sim)

            if rag_answer:
                conf = min(1.0, conf + 0.05)

            return QAResult(
                answer=llama_answer if llama_answer else "I don't have enough information to answer that.",
                confidence=conf,
                used_chunks=chunks,
                method="BioBERT→RAG→LLaMA",
                extra={
                    "retrieval_similarity": float(max_sim),
                    "biobert_score": float(biobert_score),
                    "rag_used": bool(rag_answer),
                },
            )

        except Exception as e:
            logger.error("QA engine failed: %s", e)
            traceback.print_exc()
            return QAResult(
                answer="An error occurred while processing your question.",
                confidence=0.0,
                used_chunks=[],
                method="error",
                extra={
                    "retrieval_similarity": 0.0,
                    "biobert_score": 0.0,
                    "rag_used": False,
                },
            )
