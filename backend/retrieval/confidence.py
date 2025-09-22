import numpy as np
from typing import List

class HybridConfidenceScorer:
    """
    Hybrid confidence scorer combining similarity, LLM answer quality, 
    metadata relevance, and context richness.
    """

    def __init__(
        self,
        w_sim: float = 0.4,
        w_qa: float = 0.3,
        w_meta: float = 0.2,
        w_context: float = 0.1,
        min_context_chars: int = 200
    ):
        self.w_sim = w_sim
        self.w_qa = w_qa
        self.w_meta = w_meta
        self.w_context = w_context
        self.min_context_chars = min_context_chars

    def __call__(
        self, 
        max_similarity: float, 
        qa_score: float, 
        context: str, 
        meta_scores: List[float] = None
    ) -> float:

        meta_score = np.mean(meta_scores) if meta_scores else 0.5  # default 0.5
        context_factor = 1.0 if len(context) >= self.min_context_chars else 0.85

        # Weighted sum of signals
        score = (
            self.w_sim * max_similarity +
            self.w_qa * qa_score +
            self.w_meta * meta_score +
            self.w_context * (len(context) / 2000)  # normalized context length
        )

        score *= context_factor
        return float(max(0.0, min(1.0, score)))
