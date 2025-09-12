from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from retrieval.retriever import RetrievedChunk


class RAG:
    def __init__(self, model_name: str = "google/flan-t5-base", device: Optional[int] = None, max_new_tokens: int = 256):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device if device is not None else -1  # CPU default

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Decide which class to use based on model name
        if any(x in model_name.lower() for x in ["t5", "flan", "bart", "mbart", "pegasus"]):
            # Seq2Seq models (encoder-decoder)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.kind = "seq2seq"
        else:
            # Decoder-only models (causal)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.kind = "causal"

    def _build_prompt(self, question: str, context_blocks: List[RetrievedChunk]) -> str:
        context_text = "\n\n".join([f"[Chunk {c.id} | sim={c.score:.2f}]\n{c.text}" for c in context_blocks])
        instructions = (
            "You are a careful cancer-awareness assistant. Answer the question using ONLY the provided context. "
            "If the answer is not present, say 'I don't have enough information in the provided materials.' "
            "Be concise, factual, and non-diagnostic."
        )
        return f"{instructions}\n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"

    def generate(self, question: str, context_blocks: List[RetrievedChunk]) -> str:
        prompt = self._build_prompt(question, context_blocks)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if self.kind == "causal" and "Answer:" in text:
            return text.split("Answer:")[-1].strip()
        return text
