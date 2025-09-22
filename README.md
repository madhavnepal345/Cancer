# 🩺 Cancer-Aware Q&A Engine
A medical retrieval-augmented generation (RAG) system designed to answer cancer-related questions with high reliability.
It combines:

-> Vector-based retrieval (FAISS / other DB) for context grounding

-> MEDFIT-LLM-3B (LLaMA-3B with LoRA fine-tuning on medical datasets) for medical text generation

-> Hybrid confidence scoring that considers similarity, LLM quality, metadata relevance, and context richness

 # Disclaimer: This project is for educational and research purposes only. It does not provide medical advice. Always consult healthcare professionals for clinical decisions.

 # 🚀 Features

-> Domain-Specific LLM: Uses adityak74/medfit-llm-3B, optimized for healthcare chatbots.

-> Retrieval-Augmented Generation (RAG): Fetches relevant documents before answering.

-> Hybrid Confidence Scoring: Balances similarity, metadata, context, and LLM answer quality.

-> Custom Metadata Filtering: Ensures retrieved chunks align with cancer types, organs, and treatments.

-> Quantization Support: Run with 8-bit or 4-bit quantization for CPU/low-VRAM GPUs.


# ⚙️ Installation

# Clone repo
git clone https://github.com/your-username/cancer.git

cd cancer

# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
 
# 📦 Dependencies
Key libraries used:

-> torch — PyTorch for model inference

-> transformers — Hugging Face LLaMA/MEDFIT integration

-> faiss-cpu — Vector database for retrieval

-> numpy, scikit-learn — Similarity & scoring

-> fastapi / flask — API serving

# 🔍 Confidence Scoring

The HybridConfidenceScorer computes reliability based on:

-> Cosine similarity of retrieved chunks

-> LLM answer quality

-> Metadata match (cancer type, organ, treatment)

-> Context richness (length/coverage)

Score is normalized 0.0 – 1.0.

# 📊 Roadmap

-> Add support for smaller CPU-friendly LLaMA variants (1B–2B)

-> Improve dataset curation for metadata-rich retrieval

-> Add evaluation benchmarks (BioASQ, PubMedQA)

-> Containerize with Docker for deployment

# ⚖️ Disclaimer

This system is not a substitute for medical advice. It is intended for research and educational purposes only.
Always consult qualified medical professionals before making healthcare decisions.