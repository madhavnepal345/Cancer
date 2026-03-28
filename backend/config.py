import os
from dotenv import load_dotenv

# Load environment variables from a .env file at project root if present
load_dotenv()

BIOBERT_MODEL = os.getenv("BIOBERT_MODEL", "dmis-lab/biobert-base-cased-v1.1-squad")
RAG_MODEL = os.getenv("RAG_MODEL", "google/flan-t5-base")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "adityak74/medfit-llm-3B")
LLAMA_CHECKPOINT_DIR = os.getenv("LLAMA_CHECKPOINT_DIR")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")
LLAMA_TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.7"))
TOP_K = int(os.getenv("TOP_K", "5"))
