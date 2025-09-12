import fitz
import re
import json
import uuid
import os
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

# ------------------- CONFIGURATION ------------------- #
PDF_FILES = [
    "data/Medical_book.pdf",
    "data/Encyclopedia of Cancer, 3rd Edition.pdf"
]
OUTPUT_FILE = "data/Combined_Cancer_Chunks.json"

# Chunk settings
CHUNK_SIZE_TOKENS = 500     # Number of tokens per chunk
MIN_CHUNK_LENGTH = 300      # Minimum characters per chunk
CHUNK_OVERLAP_TOKENS = 50   # Optional overlap between chunks

# Metadata keyword lists
CANCER_TYPES = ["lung cancer", "breast cancer", "glioma", "melanoma", "anal cancer", "kidney cancer"]
ORGANS = ["lung", "brain", "breast", "skin", "anus", "kidney"]
TREATMENTS = ["surgery", "chemotherapy", "radiotherapy", "immunotherapy", "targeted therapy"]
TUMOR_CHARACTERISTICS = ["malignant", "benign", "infiltrating", "metastatic", "high-grade", "low-grade"]

# Tokenizer for token-based chunking
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad")


# ------------------- FUNCTIONS ------------------- #
def extract_text_from_pdf(pdf_path, skip_first_page=True):
    """Extract all text from a PDF, optionally skipping the first page (publisher info, copyright, TOC)."""
    doc = fitz.open(pdf_path)
    text = ""
    start_page = 1 if skip_first_page else 0
    for page_num, page in enumerate(doc[start_page:], start=start_page + 1):
        page_text = page.get_text()
        text += f"\n\n--- Page {page_num} ---\n\n" + page_text
    return text


def clean_text(text):
    """Clean extracted text to remove unwanted content and formatting."""
    # Remove brackets and parentheses
    text = text.replace('[', ' ').replace(']', ' ')
    text = text.replace('(', ' ').replace(')', ' ')

    # Remove page numbers
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)

    # Fix hyphenated words across lines
    text = re.sub(r'-\s*\n\s*', '', text)

    # Remove citations like "Author et al., 2020"
    text = re.sub(r'\b\w+ et al\.,?\s*\d{4}\b', '', text)

    # Remove sections like Abstract, References, Bibliography, Foreword, Acknowledgements
    text = re.sub(r'(?i)(abstract|references|bibliography|acknowledgements|foreword)\s*:?.*', '', text)

    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_metadata(chunk_text):
    """Extract simple metadata based on keyword matching."""
    text_lower = chunk_text.lower()
    cancer_types = [c for c in CANCER_TYPES if c in text_lower]
    organs = [o for o in ORGANS if o in text_lower]
    treatments = [t for t in TREATMENTS if t in text_lower]
    tumor_chars = [tc for tc in TUMOR_CHARACTERISTICS if tc in text_lower]
    return {
        "cancer_types": cancer_types,
        "organs_affected": organs,
        "tumor_characteristics": tumor_chars,
        "treatments": treatments
    }


def split_text_into_chunks(text, chunk_size_tokens=CHUNK_SIZE_TOKENS, min_length=MIN_CHUNK_LENGTH, overlap_tokens=CHUNK_OVERLAP_TOKENS):
    """
    Split text into chunks based on token count with optional overlap.
    Returns a list of chunk dictionaries with metadata.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        tokenized_len = len(tokenizer.encode(sentence, add_special_tokens=False))
        current_chunk.append(sentence)
        current_len += tokenized_len

        if current_len >= chunk_size_tokens:
            chunk_text = " ".join(current_chunk).strip()
            if len(chunk_text) >= min_length:
                metadata = extract_metadata(chunk_text)
                chunk = {
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "metadata": metadata
                }
                chunks.append(chunk)

            # Start new chunk with overlap
            if overlap_tokens > 0:
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    s_len = len(tokenizer.encode(s, add_special_tokens=False))
                    if overlap_len + s_len <= overlap_tokens:
                        overlap_sentences.insert(0, s)
                        overlap_len += s_len
                    else:
                        break
                current_chunk = overlap_sentences
                current_len = overlap_len
            else:
                current_chunk = []
                current_len = 0

    # Add any remaining sentences
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if len(chunk_text) >= min_length:
            metadata = extract_metadata(chunk_text)
            chunk = {
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": metadata
            }
            chunks.append(chunk)

    return chunks


# ------------------- MAIN PROCESS ------------------- #
def main():
    all_chunks = []

    for pdf_file in PDF_FILES:
        if not os.path.exists(pdf_file):
            print(f"PDF file not found: {pdf_file}")
            continue

        print(f"\nProcessing {pdf_file} ...")
        raw_text = extract_text_from_pdf(pdf_file, skip_first_page=True)
        cleaned_text = clean_text(raw_text)
        chunks = split_text_into_chunks(cleaned_text)

        # Add source info to metadata
        for chunk in chunks:
            chunk["metadata"]["source"] = os.path.basename(pdf_file)

        all_chunks.extend(chunks)
        print(f"Generated {len(chunks)} chunks from {pdf_file}")

    # Save all chunks to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nAll done! Total chunks: {len(all_chunks)}")
    print(f"JSON saved at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
