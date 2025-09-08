import fitz
import re
import json
from nltk.tokenize import sent_tokenize
import uuid
import os

PDF_FILE = "data/Medical_book.pdf"
OUTPUT_DIR = "data/Preprocessed_Medical_Book.json"
CHUNK_SIZE = 5
MIN_CHUNK_LENGTH = 200

# Example lists for metadata extraction
CANCER_TYPES = ["lung cancer", "breast cancer", "glioma", "melanoma", "anal cancer", "kidney cancer"]
ORGANS = ["lung", "brain", "breast", "skin", "anus", "kidney"]
TREATMENTS = ["surgery", "chemotherapy", "radiotherapy", "immunotherapy", "targeted therapy"]
TUMOR_CHARACTERISTICS = ["malignant", "benign", "infiltrating", "metastatic", "high-grade", "low-grade"]

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        text += f"\n\n--- Page {page_num} ---\n\n" + page_text
    return text

def clean_text(text):
    text = text.replace('[', ' ').replace(']', ' ')
    text = text.replace('(', ' ').replace(')', ' ')
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)  # remove page numbers
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'-\s*\n\s*', '', text)  # fix hyphenated words
    text = re.sub(r'\b\w+ et al\.,?\s*\d{4}\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def extract_metadata(chunk_text):
    """Simple metadata extraction based on keyword matching."""
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

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, min_length=MIN_CHUNK_LENGTH):
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), chunk_size):
        chunk_text = " ".join(sentences[i:i+chunk_size]).strip()
        if len(chunk_text) >= min_length:
            metadata = extract_metadata(chunk_text)
            chunk = {
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": metadata
            }
            chunks.append(chunk)
    return chunks

def main():
    if not os.path.exists(PDF_FILE):
        print(f"PDF file not found: {PDF_FILE}")
        return

    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(PDF_FILE)

    print("Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("Splitting text into chunks and extracting metadata...")
    chunks = split_text_into_chunks(cleaned_text)
    print(f"Generated {len(chunks)} chunks.")

    print(f"Saving chunks to JSON: {OUTPUT_DIR}")
    with open(OUTPUT_DIR, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Done! JSON ready.")

if __name__ == "__main__":
    main()
