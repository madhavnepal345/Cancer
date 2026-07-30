import fitz
import re
import json
import uuid
import os
import nltk
from nltk.tokenize import sent_tokenize

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

try:
    from . import config  # pragma: no cover
except ImportError:  # pragma: no cover
    import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PDF_FILES = [
    os.path.join(DATA_DIR, "Medical_book.pdf"),
    os.path.join(DATA_DIR, "Encyclopedia of Cancer, 3rd Edition.pdf")
]
OUTPUT_FILE = os.path.join(DATA_DIR, "Combined_Cancer_Chunks.json")

CHUNK_SIZE_TOKENS = 500     
MIN_CHUNK_LENGTH = 300      
CHUNK_OVERLAP_TOKENS = 50   

CANCER_TYPES = ["lung cancer", "breast cancer", "glioma", "melanoma", "anal cancer", "kidney cancer"]
ORGANS = ["lung", "brain", "breast", "skin", "anus", "kidney"]
TREATMENTS = ["surgery", "chemotherapy", "radiotherapy", "immunotherapy", "targeted therapy"]
TUMOR_CHARACTERISTICS = ["malignant", "benign", "infiltrating", "metastatic", "high-grade", "low-grade"]


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _split_sentences(text: str):
    try:
        return sent_tokenize(text)
    except LookupError:
        # Fall back to a lightweight splitter when NLTK sentence models
        # are unavailable in the local environment.
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]


def _word_tokens(text: str):
    return re.findall(r"\b\w+\b", text)


def _token_len(text: str) -> int:
    return len(_word_tokens(text))


def _token_window_chunks(text, chunk_size_tokens, min_length, overlap_tokens, *, source, page_start, page_end, section_title, parent_id):
    words = _word_tokens(text)
    if not words:
        return []

    step = max(1, chunk_size_tokens - overlap_tokens)
    chunks = []
    window_index = 0

    for start in range(0, len(words), step):
        window_words = words[start:start + chunk_size_tokens]
        if not window_words:
            break

        chunk_text = " ".join(window_words).strip()
        if len(chunk_text) >= min_length:
            metadata = extract_metadata(chunk_text)
            metadata.update({
                "source": source,
                "page_start": page_start,
                "page_end": page_end,
                "section_title": section_title,
                "parent_id": parent_id,
                "parent_chunk_index": window_index,
                "parent_chunk_count": None,
            })
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": metadata
            })
            window_index += 1

        if start + chunk_size_tokens >= len(words):
            break

    return chunks


def _is_heading_line(line: str) -> bool:
    text = _normalize_line(line)
    if not text:
        return False

    if len(text) > 120:
        return False

    if re.match(r"^\d+(\.\d+)*\s+[A-Z0-9].*", text):
        return True

    if text.endswith(":") and len(text.split()) <= 12:
        return True

    if text.isupper() and len(text.split()) <= 12:
        return True

    if text.istitle() and len(text.split()) <= 8:
        return True

    return False


def extract_text_from_pdf(pdf_path, skip_first_page=True):
    """Extract all text from a PDF, optionally skipping the first page (publisher info, copyright, TOC)."""
    doc = fitz.open(pdf_path)
    text = ""
    start_page = 1 if skip_first_page else 0
    for page_num, page in enumerate(doc[start_page:], start=start_page + 1):
        page_text = page.get_text()
        text += f"\n\n--- Page {page_num} ---\n\n" + page_text
    return text


def extract_pages_from_pdf(pdf_path, skip_first_page=True):
    """Return a list of (page_number, page_text) tuples."""
    doc = fitz.open(pdf_path)
    pages = []
    start_page = 1 if skip_first_page else 0
    for page_num, page in enumerate(doc[start_page:], start=start_page + 1):
        pages.append((page_num, page.get_text()))
    return pages


def clean_text(text):
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


def _split_raw_text_into_sections(raw_text: str):
    sections = []
    current_title = "Untitled section"
    current_lines = []

    def flush_section():
        nonlocal current_lines
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_title, section_text))
        current_lines = []

    for raw_line in raw_text.splitlines():
        line = _normalize_line(raw_line)
        if not line:
            continue

        if _is_heading_line(line):
            flush_section()
            current_title = line.rstrip(":")
            continue

        current_lines.append(line)

    flush_section()

    if not sections:
        sections = [("Untitled section", raw_text)]

    return sections


def _make_chunk(chunk_sentences, *, source, page_start, page_end, section_title, parent_id, parent_chunk_index, parent_chunk_count):
    chunk_text = " ".join(chunk_sentences).strip()
    if len(chunk_text) < MIN_CHUNK_LENGTH:
        return None
    metadata = extract_metadata(chunk_text)
    metadata.update({
        "source": source,
        "page_start": page_start,
        "page_end": page_end,
        "section_title": section_title,
        "parent_id": parent_id,
        "parent_chunk_index": parent_chunk_index,
        "parent_chunk_count": parent_chunk_count,
    })
    return {
        "id": str(uuid.uuid4()),
        "text": chunk_text,
        "metadata": metadata
    }


def split_text_into_chunks(
    raw_text,
    chunk_size_tokens=CHUNK_SIZE_TOKENS,
    min_length=MIN_CHUNK_LENGTH,
    overlap_tokens=CHUNK_OVERLAP_TOKENS,
    *,
    source=None,
    page_start=None,
    page_end=None,
):
    chunks = []
    sections = _split_raw_text_into_sections(raw_text)

    for section_title, section_text in sections:
        section_parent_id = str(uuid.uuid4())
        cleaned_section_text = clean_text(section_text)
        if not cleaned_section_text:
            continue

        section_sentences = _split_sentences(cleaned_section_text)
        current_chunk = []
        current_len = 0
        section_chunks = []

        for sentence in section_sentences:
            sentence_token_len = _token_len(sentence)

            if sentence_token_len > chunk_size_tokens:
                if current_chunk:
                    section_chunks.append((current_chunk, section_title))
                    current_chunk = []
                    current_len = 0

                window_chunks = _token_window_chunks(
                    sentence,
                    chunk_size_tokens,
                    min_length,
                    overlap_tokens,
                    source=source,
                    page_start=page_start,
                    page_end=page_end,
                    section_title=section_title,
                    parent_id=section_parent_id,
                )
                chunks.extend(window_chunks)
                continue

            if current_len + sentence_token_len > chunk_size_tokens and current_chunk:
                section_chunks.append((current_chunk, section_title))
                if overlap_tokens > 0:
                    overlap_sentences = []
                    overlap_len = 0
                    for s in reversed(current_chunk):
                        s_len = _token_len(s)
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

            current_chunk.append(sentence)
            current_len += sentence_token_len

        if current_chunk:
            section_chunks.append((current_chunk, section_title))

        for idx, (chunk_sentences, title) in enumerate(section_chunks):
            chunk = _make_chunk(
                chunk_sentences,
                source=source,
                page_start=page_start,
                page_end=page_end,
                section_title=title,
                parent_id=section_parent_id,
                parent_chunk_index=idx,
                parent_chunk_count=len(section_chunks),
            )
            if chunk:
                chunks.append(chunk)

    return chunks


def main():
    all_chunks = []
    os.makedirs(DATA_DIR, exist_ok=True)

    for pdf_file in PDF_FILES:
        if not os.path.exists(pdf_file):
            print(f"PDF file not found: {pdf_file}")
            continue

        print(f"\nProcessing {pdf_file} ...")
        pages = extract_pages_from_pdf(pdf_file, skip_first_page=True)
        chunks = []
        for page_num, page_text in pages:
            page_chunks = split_text_into_chunks(
                page_text,
                source=os.path.basename(pdf_file),
                page_start=page_num,
                page_end=page_num,
            )
            chunks.extend(page_chunks)

        all_chunks.extend(chunks)
        print(f"Generated {len(chunks)} chunks from {pdf_file}")

    # Save all chunks to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nAll done! Total chunks: {len(all_chunks)}")
    print(f"JSON saved at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
