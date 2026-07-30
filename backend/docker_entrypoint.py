import os
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDFS = [
    DATA_DIR / "Medical_book.pdf",
    DATA_DIR / "Encyclopedia of Cancer, 3rd Edition.pdf",
]
JSON_PATH = DATA_DIR / "Combined_Cancer_Chunks.json"
INDEX_PATH = DATA_DIR / "cancer_index_checkpoint.faiss"
CHUNKS_PATH = DATA_DIR / "cancer_chunks.pkl"


def run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def needs_bootstrap() -> bool:
    return not (INDEX_PATH.exists() and CHUNKS_PATH.exists())


def bootstrap_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_count = sum(1 for pdf in PDFS if pdf.exists())
    if pdf_count == 0:
        print("No PDFs found in backend/data; starting API without generated FAISS artifacts.", flush=True)
        return

    if not JSON_PATH.exists():
        run(["python", "-m", "backend.pdf_cleaning"])

    if not (INDEX_PATH.exists() and CHUNKS_PATH.exists()):
        run(["python", "-m", "backend.embeddings"])


def main() -> None:
    if needs_bootstrap():
        bootstrap_data()

    os.execvp(
        "uvicorn",
        [
            "uvicorn",
            "backend.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
    )


if __name__ == "__main__":
    main()
