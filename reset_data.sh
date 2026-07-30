#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$ROOT_DIR/backend/data"

mkdir -p "$DATA_DIR"

rm -f \
  "$DATA_DIR/Combined_Cancer_Chunks.json" \
  "$DATA_DIR/cancer_index_checkpoint.faiss" \
  "$DATA_DIR/cancer_chunks.pkl" \
  "$DATA_DIR/live_responses.txt" \
  "$DATA_DIR/evaluation_results.csv"

echo "Removed generated backend/data artifacts."
