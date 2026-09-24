#!/usr/bin/env bash
# Download the weights (~13 GB).  Safe to re-run: already-present files are
# skipped by the downloader.
#
# R2T2's Hugging Face checkpoint is fetched WITHOUT its safetensors: the
# default llama.cpp backend needs only the processor and tokenizer from it
# (15 MB), and the weights themselves come from the GGUF repository.  Drop the
# --exclude if you want the ASR_BACKEND=mps fallback, which loads them.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF="$ROOT/.venv/bin/hf"
cd "$ROOT"

"$HF" download netease-youdao/Confucius4-T3PO-GGUF \
    Confucius4-T3PO-Q5_K_M.gguf --local-dir models/Confucius4-T3PO-GGUF
"$HF" download netease-youdao/Confucius4-R2T2-GGUF \
    --include "Confucius4-R2T2-Q4_K_M.gguf" "mmproj-Confucius4-R2T2-Q8_0.gguf" \
    --local-dir models/Confucius4-R2T2-GGUF
"$HF" download netease-youdao/Confucius4-R2T2 \
    --local-dir models/Confucius4-R2T2 --exclude "*.safetensors"
"$HF" download FireRedTeam/FireRedVAD \
    --include "Stream-VAD/*" --local-dir models/vad

echo
du -sh models/*
