#!/usr/bin/env bash
# Restore everything that is not in version control except the weights:
# the two upstream checkouts, the Python environment, and the native ASR
# extension.  Run scripts/fetch_models.sh for the weights.
#
# Upstream is pinned by commit rather than tracked as a submodule: these are
# read-only vendor trees, and a pin records exactly what the measurements in
# the README were taken against.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

T3PO_REPO="https://github.com/netease-youdao/Confucius4-T3PO"
T3PO_COMMIT="4827f02"
R2T2_REPO="https://github.com/netease-youdao/Confucius4-R2T2"
R2T2_COMMIT="c461192"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v llama-server >/dev/null || echo "note: llama-server missing; brew install llama.cpp" >&2

fetch() {
    local dir="$1" repo="$2" commit="$3"
    if [[ ! -d "$dir/.git" ]]; then
        echo "cloning $repo"
        git clone --filter=blob:none --no-checkout "$repo" "$dir"
    fi
    git -C "$dir" fetch --depth 1 origin "$commit"
    git -C "$dir" checkout -q "$commit"
    echo "  $dir at $(git -C "$dir" log -1 --format='%h %ad' --date=short)"
}

mkdir -p vendor
fetch vendor/Confucius4-T3PO "$T3PO_REPO" "$T3PO_COMMIT"
fetch vendor/Confucius4-R2T2 "$R2T2_REPO" "$R2T2_COMMIT"

if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12 .venv
fi
uv pip install -p .venv torch torchaudio qwen-asr librosa soundfile numpy \
    fastapi "uvicorn[standard]" websockets httpx pydantic fireredvad pybind11 \
    opencc-python-reimplemented "huggingface_hub[cli]>=0.34"

bash scripts/build_r2t2_native.sh

echo
echo "setup done. next: bash scripts/fetch_models.sh, then bash scripts/start_all.sh"
