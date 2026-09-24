#!/usr/bin/env bash
# Build r2t2_llama's pybind11 extension for macOS ARM + Metal.
#
# Upstream ships this extension prebuilt for Linux x86_64 + CUDA only, so the
# llama.cpp decode route -- the one that keeps R2T2's weights at 1.5 GB instead
# of 4.3 GB -- needs a local build.  native_ext.cpp itself is portable: it uses
# only llama.h, mtmd.h and the standard library.
#
# Two things differ from upstream's Linux instructions:
#   * GGML_METAL_EMBED_LIBRARY=ON, so the Metal shaders live inside the dylib.
#     Without it the shader library is looked up next to the executable, which
#     for a Python extension is the interpreter's directory.
#   * the rpath: upstream's CMakeLists sets the ELF-only "$ORIGIN/../bin", so
#     the Mach-O equivalent is added afterwards with install_name_tool.
#
# Idempotent: re-running reconfigures and rebuilds in place.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$ROOT/vendor/Confucius4-R2T2/r2t2_llama"
SRC="$ROOT/third_party/llama.cpp"
BUILD="$ROOT/build/r2t2_native"
# The commit r2t2_llama pins (llama.cpp b10950 / ggml 0.23.0).
PINNED="ad6c66839af3c5646fba8c6c2e2087a1e4e38948"

command -v cmake >/dev/null || { echo "cmake is required: brew install cmake" >&2; exit 1; }
"$ROOT/.venv/bin/python" -c "import pybind11" 2>/dev/null \
    || uv pip install -p "$ROOT/.venv" pybind11

if [[ ! -d "$SRC/.git" ]]; then
    echo "fetching llama.cpp at the pinned commit..."
    mkdir -p "$ROOT/third_party"
    git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp "$SRC"
fi
git -C "$SRC" fetch --depth 1 origin "$PINNED"
git -C "$SRC" checkout -q "$PINNED"
echo "llama.cpp at $(git -C "$SRC" log -1 --format='%h %ad' --date=short)"

cmake -S "$PKG" -B "$BUILD" \
    -DLLAMA_CPP_DIR="$SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$ROOT/.venv/bin/python" \
    -Dpybind11_DIR="$("$ROOT/.venv/bin/python" -m pybind11 --cmakedir)" \
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=OFF
cmake --build "$BUILD" -j "$(sysctl -n hw.perflevel0.physicalcpu)"

# The extension resolves its dylibs through @loader_path/../bin.
mkdir -p "$PKG/bin"
cp -a "$BUILD"/bin/*.dylib "$PKG/bin/"
EXT="$(ls "$PKG"/native/qwen3asr_native*.so)"
otool -l "$EXT" | grep -q "@loader_path/../bin" \
    || install_name_tool -add_rpath @loader_path/../bin "$EXT"

echo
echo "built $EXT"
PYTHONPATH="$ROOT/vendor/Confucius4-R2T2" "$ROOT/.venv/bin/python" \
    -c "from r2t2_llama.native.qwen3asr_native import Qwen3ASRNative; print('import check: ok')"
