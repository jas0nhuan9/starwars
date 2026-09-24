# coding=utf-8
"""Confucius4-R2T2 streaming ASR decoded by llama.cpp on Metal.

Upstream's ``r2t2_llama`` package already implements this route -- it is their
recommended one for llama.cpp deployments -- but ships the pybind11 extension
prebuilt for Linux x86_64 + CUDA only.  Rebuilt for macOS ARM + Metal
(``scripts/build_r2t2_native.sh``), it runs unmodified, so this module is only
a loader with Mac-appropriate defaults.

Why prefer it over the Transformers/MPS route in :mod:`asr_server.r2t2_mps`:

* memory: 1.5 GB of GGUF weights instead of 4.3 GB of fp16 tensors, which is
  what makes room for the 14B translator on a 24 GB machine;
* cost: llama.cpp keeps the KV cache across chunks, so a chunk costs the same
  at the end of a long utterance as at its start.  The Transformers route has
  no prefix cache and re-prefills the accumulated audio every chunk.

The trade-off upstream documents is accuracy: their ``stream_llama_hybrid``
mode exists because the llama.cpp mel/encoder path is less robust on quiet
audio onsets than the PyTorch encoder.  ``scripts/bench_asr.py --backend``
compares the two on this machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from asr_server.r2t2_mps import install_vllm_shim

# Streaming utterances are capped at a few seconds of audio, so the huge
# upstream defaults (32k context, 8k batch) only waste memory here.
DEFAULT_N_CTX = 4096
DEFAULT_N_BATCH = 2048
DEFAULT_N_THREADS = 10


def load_r2t2_gguf(
    gguf_dir: str,
    processor_path: str,
    *,
    model_gguf_name: Optional[str] = None,
    mmproj_gguf_name: Optional[str] = None,
    n_ctx: int = DEFAULT_N_CTX,
    n_batch: int = DEFAULT_N_BATCH,
    n_threads: int = DEFAULT_N_THREADS,
    max_new_tokens: int = 8,
) -> Any:
    """Load R2T2 for streaming decode through llama.cpp with Metal offload.

    ``processor_path`` is still the Hugging Face checkpoint: the llama.cpp
    routes deliberately keep tokenization and feature extraction on the HF
    processor so prompts and token boundaries match the vLLM route exactly.
    Only the weights come from ``gguf_dir``.
    """

    # R2T2's streaming methods do `from vllm import SamplingParams` internally.
    install_vllm_shim()

    from r2t2_llama import R2T2LlamaASRModel

    gguf_root = Path(gguf_dir).expanduser()
    if not gguf_root.is_dir():
        raise FileNotFoundError(f"GGUF directory not found: {gguf_root}")

    asr = R2T2LlamaASRModel.LlamaNative(
        processor_path=str(Path(processor_path).expanduser()),
        gguf_dir=str(gguf_root),
        model_gguf_name=model_gguf_name,
        mmproj_gguf_name=mmproj_gguf_name,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_threads=n_threads,
        use_gpu=True,
        n_gpu_layers=-1,      # every layer on the Apple GPU
        max_new_tokens=max_new_tokens,
    )
    asr.decode_backend = (
        f"llama.cpp:metal:{model_gguf_name or 'auto'}+{mmproj_gguf_name or 'auto'}"
    )
    return asr
