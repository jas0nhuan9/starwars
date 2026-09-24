# coding=utf-8
"""Confucius4-R2T2 streaming ASR on Apple Silicon, decoded by Transformers/MPS.

Upstream ships two decode paths for R2T2's streaming methods: vLLM (CUDA) and
llama.cpp (prebuilt for Linux x86_64 + CUDA only).  Neither runs on a Mac, so
this module reuses the seam ``r2t2_llama`` documents: R2T2's streaming code
only ever touches the engine through vLLM's ``generate()`` signature, so an
adapter with that signature can stand in for the engine and the streaming
behaviour -- chunk schedule, prefix rollback, append-only text -- is reused
verbatim from ``r2t2.R2T2ASRModel``.

Two upstream assumptions have to be satisfied:

* The streaming methods assert ``backend == "vllm"``, so that is what we report
  to the parent.  ``decode_backend`` records what actually decodes.
* They do ``from vllm import SamplingParams`` inside each streaming method, so
  a minimal stand-in module is installed when vLLM is absent (it is
  unbuildable on macOS).  Only the attribute access the parent performs
  (``max_tokens``) has to work.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

SAMPLE_RATE = 16_000


# --------------------------------------------------------------------------- #
# vLLM stand-in
# --------------------------------------------------------------------------- #

def install_vllm_shim() -> None:
    """Make ``from vllm import SamplingParams`` work without vLLM installed.

    R2T2's streaming methods import it lazily and only read back the fields
    they set, so a plain attribute holder is enough.  A real vLLM install is
    left untouched.
    """

    if "vllm" in sys.modules:
        return
    try:
        import vllm  # noqa: F401
        return
    except ImportError:
        pass

    class SamplingParams:
        """The handful of fields R2T2 sets on a sampling request."""

        def __init__(self, **kwargs: Any) -> None:
            self.temperature = kwargs.get("temperature", 0.0)
            self.max_tokens = kwargs.get("max_tokens", None)
            self.skip_special_tokens = kwargs.get("skip_special_tokens", True)
            for key, value in kwargs.items():
                setattr(self, key, value)

    module = types.ModuleType("vllm")
    module.SamplingParams = SamplingParams
    module.__doc__ = "Minimal stand-in installed by asr_server.r2t2_mps."
    sys.modules["vllm"] = module


# --------------------------------------------------------------------------- #
# vLLM-shaped result objects (the parent reads outputs[0].outputs[0].text)
# --------------------------------------------------------------------------- #

class _Completion:
    __slots__ = ("text", "token_ids", "finish_reason")

    def __init__(self, text: str, token_ids: Optional[List[int]] = None,
                 finish_reason: Optional[str] = None) -> None:
        self.text = text
        self.token_ids = token_ids or []
        self.finish_reason = finish_reason


class _RequestOutput:
    __slots__ = ("outputs",)

    def __init__(self, completion: _Completion) -> None:
        self.outputs = [completion]


# --------------------------------------------------------------------------- #
# engine adapter
# --------------------------------------------------------------------------- #

class TransformersEngineAdapter:
    """Presents a Transformers model through vLLM's ``generate()`` signature.

    The parent hands over one request holding the fully built prompt and the
    audio accumulated since the start of the utterance; this unpacks it into
    the processor call the Transformers backend uses and returns only the newly
    generated continuation, which is what the vLLM path also yields.
    """

    def __init__(self, model: Any, processor: Any, max_new_tokens: int = 8) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)

    def tokenize(self, text: str) -> List[int]:
        return self.processor.tokenizer.encode(text)

    def detokenize(self, tokens: List[int]) -> str:
        return self.processor.tokenizer.decode(tokens)

    @torch.inference_mode()
    def generate(self, prompts: Any, sampling_params: Any = None,
                 use_tqdm: bool = False) -> List[_RequestOutput]:
        if not prompts:
            raise ValueError("generate() requires exactly one request")
        request = prompts[0]

        prompt = request.get("prompt", "")
        audio_list = (request.get("multi_modal_data") or {}).get("audio") or []
        if not len(audio_list):
            raise ValueError("generate() requires audio in multi_modal_data")
        audio = np.asarray(audio_list[0], dtype=np.float32)

        max_tokens = getattr(sampling_params, "max_tokens", None)
        max_tokens = int(max_tokens) if max_tokens else self.max_new_tokens

        inputs = self.processor(
            text=[prompt], audio=[audio], return_tensors="pt", padding=True
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        generated = self.model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False
        )
        sequences = getattr(generated, "sequences", generated)
        prompt_len = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(
            sequences[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return [_RequestOutput(_Completion(text))]


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #

def _resolve_dtype(name: str, device: str) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    # The checkpoint is bfloat16; float16 is the faster of the two on MPS and
    # keeps ample range for this model's activations.
    return torch.float16 if device == "mps" else torch.float32


def load_r2t2(model_path: str, device: str = "mps", dtype: str = "auto",
              max_new_tokens: int = 8) -> Any:
    """Load R2T2 for streaming decode on ``device`` (default Apple GPU).

    Returns an ``r2t2.R2T2ASRModel`` whose engine is a Transformers model, so
    ``init_streaming_state`` / ``streaming_transcribe`` /
    ``finish_streaming_transcribe`` work as documented upstream.
    """

    install_vllm_shim()

    from qwen_asr.core.transformers_backend import (
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
    )
    from r2t2 import R2T2ASRModel

    path = str(Path(model_path).expanduser())
    torch_dtype = _resolve_dtype(dtype, device)

    # The processor's window size must match the checkpoint's audio config, the
    # same alignment the llama.cpp route performs, or chunk boundaries drift.
    processor = Qwen3ASRProcessor.from_pretrained(path, fix_mistral_regex=True)
    config = Qwen3ASRConfig.from_pretrained(path)
    processor.n_window = config.thinker_config.audio_config.n_window

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        path, dtype=torch_dtype
    )
    model.to(device)
    model.eval()

    adapter = TransformersEngineAdapter(
        model=model, processor=processor, max_new_tokens=max_new_tokens
    )
    asr = R2T2ASRModel(
        backend="vllm",           # what the parent's streaming asserts on
        model=adapter,
        processor=processor,
        max_new_tokens=max_new_tokens,
    )
    asr.decode_backend = f"transformers:{device}:{torch_dtype}"
    return asr
