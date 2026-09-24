# coding=utf-8
"""Measure R2T2 streaming cost per chunk on this Mac.

Real-time simultaneous interpretation only works if one 160 ms chunk decodes in
well under 160 ms.  The vLLM path gets that from prefix caching; this prints
what the Transformers/MPS path actually costs, and how the cost grows as the
utterance's accumulated audio gets longer.
"""

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "Confucius4-R2T2"))

from asr_server.r2t2_gguf import load_r2t2_gguf  # noqa: E402
from asr_server.r2t2_mps import load_r2t2  # noqa: E402

SR = 16_000


def split_text_to_tokens(text: str) -> list:
    """Chinese by character, English by word, punctuation dropped (upstream)."""
    tokens = []
    for match in re.finditer(r"[a-zA-Z]+|[^a-zA-Z]", text):
        token = match.group()
        if token.strip() and not re.match(r"[^\w]", token):
            tokens.append(token)
    return tokens


def is_last_token_chinese(tokens: list) -> bool:
    if not tokens:
        return False
    return any("一" <= ch <= "鿿" for ch in tokens[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=str(ROOT / "models" / "Confucius4-R2T2"))
    parser.add_argument("--audio", default=str(
        ROOT / "vendor" / "Confucius4-R2T2" / "resources" / "test.wav"))
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--backend", default="gguf", choices=("gguf", "mps"),
                        help="gguf = llama.cpp/Metal, mps = Transformers/MPS")
    parser.add_argument("--gguf_dir", default=str(ROOT / "models" / "Confucius4-R2T2-GGUF"))
    parser.add_argument("--mmproj", default="mmproj-Confucius4-R2T2-Q8_0.gguf")
    parser.add_argument("--model_gguf", default="Confucius4-R2T2-Q4_K_M.gguf")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--chunk_ms", type=int, default=160)
    parser.add_argument("--lookahead_ms", type=int, default=160)
    parser.add_argument("--max_chunks", type=int, default=0,
                        help="stop early; 0 runs the whole file")
    args = parser.parse_args()

    wav, sr = librosa.load(args.audio, sr=SR, mono=True)
    print(f"audio: {args.audio}  {wav.shape[0] / SR:.2f}s @ {SR}Hz", flush=True)

    t0 = time.time()
    if args.backend == "gguf":
        asr = load_r2t2_gguf(args.gguf_dir, args.model_path,
                             model_gguf_name=args.model_gguf,
                             mmproj_gguf_name=args.mmproj, max_new_tokens=8)
    else:
        asr = load_r2t2(args.model_path, device=args.device, dtype=args.dtype,
                        max_new_tokens=8)
    print(f"model loaded in {time.time() - t0:.1f}s  ({asr.decode_backend})", flush=True)

    step = int(round(args.chunk_ms / 1000.0 * SR))
    lookahead = int(round(args.lookahead_ms / 1000.0 * SR))
    chunk_sec = args.chunk_ms / 1000.0

    # Warm up: first call pays Metal kernel compilation.
    warm_state = asr.init_streaming_state(
        language=args.language, unfixed_chunk_num=0, unfixed_token_num=1,
        chunk_size_sec=chunk_sec)
    warm_state.chunk_size_sec = (step + lookahead) / SR
    warm_state.chunk_size_samples = step + lookahead
    tw = time.time()
    asr.streaming_transcribe(wav[: step + lookahead], warm_state, 2)
    print(f"warmup chunk: {(time.time() - tw) * 1000:.0f}ms", flush=True)

    state = asr.init_streaming_state(
        language=args.language, unfixed_chunk_num=0, unfixed_token_num=1,
        chunk_size_sec=chunk_sec)

    max_new_tokens = max(1, int((step + lookahead) / 1280))
    max_new_tokens_floor = min(32, max(4, 2 * int(step / 1280)))
    costs, accum, seen = [], [], []
    last_fixed, total_tokens = "", []
    pos, is_first, budget_ms = 0, True, args.chunk_ms

    while pos < wav.shape[0]:
        if is_first:
            seg = wav[pos: pos + step + lookahead]
            state.chunk_size_sec = (step + lookahead) / SR
            state.chunk_size_samples = step + lookahead
            is_first = False
        else:
            seg = wav[pos: pos + step]
            state.chunk_size_sec = chunk_sec
            state.chunk_size_samples = step
        pos += seg.shape[0]

        t = time.time()
        _, fixed = asr.streaming_transcribe(seg, state, int(max_new_tokens))
        cost_ms = (time.time() - t) * 1000
        costs.append(cost_ms)
        accum.append(state.audio_accum.shape[0] / SR)

        if len(fixed) > len(last_fixed):
            for tok in split_text_to_tokens(fixed[len(last_fixed):]):
                total_tokens.append(tok)
                if len(total_tokens) > 10:
                    total_tokens.pop(0)
            last_fixed = fixed
            max_new_tokens = max(1, int(step / 1280))
        else:
            if not is_last_token_chinese(total_tokens):
                max_new_tokens += 0.5
            else:
                max_new_tokens = max(1, int(step / 1280))
        if is_last_token_chinese(total_tokens):
            max_new_tokens *= 2
        max_new_tokens = min(max_new_tokens_floor, max_new_tokens)

        seen.append(len(costs))
        if len(costs) % 10 == 0:
            print(f"  chunk {len(costs):3d}  accum={accum[-1]:5.2f}s  "
                  f"cost={cost_ms:7.1f}ms  text={last_fixed[-24:]!r}", flush=True)
        if args.max_chunks and len(costs) >= args.max_chunks:
            break
        if seg.shape[0] / SR < chunk_sec:
            break

    print()
    print(f"chunks              : {len(costs)}")
    print(f"budget per chunk    : {budget_ms} ms")
    print(f"median cost         : {statistics.median(costs):.1f} ms")
    print(f"mean cost           : {statistics.mean(costs):.1f} ms")
    print(f"p90 cost            : {sorted(costs)[int(len(costs) * 0.9)]:.1f} ms")
    print(f"max cost            : {max(costs):.1f} ms")
    over = sum(1 for c in costs if c > budget_ms)
    print(f"over budget         : {over}/{len(costs)} chunks "
          f"({100 * over / len(costs):.0f}%)")
    print(f"real-time factor    : {statistics.mean(costs) / budget_ms:.2f}x "
          f"(<1.0 keeps up)")
    if len(costs) >= 20:
        first10 = statistics.mean(costs[:10])
        last10 = statistics.mean(costs[-10:])
        print(f"first 10 chunks     : {first10:.1f} ms  (accum {accum[9]:.2f}s)")
        print(f"last 10 chunks      : {last10:.1f} ms  (accum {accum[-1]:.2f}s)")
        print(f"growth              : {last10 / first10:.2f}x")
    print(f"\ntranscript: {last_fixed!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
