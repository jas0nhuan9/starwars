# coding=utf-8
"""Drive the whole local interpreter the way the web page does.

Connects to T3PO's own /ws/simul-demo, streams a wav at wall-clock speed as the
browser's microphone would, and prints the ASR and translation events with the
time each one arrived.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import librosa
import numpy as np
import websockets

ROOT = Path(__file__).resolve().parents[1]
SR = 16_000


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/simul-demo")
    parser.add_argument("--audio", default=str(
        ROOT / "vendor" / "Confucius4-R2T2" / "resources" / "test.wav"))
    parser.add_argument("--direction", default="zh2en", choices=("zh2en", "en2zh"))
    parser.add_argument("--latency_mode", default="native", choices=("low", "native", "high"))
    parser.add_argument("--frame_ms", type=int, default=40)
    args = parser.parse_args()

    wav, _ = librosa.load(args.audio, sr=SR, mono=True)
    pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
    frame_bytes = int(args.frame_ms / 1000 * SR) * 2
    print(f"{args.audio}  {len(wav) / SR:.2f}s  {args.direction}  "
          f"latency_mode={args.latency_mode}")

    source_parts: list[str] = []
    target_parts: list[str] = []

    async with websockets.connect(args.url, ping_interval=None, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "init",
            "direction": args.direction,
            "latency_mode": args.latency_mode,
        }))

        start = time.time()
        finished = asyncio.Event()

        async def reader() -> None:
            try:
                async for raw in ws:
                    event = json.loads(raw)
                    kind = event.get("type")
                    stamp = time.time() - start
                    if kind == "asr":
                        text = event.get("text") or ""
                        source_parts.append(text)
                        flag = " [utterance end]" if event.get("reset") else ""
                        print(f"  {stamp:5.2f}s  ASR  +{text!r}{flag}")
                    elif kind == "translation":
                        text = event.get("text") or ""
                        target_parts.append(text)
                        print(f"  {stamp:5.2f}s  >>>  {text}")
                    elif kind in {"loading", "init_ok", "pause_ok", "resume_ok"}:
                        print(f"  {stamp:5.2f}s  {kind} {event.get('component', '')}".rstrip())
                    elif kind == "metrics":
                        print(f"  {stamp:5.2f}s  metrics {json.dumps(event, ensure_ascii=False)[:160]}")
                    elif kind == "error":
                        print(f"  {stamp:5.2f}s  ERROR {event.get('message') or event}")
                    elif kind == "ended":
                        print(f"  {stamp:5.2f}s  ended")
                        break
            finally:
                finished.set()

        reader_task = asyncio.create_task(reader())

        # Give the server a moment to open its ASR connection before audio.
        await asyncio.sleep(0.5)
        for offset in range(0, len(pcm), frame_bytes):
            await ws.send(pcm[offset: offset + frame_bytes])
            await asyncio.sleep(args.frame_ms / 1000)
        await ws.send(json.dumps({"type": "end"}))

        try:
            await asyncio.wait_for(finished.wait(), timeout=60)
        except asyncio.TimeoutError:
            print("  timed out waiting for 'ended'")
        reader_task.cancel()

    print(f"\nwall clock   : {time.time() - start:.2f}s for {len(wav) / SR:.2f}s of audio")
    print(f"source       : {''.join(source_parts)}")
    print(f"translation  : {' '.join(t for t in target_parts if t)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
