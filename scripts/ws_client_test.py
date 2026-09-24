# coding=utf-8
"""Exercise the R2T2 WebSocket service the way T3PO's client does.

Streams a wav file at wall-clock speed (as a microphone would) and prints each
increment with the latency between sending the audio that produced it and
receiving the text.
"""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import librosa
import numpy as np
import websockets

ROOT = Path(__file__).resolve().parents[1]
EOS = "YOUDAO_ONETIME_ASR_STREAM_EOS"
SR = 16_000


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8093/asr_stream_api_v1")
    parser.add_argument("--audio", default=str(
        ROOT / "vendor" / "Confucius4-R2T2" / "resources" / "test.wav"))
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--secret", default="test0102")
    parser.add_argument("--frame_ms", type=int, default=40,
                        help="audio frame size sent per message")
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--fast", dest="realtime", action="store_false",
                        help="send as fast as possible instead of at 1x speed")
    args = parser.parse_args()

    wav, _ = librosa.load(args.audio, sr=SR, mono=True)
    pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
    frame_bytes = int(args.frame_ms / 1000 * SR) * 2
    print(f"streaming {len(wav) / SR:.2f}s of audio in {args.frame_ms}ms frames "
          f"({'1x realtime' if args.realtime else 'as fast as possible'})")

    transcript = []
    async with websockets.connect(args.url, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "requestId": str(uuid.uuid4()),
            "channels": 1,
            "sample_rate": SR,
            "language": args.language,
            "use_vad": True,
            "secret_key": args.secret,
            "mode": "slow",
            "smooth": False,
        }))
        greeting = json.loads(await ws.recv())
        print("server:", greeting)

        start = time.time()
        done = asyncio.Event()

        async def reader() -> None:
            try:
                async for raw in ws:
                    payload = json.loads(raw)
                    if payload.get("status") == "error":
                        print("ERROR:", payload.get("msg"))
                        break
                    message = payload.get("msg") or {}
                    text = message.get("text") or ""
                    if text or message.get("reset"):
                        transcript.append(text)
                        flag = " [reset]" if message.get("reset") else ""
                        print(f"  t={time.time() - start:5.2f}s  "
                              f"asr={message.get('asr_cost_ms', 0):6.1f}ms  "
                              f"lag={message.get('lag_ms', 0):6.1f}ms  "
                              f"+{text!r}{flag}")
            finally:
                done.set()

        reader_task = asyncio.create_task(reader())

        for offset in range(0, len(pcm), frame_bytes):
            await ws.send(pcm[offset: offset + frame_bytes])
            if args.realtime:
                await asyncio.sleep(args.frame_ms / 1000)
        await ws.send(EOS)
        try:
            await asyncio.wait_for(done.wait(), timeout=30)
        except asyncio.TimeoutError:
            print("timed out waiting for the final result")
        reader_task.cancel()

    print(f"\nelapsed {time.time() - start:.2f}s for {len(wav) / SR:.2f}s of audio")
    print("transcript:", "".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
