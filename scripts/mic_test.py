# coding=utf-8
"""Live interpretation from the Mac's microphone, in the terminal.

Captures the default (or a chosen) input device through ffmpeg, streams 16 kHz
PCM16 to T3PO's own /ws/simul-demo, and prints the transcript and translation
as they arrive -- the same path the web page uses, without a browser.

    .venv/bin/python scripts/mic_test.py                 # talk, Ctrl-C to stop
    .venv/bin/python scripts/mic_test.py --seconds 20 --direction en2zh
    .venv/bin/python scripts/mic_test.py --list-devices

The level meter is there to answer the first question of any microphone
problem: is audio actually arriving?  A row of dots means silence reached the
socket, so the issue is capture (device or macOS microphone permission), not
the models.
"""

import argparse
import asyncio
import json
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parents[1]
SR = 16_000

# The level meter redraws one line with \r, so every other line has to erase
# what the meter left behind or the two interleave mid-word.
CLEAR_LINE = "\r\033[2K"


# avfoundation renumbers its devices whenever one connects or disconnects --
# unplug a headset and the built-in microphone's index changes underneath you --
# so devices are enumerated on every run and matched by name.
BUILT_IN_HINTS = ("macbook", "built-in", "builtin", "內建", "内建", "imac", "mac mini")


def enumerate_devices() -> list[tuple[int, str]]:
    """Return ``[(index, name)]`` for the audio inputs ffmpeg can see."""

    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation",
         "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    devices: list[tuple[int, str]] = []
    in_audio = False
    for line in proc.stderr.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if not in_audio:
            continue
        body = line.split("] ", 1)[-1] if "] " in line else ""
        if not body.startswith("["):
            break
        index, _, name = body.partition("] ")
        try:
            devices.append((int(index.lstrip("[")), name.strip()))
        except ValueError:
            break
    return devices


def list_devices() -> int:
    devices = enumerate_devices()
    if not devices:
        print("no audio input devices found")
        return 1
    print("audio input devices (pass the number or part of the name to --device):")
    for index, name in devices:
        print(f"  [{index}] {name}")
    return 0


def resolve_device(spec: str, devices: list[tuple[int, str]]) -> tuple[int, str]:
    """Pick a device from ``spec``: an index, a name fragment, or "" for auto.

    Auto prefers the machine's own microphone, so unplugging a headset mid
    session does not silently point the capture at a device that is gone.
    """

    if not devices:
        raise RuntimeError("ffmpeg reported no audio input devices")

    spec = (spec or "").strip()
    if not spec:
        for index, name in devices:
            if any(hint in name.lower() for hint in BUILT_IN_HINTS):
                return index, name
        return devices[0]

    if spec.isdigit():
        wanted = int(spec)
        for index, name in devices:
            if index == wanted:
                return index, name
        available = ", ".join(f"[{i}] {n}" for i, n in devices)
        raise RuntimeError(f"no audio device with index {wanted}; available: {available}")

    for index, name in devices:
        if spec.lower() in name.lower():
            return index, name
    available = ", ".join(f"[{i}] {n}" for i, n in devices)
    raise RuntimeError(f"no audio device matching {spec!r}; available: {available}")


def open_capture(device: str) -> subprocess.Popen:
    """ffmpeg reading the input device as raw 16 kHz mono PCM16 on stdout."""

    return subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "avfoundation", "-i", f":{device}",
         "-ar", str(SR), "-ac", "1", "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def meter(samples: np.ndarray, width: int = 12) -> str:
    rms = float(np.sqrt((samples.astype(np.float32) / 32768.0) ** 2).mean())
    filled = min(width, int(rms * width * 12))
    return "#" * filled + "." * (width - filled)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/simul-demo")
    parser.add_argument("--direction", default="zh2en", choices=("zh2en", "en2zh"))
    parser.add_argument("--latency_mode", default="native", choices=("low", "native", "high"))
    parser.add_argument("--device", default="",
                        help="audio device index or name fragment; "
                             "default picks this Mac's own microphone")
    parser.add_argument("--frame_ms", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="stop after this long; 0 waits for Ctrl-C")
    parser.add_argument("--quiet", action="store_true", help="hide the level meter")
    parser.add_argument("--save", default="",
                        help="also write the captured audio to this wav, so the "
                             "same words can be replayed against other configs")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        return list_devices()

    try:
        device_index, device_name = resolve_device(args.device, enumerate_devices())
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    frame_bytes = int(args.frame_ms / 1000 * SR) * 2
    capture = open_capture(str(device_index))

    recording = None
    if args.save:
        recording = wave.open(args.save, "wb")
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(SR)
    print(f"capturing [{device_index}] {device_name}  {args.direction}  "
          f"{'Ctrl-C to stop' if not args.seconds else f'{args.seconds:.0f}s'}")

    source: list[str] = []
    target: list[str] = []
    start = time.time()

    try:
        async with websockets.connect(args.url, ping_interval=None, max_size=None) as ws:
            await ws.send(json.dumps({
                "type": "init",
                "direction": args.direction,
                "latency_mode": args.latency_mode,
            }))

            async def reader() -> None:
                async for raw in ws:
                    event = json.loads(raw)
                    kind = event.get("type")
                    stamp = time.time() - start
                    if kind == "asr":
                        text = event.get("text") or ""
                        if text:
                            source.append(text)
                            print(f"{CLEAR_LINE}{stamp:6.2f}s  ASR  {text}", flush=True)
                    elif kind == "translation":
                        text = event.get("text") or ""
                        if text:
                            target.append(text)
                            print(f"{CLEAR_LINE}{stamp:6.2f}s  >>>  {text}", flush=True)
                    elif kind == "error":
                        print(f"{CLEAR_LINE}{stamp:6.2f}s  ERROR "
                              f"{event.get('message') or event}", flush=True)
                    elif kind == "ended":
                        break

            reader_task = asyncio.create_task(reader())
            loop = asyncio.get_running_loop()

            # Ctrl-C has to stop the capture loop rather than tear down the
            # event loop: the tail of what was said is only translated once
            # the server is told the stream ended.
            stop = asyncio.Event()
            try:
                loop.add_signal_handler(signal.SIGINT, stop.set)
            except NotImplementedError:      # pragma: no cover - non-POSIX
                pass

            while not stop.is_set():
                if args.seconds and time.time() - start > args.seconds:
                    break
                chunk = await loop.run_in_executor(
                    None, capture.stdout.read, frame_bytes)
                if not chunk:
                    break
                await ws.send(chunk)
                if recording is not None:
                    recording.writeframes(chunk)
                if not args.quiet:
                    samples = np.frombuffer(chunk, dtype="<i2")
                    print(f"\rmic [{meter(samples)}] {time.time() - start:5.1f}s",
                          end="", flush=True)

            print(CLEAR_LINE, end="", flush=True)
            if stop.is_set():
                print("stopping, flushing the last words...", flush=True)
            await ws.send(json.dumps({"type": "end"}))
            try:
                await asyncio.wait_for(reader_task, timeout=30)
            except asyncio.TimeoutError:
                reader_task.cancel()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if recording is not None:
            recording.close()
        capture.terminate()
        try:
            capture.wait(timeout=5)
        except subprocess.TimeoutExpired:
            capture.kill()
        print()
        print(f"heard      : {''.join(source) or '(nothing)'}")
        print(f"translated : {' '.join(target) or '(nothing)'}")
        if args.save:
            print(f"audio saved : {args.save}")
        stderr = capture.stderr.read().decode(errors="replace").strip()
        if stderr:
            print(f"ffmpeg: {stderr.splitlines()[-1]}", file=sys.stderr)
            if "Input/output error" in stderr:
                print("the capture device could not be opened. Devices now:",
                      file=sys.stderr)
                for index, name in enumerate_devices():
                    print(f"  [{index}] {name}", file=sys.stderr)
                print("macOS also returns this when microphone access is denied "
                      "to the app running this script.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:      # a second Ctrl-C during the flush
        raise SystemExit(130)
