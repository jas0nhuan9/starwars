# coding=utf-8
"""Check the whole local interpreter, end to end, and report pass/fail.

Exercises the real services over their real protocols -- no mocks -- so a pass
means the stack a person would use is working:

    bash scripts/start_all.sh
    .venv/bin/python scripts/selftest.py

Exit code is the number of failed checks.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zh_script import to_display, to_model  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
REFERENCE_WAV = ROOT / "vendor" / "Confucius4-R2T2" / "resources" / "test.wav"


def has_simplified(text: str) -> bool:
    """True if any character is a Simplified-only form.

    Asking OpenCC beats hand-listing characters: a hand-written list is easy
    to contaminate with forms both scripts share -- 客 in 顧客, say -- which
    then flags correct Traditional text as Simplified.  A character is
    Simplified-only exactly when converting it to Traditional changes it.
    """

    from opencc import OpenCC

    converter = OpenCC("s2t")
    return any(converter.convert(char) != char for char in text)

results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""),
          flush=True)
    return ok


def run(args: list[str], timeout: float = 240) -> str:
    proc = subprocess.run([PY, *args], capture_output=True, text=True, timeout=timeout)
    return proc.stdout + proc.stderr


def section(title: str) -> None:
    print(f"\n{title}")


def main() -> int:
    section("services")
    for name, url in (
        ("llama-server", "http://127.0.0.1:8010/v1/models"),
        ("min_tokens proxy", "http://127.0.0.1:8011/v1/models"),
        ("ASR", "http://127.0.0.1:8093/health"),
        ("web UI", "http://127.0.0.1:8000/api/v1/health"),
    ):
        try:
            response = httpx.get(url, timeout=10)
            check(f"{name} responds", response.status_code == 200,
                  f"HTTP {response.status_code}")
        except Exception as error:
            check(f"{name} responds", False, type(error).__name__)

    section("configuration")
    try:
        asr = httpx.get("http://127.0.0.1:8093/health", timeout=10).json()
        check("ASR decodes through llama.cpp/Metal", asr.get("backend") == "gguf",
              asr.get("decode_backend", ""))
        check("chunk size is the measured default", asr.get("chunk_ms") == 320,
              f"{asr.get('chunk_ms')} ms")
        check("Chinese display is Traditional", asr.get("zh_variant") == "traditional",
              str(asr.get("zh_variant")))
    except Exception as error:
        check("ASR health readable", False, type(error).__name__)
    try:
        web = httpx.get("http://127.0.0.1:8000/api/v1/health", timeout=10).json()
        check("web UI has the ASR service wired", bool(web.get("asr_configured")))
        check("web UI targets the T3PO model",
              web.get("translation_model") == "Confucius4-T3PO",
              str(web.get("translation_model")))
    except Exception as error:
        check("web health readable", False, type(error).__name__)

    section("script conversion")
    probe = "这个软件需要更多内存，打印机的驱动程序也要更新。"
    displayed = to_display(probe)
    check("Taiwanese vocabulary is applied",
          all(term in displayed for term in ("軟體", "記憶體", "印表機", "驅動程式")),
          displayed)
    check("display -> model round trip is lossless", to_model(displayed) == probe)
    check("English passes through untouched",
          to_display("Now please do some tests.") == "Now please do some tests.")

    section("ASR service protocol")
    output = run(["scripts/ws_client_test.py", "--audio", str(REFERENCE_WAV)])
    transcript = ""
    for line in output.splitlines():
        if line.startswith("transcript:"):
            transcript = line.split(":", 1)[1].strip()
    check("reference audio transcribes correctly",
          transcript.startswith("之前有顧客自己帶酒水"), transcript or "(nothing)")
    check("transcript is Traditional",
          bool(transcript) and not has_simplified(transcript))
    check("increments arrive while speaking", output.count("asr=") >= 5,
          f"{output.count('asr=')} increments")

    section("full pipeline, Chinese to English")
    output = run(["scripts/e2e_test.py", "--audio", str(REFERENCE_WAV)])
    source = translation = ""
    for line in output.splitlines():
        if line.startswith("source"):
            source = line.split(":", 1)[1].strip()
        elif line.startswith("translation"):
            translation = line.split(":", 1)[1].strip()
    check("source transcript is Traditional",
          bool(source) and not has_simplified(source), source or "(nothing)")
    check("translation is produced in English",
          "drink" in translation.lower() or "charge" in translation.lower(),
          translation or "(nothing)")

    section("full pipeline, English to Chinese")
    english = ROOT / "logs" / "_selftest_en.wav"
    if not english.exists():
        subprocess.run(["say", "-v", "Samantha", "-o", str(english.with_suffix(".aiff")),
                        "The new software needs more memory, and the printer driver "
                        "must be updated."], capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i",
                        str(english.with_suffix(".aiff")), "-ar", "16000", "-ac", "1",
                        str(english)], capture_output=True)
        english.with_suffix(".aiff").unlink(missing_ok=True)
    output = run(["scripts/e2e_test.py", "--audio", str(english), "--direction", "en2zh"])
    translation = ""
    for line in output.splitlines():
        if line.startswith("translation"):
            translation = line.split(":", 1)[1].strip()
    check("Chinese translation is produced", len(translation) > 4, translation or "(nothing)")
    check("Chinese translation is Traditional",
          bool(translation) and not has_simplified(translation))

    section("web UI")
    try:
        page = httpx.get("http://127.0.0.1:8000/", timeout=10).text
        check("interface text is Traditional",
              "延遲檔位" in page and "延迟档位" not in page)
        check("document locale is zh-TW", 'lang="zh-TW"' in page)
        script = httpx.get("http://127.0.0.1:8000/app.js", timeout=10).text
        check("app.js is served converted", "尚未開始" in script or "未連線" in script)
        check("upstream routes still reachable behind the overrides",
              httpx.get("http://127.0.0.1:8000/api/v1/health", timeout=10).status_code == 200)
    except Exception as error:
        check("web UI readable", False, type(error).__name__)

    section("microphone capture")
    output = run(["scripts/mic_test.py", "--list-devices"], timeout=60)
    check("an input device is discoverable", "[0]" in output,
          output.strip().splitlines()[-1] if output.strip() else "(none)")
    saved = ROOT / "logs" / "_selftest_mic.wav"
    output = run(["scripts/mic_test.py", "--seconds", "3", "--quiet",
                  "--save", str(saved)], timeout=120)
    check("capture resolves a device by name", "capturing [" in output,
          next((l for l in output.splitlines() if l.startswith("capturing")), ""))
    check("captured audio is written", saved.exists() and saved.stat().st_size > 32_000,
          f"{saved.stat().st_size if saved.exists() else 0} bytes")
    check("no traceback on a clean exit", "Traceback" not in output)
    saved.unlink(missing_ok=True)

    section("decode cost")
    output = run(["scripts/bench_asr.py", "--audio", str(REFERENCE_WAV),
                  "--chunk_ms", "320"], timeout=300)
    rtf = median = None
    for line in output.splitlines():
        if line.startswith("real-time factor"):
            rtf = float(line.split(":")[1].strip().split("x")[0])
        elif line.startswith("median cost"):
            median = float(line.split(":")[1].strip().split()[0])
    check("decoding keeps up with real time", rtf is not None and rtf < 1.0,
          f"real-time factor {rtf}, median {median} ms")

    failed = sum(1 for ok, _, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks passed"
          + (f", {failed} FAILED" if failed else ""))
    if failed:
        for ok, name, detail in results:
            if not ok:
                print(f"  FAILED: {name}  {detail}")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
