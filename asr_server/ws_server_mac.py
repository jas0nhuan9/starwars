# coding=utf-8
"""R2T2 streaming ASR as a WebSocket service, for Apple Silicon.

Speaks the same ``/asr_stream_api_v1`` protocol as upstream's ``ws_server.py``,
so Confucius4-T3PO's ``inference/asr.py`` client connects to it unmodified.
Upstream's server is Sanic + vLLM + CUDA; this one is FastAPI and decodes
through :mod:`asr_server.r2t2_mps` on the Apple GPU.

Protocol (unchanged):

* first frame is a JSON header; ``secret_key`` is validated and a mismatch
  closes with 4401.  The reply is ``{"status": "connected", ...}``.
* then raw PCM16LE mono frames at 16 kHz as binary messages.
* the text ``YOUDAO_ONETIME_ASR_STREAM_EOS`` ends the request; the final
  result is sent and the socket closes.
* each result is ``{"status": "success", "msg": {"text": <new increment>,
  "reset": <utterance ended>, ...}}``.  ``text`` carries only newly fixed
  characters, so the client needs no prefix reconciliation.

Three deviations from upstream, all measured on an M4 Pro (see
``scripts/bench_asr.py``):

* 320 ms chunks instead of 160 ms.  Decoding one chunk costs ~112 ms through
  llama.cpp, which fits a 160 ms budget in isolation (real-time factor 0.70),
  but not once the 14B translator is probing the same GPU: at 160 ms the
  transcript ends up ~2.8 s behind the speaker, at 320 ms ~0.5 s, and 480 ms
  buys nothing further.
* utterances are capped (``MAX_UTTERANCE_SEC``).  R2T2's streaming design
  re-sends the whole accumulated utterance every chunk, so the audio encoder
  redoes that work and cost grows with utterance length -- roughly
  70 ms + 12 ms per accumulated second here -- no matter which decoder is
  underneath.  VAD normally cuts earlier; this is the backstop.
* VAD runs on CPU, and waits ~800 ms of silence instead of upstream's ~200 ms,
  so it segments on sentence pauses rather than on every comma.  At 200 ms a
  single sentence got cut three times and lost a character at each seam.
* one more token is held back before being committed
  (``ASR_UNFIXED_TOKEN_NUM``, 2 rather than upstream's 1).  Committed text is
  never revised, so a token fixed before its right context arrives stays wrong
  -- "请做些" came out as "请坐下" in a live run.  Holding one more token back
  buys that context.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "Confucius4-R2T2"))

from asr_server.r2t2_gguf import load_r2t2_gguf  # noqa: E402
from asr_server.r2t2_mps import load_r2t2  # noqa: E402
from zh_script import to_display, variant as zh_variant  # noqa: E402

EOS_MESSAGE = "YOUDAO_ONETIME_ASR_STREAM_EOS"
SAMPLE_RATE = 16_000


def _divert_native_logs(path: str) -> Any:
    """Send the C-level stdout/stderr to their own file.

    llama.cpp logs per chunk through its global logger, and neither
    ``mtmd_context_params`` nor the pybind extension exposes a verbosity knob,
    so the only place to quiet it is the file descriptors.  Python keeps
    writing to a duplicate of the original stderr, so our own log lines and
    tracebacks stay where the operator expects them.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    kept = io.TextIOWrapper(io.FileIO(os.dup(2), "w", closefd=True), write_through=True)
    # Truncated per start, like the logs start_all.sh redirects into.
    sink = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(sink, 1)
    os.dup2(sink, 2)
    os.close(sink)
    sys.stderr = kept
    sys.stdout = kept
    return kept


_LOG_STREAM = sys.stderr
if os.getenv("ASR_NATIVE_LOG", "").strip().lower() not in {"stderr", "-"}:
    _LOG_STREAM = _divert_native_logs(
        os.getenv("ASR_NATIVE_LOG", "").strip()
        or str(ROOT / "logs" / "asr-native.log")
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=_LOG_STREAM,
)
logger = logging.getLogger("r2t2-mac")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class Settings:
    """Service configuration, all overridable by environment variable."""

    def __init__(self) -> None:
        # "gguf" decodes through llama.cpp/Metal (1.5 GB of weights); "mps"
        # runs the fp16 checkpoint through Transformers (4.3 GB).  Both reuse
        # R2T2's own streaming logic; see asr_server/r2t2_gguf.py.
        self.backend = _env("ASR_BACKEND", "gguf").lower()
        self.model_path = _env("ASR_MODEL_PATH", str(ROOT / "models" / "Confucius4-R2T2"))
        self.gguf_dir = _env("ASR_GGUF_DIR", str(ROOT / "models" / "Confucius4-R2T2-GGUF"))
        self.gguf_model = _env("ASR_GGUF_MODEL", "Confucius4-R2T2-Q4_K_M.gguf")
        self.gguf_mmproj = _env("ASR_GGUF_MMPROJ", "mmproj-Confucius4-R2T2-Q8_0.gguf")
        self.vad_model_path = _env("VAD_MODEL_PATH", str(ROOT / "models" / "vad" / "Stream-VAD"))
        self.host = _env("ASR_HOST", "127.0.0.1")
        self.port = _int("ASR_PORT", 8093)
        self.device = _env("ASR_DEVICE", "mps")
        self.dtype = _env("ASR_DTYPE", "auto")
        self.secret_keys = [
            k.strip() for k in _env("ASR_SECRET_KEYS", "test0102").split(",") if k.strip()
        ]
        self.chunk_ms = _int("ASR_CHUNK_MS", 320)
        self.lookahead_ms = _int("ASR_LOOKAHEAD_MS", 160)
        self.unfixed_token_num = _int("ASR_UNFIXED_TOKEN_NUM", 2)
        self.max_utterance_sec = _float("ASR_MAX_UTTERANCE_SEC", 12.0)
        self.vad_min_silence_frame = _int("VAD_MIN_SILENCE_FRAME", 80)
        self.vad_speech_threshold = _float("VAD_SPEECH_THRESHOLD", 0.4)


settings = Settings()

# One model, one GPU: inference is serialized on a dedicated thread so the
# event loop stays free to read audio and answer pings while a chunk decodes.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="r2t2")
_infer_lock = asyncio.Lock()

asr_model: Any = None
vad_factory: Any = None


# --------------------------------------------------------------------------- #
# text helpers (ported from upstream ws_server.py)
# --------------------------------------------------------------------------- #

_PUNCT_CHARS = "，。！？、；：,.!?;:~…·\"'()（）《》—-"
_PUNCT_RE = re.compile(r"[%s\s]+" % re.escape(_PUNCT_CHARS))


def _normalize_for_pattern(text: str) -> str:
    return _PUNCT_RE.sub("", text).lower()


def split_text_to_tokens(text: str) -> list[str]:
    """Chinese by character, English by word, punctuation dropped."""
    tokens: list[str] = []
    for match in re.finditer(r"[a-zA-Z]+|[^a-zA-Z]", text):
        token = match.group()
        if token.strip() and token not in _PUNCT_CHARS:
            tokens.append(token)
    return tokens


def is_last_token_chinese(tokens: list[str]) -> bool:
    if not tokens:
        return False
    return any("一" <= ch <= "鿿" for ch in tokens[-1])


def detect_hallucination(text: str, pattern_repeat_thresh: int = 5,
                         max_pattern_len: int = 50,
                         tail_check_len: int = 256) -> tuple[bool, str]:
    """Spot a repetition loop in the tail of the transcript.

    A stuck decoder emits the same fragment over and over; the state has to be
    thrown away when that happens.  Checks the tail both literally and with
    punctuation and spacing removed.
    """

    if not text:
        return False, ""
    tail = text[-tail_check_len:]

    n = len(tail)
    for k in range(1, max_pattern_len + 1):
        if n < k * pattern_repeat_thresh:
            continue
        pattern = tail[-k:]
        if pattern.strip(_PUNCT_CHARS + " \t") == "":
            continue
        if all(tail[-(r + 1) * k: -r * k] == pattern
               for r in range(1, pattern_repeat_thresh)):
            return True, f"tail_pattern:'{pattern}'x{pattern_repeat_thresh}+"

    norm = _normalize_for_pattern(tail)
    n2 = len(norm)
    for k in range(3, max_pattern_len + 1):
        if n2 < k * pattern_repeat_thresh:
            continue
        pattern = norm[-k:]
        if all(norm[-(r + 1) * k: -r * k] == pattern
               for r in range(1, pattern_repeat_thresh)):
            return True, f"tail_pattern_norm:'{pattern}'x{pattern_repeat_thresh}+"

    return False, ""


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the models once, before the first connection is served."""

    await _load_models()
    yield


app = FastAPI(
    title="Confucius4-R2T2 streaming ASR (Apple Silicon)", lifespan=_lifespan
)


async def _load_models() -> None:
    global asr_model, vad_factory

    t0 = time.time()
    if settings.backend == "gguf":
        logger.info("loading R2T2 GGUF from %s (llama.cpp/Metal)", settings.gguf_dir)
        asr_model = load_r2t2_gguf(
            settings.gguf_dir, settings.model_path,
            model_gguf_name=settings.gguf_model,
            mmproj_gguf_name=settings.gguf_mmproj,
            max_new_tokens=8,
        )
    else:
        logger.info("loading R2T2 from %s on %s", settings.model_path, settings.device)
        asr_model = load_r2t2(
            settings.model_path, device=settings.device, dtype=settings.dtype,
            max_new_tokens=8,
        )
    logger.info("R2T2 ready in %.1fs (%s)", time.time() - t0, asr_model.decode_backend)

    from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

    def _make_vad() -> Any:
        # use_gpu=False: the detector is small, and the Apple GPU is busy with
        # ASR.  A fresh instance per request keeps its state per stream.
        config = FireRedStreamVadConfig(
            use_gpu=False,
            smooth_window_size=5,
            speech_threshold=settings.vad_speech_threshold,
            pad_start_frame=5,
            min_speech_frame=8,
            max_speech_frame=2000,
            min_silence_frame=settings.vad_min_silence_frame,
            chunk_max_frame=30000,
        )
        return FireRedStreamVad.from_pretrained(settings.vad_model_path, config)

    vad_factory = _make_vad
    _make_vad()  # fail fast if the weights are missing
    logger.info("VAD ready (min_silence_frame=%d)", settings.vad_min_silence_frame)

    # Warm up Metal kernels so the first real chunk is not the slow one.
    step = int(settings.chunk_ms / 1000 * SAMPLE_RATE)
    look = int(settings.lookahead_ms / 1000 * SAMPLE_RATE)
    for language in ("Chinese", "English"):
        state = asr_model.init_streaming_state(
            language=language, unfixed_chunk_num=0,
            unfixed_token_num=settings.unfixed_token_num,
            chunk_size_sec=settings.chunk_ms / 1000,
        )
        state.chunk_size_sec = (step + look) / SAMPLE_RATE
        state.chunk_size_samples = step + look
        asr_model.streaming_transcribe(
            np.zeros(step + look, dtype=np.float32), state, 2)
    logger.info("warmup done; listening on ws://%s:%d/asr_stream_api_v1",
                settings.host, settings.port)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok" if asr_model is not None else "loading",
        "backend": settings.backend,
        "decode_backend": getattr(asr_model, "decode_backend", None),
        "chunk_ms": settings.chunk_ms,
        "lookahead_ms": settings.lookahead_ms,
        "max_utterance_sec": settings.max_utterance_sec,
        "zh_variant": zh_variant(),
    }


# --------------------------------------------------------------------------- #
# one streaming request
# --------------------------------------------------------------------------- #

class Session:
    """Chunk schedule and append-only bookkeeping for one audio stream."""

    def __init__(self, request_id: str, language: Optional[str], context: str,
                 use_vad: bool) -> None:
        self.request_id = request_id
        self.language = language
        self.context = context
        self.use_vad = use_vad

        self.step = int(round(settings.chunk_ms / 1000.0 * SAMPLE_RATE))
        self.lookahead = int(round(settings.lookahead_ms / 1000.0 * SAMPLE_RATE))
        self.chunk_sec = settings.chunk_ms / 1000.0

        self.buffer = np.zeros(0, dtype=np.float32)
        self.is_first_chunk = True
        self.last_fixed_text = ""
        self.recent_tokens: list[str] = []
        self.vad = vad_factory() if use_vad else None

        # Upstream's adaptive budget: spend more decode steps when the text
        # stopped growing, fewer right after it moved.
        self.max_new_tokens: float = max(1, (self.step + self.lookahead) / 1280)
        self.max_new_tokens_floor = min(32, max(4, 2 * int(self.step / 1280)))

        self.state = self._new_state()

    def _new_state(self) -> Any:
        return asr_model.init_streaming_state(
            context=self.context,
            language=self.language,
            unfixed_chunk_num=0,
            unfixed_token_num=settings.unfixed_token_num,
            chunk_size_sec=self.chunk_sec,
        )

    def add_audio(self, pcm16: bytes) -> None:
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self.buffer = np.concatenate([self.buffer, samples])

    def next_segment(self) -> Optional[np.ndarray]:
        """Pop one scheduled chunk, or None while the buffer is short.

        The first chunk carries the lookahead, matching upstream so the model
        sees the same context it was tuned with.
        """

        want = self.step + self.lookahead if self.is_first_chunk else self.step
        if self.buffer.shape[0] < want:
            return None
        segment = self.buffer[:want]
        self.buffer = self.buffer[want:]
        self.state.chunk_size_sec = want / SAMPLE_RATE
        self.state.chunk_size_samples = want
        self.is_first_chunk = False
        return segment

    def update_token_budget(self, grew: bool) -> None:
        if grew:
            self.max_new_tokens = max(1, int(self.step / 1280))
        elif not is_last_token_chinese(self.recent_tokens):
            self.max_new_tokens += 0.5
        else:
            self.max_new_tokens = max(1, int(self.step / 1280))
        if is_last_token_chinese(self.recent_tokens):
            self.max_new_tokens *= 2
        self.max_new_tokens = min(self.max_new_tokens_floor, self.max_new_tokens)

    def note_new_text(self, increment: str) -> None:
        for token in split_text_to_tokens(increment):
            self.recent_tokens.append(token)
            if len(self.recent_tokens) > 10:
                self.recent_tokens.pop(0)

    @property
    def accumulated_sec(self) -> float:
        return float(self.state.audio_accum.shape[0]) / SAMPLE_RATE

    def speech_ended(self, segment: np.ndarray) -> bool:
        if self.vad is None:
            return False
        pcm = (segment * 32768).astype(np.int16)
        return any(r.is_speech_end for r in self.vad.detect_chunk(pcm))

    def reset_utterance(self) -> None:
        self.state = self._new_state()
        self.last_fixed_text = ""
        self.is_first_chunk = True


def _validate_header(header: Any) -> tuple[bool, str]:
    if not isinstance(header, dict):
        return False, "json header is expected"
    if not header.get("requestId"):
        return False, "requestId is required"
    if header.get("secret_key") not in settings.secret_keys:
        return False, "unauthorized"
    return True, ""


async def _run(func: Any, *args: Any) -> Any:
    """Run one blocking model call off the event loop, one at a time."""
    loop = asyncio.get_running_loop()
    async with _infer_lock:
        return await loop.run_in_executor(_executor, func, *args)


@app.websocket("/asr_stream_api_v1")
async def asr_stream(ws: WebSocket) -> None:
    await ws.accept()
    request_id = "-"
    try:
        first = await ws.receive()
        if first.get("type") == "websocket.disconnect":
            return
        raw = first.get("text")
        if raw is None:
            await ws.send_text(json.dumps({"status": "error", "msg": "json header is expected"}))
            await ws.close()
            return
        try:
            header = json.loads(raw)
        except ValueError:
            await ws.send_text(json.dumps({"status": "error", "msg": "invalid json header"}))
            await ws.close()
            return

        ok, error = _validate_header(header)
        if not ok:
            if error == "unauthorized":
                logger.info("rejected connection: invalid secret_key")
                await ws.close(code=4401, reason="Unauthorized")
            else:
                await ws.send_text(json.dumps({"status": "error", "msg": error}))
                await ws.close()
            return

        request_id = str(header.get("requestId") or uuid.uuid4())
        language = header.get("language", "zhen")
        if language == "zhen":
            language = None  # let the model decide, per upstream
        use_vad = bool(header.get("use_vad", False))
        context = "Smooth the text" if header.get("smooth", False) else ""
        system_prompt = header.get("system_prompt")
        if isinstance(system_prompt, str) and system_prompt.strip():
            context = "\n".join(filter(None, [context, system_prompt.strip()]))

        session = Session(request_id, language, context, use_vad)
        await ws.send_text(json.dumps({
            "status": "connected", "requestId": request_id, "msg": "",
        }))
        logger.info("[%s] connected language=%s use_vad=%s", request_id, language, use_vad)

        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                logger.info("[%s] client disconnected", request_id)
                return

            text_frame = message.get("text")
            if text_frame is not None:
                if text_frame.strip() == EOS_MESSAGE:
                    await _finish(ws, session)
                    await ws.close()
                    return
                continue  # ignore any other control text

            payload = message.get("bytes")
            if not payload:
                continue
            session.add_audio(payload)
            recv_time = time.time()

            while True:
                segment = session.next_segment()
                if segment is None:
                    break
                await _decode_one(ws, session, segment, recv_time)

    except WebSocketDisconnect:
        logger.info("[%s] disconnected", request_id)
    except Exception:
        logger.exception("[%s] request failed", request_id)
        try:
            await ws.send_text(json.dumps({
                "status": "error", "requestId": request_id,
                "msg": "the ASR service failed while decoding",
            }))
            await ws.close()
        except Exception:
            pass


async def _decode_one(ws: WebSocket, session: Session, segment: np.ndarray,
                      recv_time: float) -> None:
    """Decode one chunk and send its increment."""

    vad_t0 = time.time()
    speech_ended = session.speech_ended(segment)
    vad_cost_ms = round((time.time() - vad_t0) * 1000, 1)

    asr_t0 = time.time()
    _, fixed_text = await _run(
        asr_model.streaming_transcribe, segment, session.state,
        int(session.max_new_tokens),
    )
    asr_cost_ms = round((time.time() - asr_t0) * 1000, 1)

    grew = len(fixed_text) > len(session.last_fixed_text)
    if grew:
        increment = fixed_text[len(session.last_fixed_text):]
        session.note_new_text(increment)
        session.last_fixed_text = fixed_text
    else:
        increment = ""
    session.update_token_budget(grew)

    message = {"text": to_display(increment), "reset": False,
               "asr_cost_ms": asr_cost_ms}

    # End the utterance on a real pause, on a repetition loop, or when the
    # accumulated audio has grown enough that decode cost would start to lag.
    halluc, reason = detect_hallucination(session.last_fixed_text)
    too_long = session.accumulated_sec >= settings.max_utterance_sec
    finish_t0 = time.time()
    if speech_ended or halluc or too_long:
        if halluc:
            logger.info("[%s] utterance reset: hallucination (%s)", session.request_id, reason)
        elif too_long:
            logger.info("[%s] utterance reset: %.1fs accumulated",
                        session.request_id, session.accumulated_sec)
        final = await _run(
            asr_model.finish_streaming_transcribe, session.state,
            max(1, int((session.step + session.lookahead) / 1280)),
        )
        final = str(final).split("|")[0]
        tail = final[len(session.last_fixed_text):] if len(final) > len(session.last_fixed_text) else ""
        message = {
            "text": to_display(increment + tail),
            "reset": True,
            "asr_cost_ms": asr_cost_ms,
        }
        session.reset_utterance()
    finish_cost_ms = round((time.time() - finish_t0) * 1000, 1)

    message["total_cost_ms"] = vad_cost_ms + asr_cost_ms + finish_cost_ms
    message["lag_ms"] = round((time.time() - recv_time) * 1000, 1)
    await ws.send_text(json.dumps(
        {"status": "success", "requestId": session.request_id, "msg": message},
        ensure_ascii=False,
    ))


async def _finish(ws: WebSocket, session: Session) -> None:
    """Flush whatever is buffered, then send the final increment."""

    if session.buffer.shape[0] > 0:
        # Decode the partial tail as one short chunk.
        session.state.chunk_size_sec = session.buffer.shape[0] / SAMPLE_RATE
        session.state.chunk_size_samples = session.buffer.shape[0]
        segment, session.buffer = session.buffer, np.zeros(0, dtype=np.float32)
        try:
            _, fixed_text = await _run(
                asr_model.streaming_transcribe, segment, session.state,
                int(session.max_new_tokens),
            )
            if len(fixed_text) > len(session.last_fixed_text):
                session.last_fixed_text = fixed_text
        except Exception:
            logger.exception("[%s] tail chunk failed", session.request_id)

    final = await _run(
        asr_model.finish_streaming_transcribe, session.state,
        max(1, int((session.step + session.lookahead) / 1280)),
    )
    final = str(final).split("|")[0]
    tail = final[len(session.last_fixed_text):] if len(final) > len(session.last_fixed_text) else ""
    logger.info("[%s] finished: %r", session.request_id, final)
    await ws.send_text(json.dumps({
        "status": "success",
        "requestId": session.request_id,
        "msg": {"text": to_display(tail), "reset": True, "asr_cost_ms": 0.0,
                "total_cost_ms": 0.0},
    }, ensure_ascii=False))


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port, ws_ping_interval=None,
                log_level="warning")
