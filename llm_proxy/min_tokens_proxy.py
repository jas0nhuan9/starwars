# coding=utf-8
"""An OpenAI-compatible shim that gives llama.cpp vLLM's ``min_tokens``.

T3PO's streaming protocol treats an empty completion as ``WAIT``: the model
emits EOS as its first token when it wants more source text.  To *force* a
segment -- at a sentence end, at the buffer limit, or on the final flush -- it
asks the endpoint for at least one real token via vLLM's ``min_tokens``.
llama.cpp silently ignores that field, so every forced segment came back empty
and was read as another WAIT: subtitles never appeared at all.

This proxy reinstates the semantics with two requests, which is what
``min_tokens=1`` means:

1. one token with EOS suppressed, so the model cannot answer WAIT;
2. the rest with EOS allowed again, continuing from that token through
   llama.cpp's assistant prefill, so the segment still ends where the model
   wants it to.

Step 2 re-sends a prompt that is one token longer than step 1, so llama.cpp's
prefix cache serves all of it but that token.  Requests without ``min_tokens``
are forwarded untouched, which keeps the probe path -- the overwhelming
majority of calls -- byte-identical and cache-friendly.

Run it between T3PO and llama-server:

    T3PO (.env VLLM_BASE_URL) -> this proxy :8011 -> llama-server :8010
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zh_script import to_display, to_model  # noqa: E402

UPSTREAM = os.getenv("LLM_UPSTREAM", "http://127.0.0.1:8010").rstrip("/")
HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PORT = int(os.getenv("PROXY_PORT", "8011"))
TIMEOUT = float(os.getenv("PROXY_TIMEOUT_SECONDS", "300"))

app = FastAPI(title="min_tokens shim for llama.cpp")
_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=UPSTREAM,
            timeout=httpx.Timeout(TIMEOUT, connect=15.0),
        )
    return _client


def _to_model_script(payload: dict) -> None:
    """Put the prompt into the script T3PO was trained on, in place.

    The prompt carries the interleaved history of everything already shown, so
    when the display script is Traditional this is what keeps the model on
    Simplified for both the history and the new input.
    """

    for message in payload.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = to_model(content)


def _respond(response: httpx.Response) -> Response:
    """Forward one upstream reply, converted back to the display script."""

    body = response.content
    if response.status_code < 400:
        try:
            parsed = response.json()
            text = _content_of(parsed)
            if text:
                parsed["choices"][0]["message"]["content"] = to_display(text)
                body = _dump(parsed)
        except Exception:
            pass  # not a completion we recognise; forward it untouched
    return Response(
        content=body,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


def _content_of(body: dict) -> str:
    try:
        return str(body["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        return ""


async def _post(path: str, payload: dict, headers: dict) -> httpx.Response:
    return await client().post(path, json=payload, headers=headers)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload: dict[str, Any] = await request.json()
    headers = {"content-type": "application/json"}
    if auth := request.headers.get("authorization"):
        headers["authorization"] = auth

    min_tokens = payload.pop("min_tokens", 0) or 0
    # vLLM's name for it; llama.cpp calls it repeat_penalty.
    if "repetition_penalty" in payload:
        payload["repeat_penalty"] = payload.pop("repetition_penalty")

    try:
        min_tokens = int(min_tokens)
    except (TypeError, ValueError):
        min_tokens = 0

    _to_model_script(payload)

    if min_tokens < 1:
        return _respond(await _post("/v1/chat/completions", payload, headers))

    # --- forced segment: EOS must not win the first token ---
    probe = dict(payload)
    probe["max_tokens"] = 1
    probe["ignore_eos"] = True
    first_response = await _post("/v1/chat/completions", probe, headers)
    if first_response.status_code >= 400:
        return _respond(first_response)
    first = _content_of(first_response.json())
    if not first:
        # Nothing to build on (an empty token, or a model that answered
        # nothing even with EOS suppressed); let the caller see that.
        return _respond(first_response)

    follow = dict(payload)
    follow["messages"] = list(payload.get("messages") or []) + [
        {"role": "assistant", "content": first}
    ]
    second_response = await _post("/v1/chat/completions", follow, headers)
    if second_response.status_code >= 400:
        # The first token alone is still a valid forced segment.
        return _respond(first_response)

    body = second_response.json()
    text = _content_of(body)
    # llama.cpp returns the prefilled message in full, but do not depend on it.
    # Both parts are still in the model's script here; the display conversion
    # happens once, after they are joined.
    if not text.startswith(first):
        text = first + text
    try:
        body["choices"][0]["message"]["content"] = to_display(text)
    except (KeyError, IndexError, TypeError):
        pass
    return Response(content=_dump(body), media_type="application/json")


def _dump(body: dict) -> bytes:
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


@app.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
async def passthrough(path: str, request: Request) -> Response:
    """Everything else (``/v1/models``, ``/health``, ...) goes straight on."""

    headers = {}
    if auth := request.headers.get("authorization"):
        headers["authorization"] = auth
    body = await request.body()
    if request.method == "GET":
        upstream = await client().get(f"/{path}", headers=headers)
    else:
        headers["content-type"] = request.headers.get("content-type", "application/json")
        upstream = await client().request(
            request.method, f"/{path}", content=body, headers=headers
        )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
