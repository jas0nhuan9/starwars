# coding=utf-8
"""T3PO's own web UI, with its interface text in the display script.

The upstream UI is written in Simplified Chinese.  Rather than edit files in
``vendor/`` -- which would collide the next time upstream is updated -- the
three static assets are converted on the way out and served ahead of the
catch-all ``StaticFiles`` mount.

The routes are inserted into the upstream app rather than mounting it inside a
wrapper: its lifespan runs the session-reaper task, and a mounted sub-app does
not receive lifespan events.  Inserting at the front of the router is what puts
these ahead of the mount, since Starlette matches routes in order and the mount
matches everything.

Only display text is converted.  The UI's Chinese never takes part in any
comparison -- ``value="zh2en"`` and friends are ASCII -- so converting whole
files cannot change behaviour, and JavaScript syntax is ASCII throughout.

``ZH_VARIANT=simplified`` skips the whole thing and serves upstream untouched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

from starlette.responses import Response
from starlette.routing import Route

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "Confucius4-T3PO"))

from inference.server import STATIC_DIR, app  # noqa: E402  (vendor app, lifespan intact)
from zh_script import to_display, variant  # noqa: E402

# The product name appears only in index.html's <title> and <h1>; app.js never
# mentions it and nothing sends it to the server, so renaming it here cannot
# affect behaviour.  The model id the requests actually carry lives in
# config/t3po.env.
UI_TITLE = os.getenv("UI_TITLE", "Star Wars").strip() or "Star Wars"
UPSTREAM_TITLE = "Confucius4-T3PO"

MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}

# Request path -> file in upstream's static directory.
OVERRIDES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/style.css": "style.css",
}

_cache: dict[str, tuple[float, bytes]] = {}


def _render(name: str) -> bytes:
    """Convert one asset, re-reading it when upstream's copy changes."""

    path = STATIC_DIR / name
    stamp = path.stat().st_mtime
    cached = _cache.get(name)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    text = to_display(path.read_text(encoding="utf-8"))
    if name.endswith(".html"):
        # Keep the document's declared locale honest for screen readers and
        # for the font stack the browser picks.
        text = text.replace('lang="zh-CN"', 'lang="zh-TW"')
        if UI_TITLE != UPSTREAM_TITLE:
            text = text.replace(UPSTREAM_TITLE, UI_TITLE)
    body = text.encode("utf-8")
    _cache[name] = (stamp, body)
    return body


def _endpoint(name: str) -> Callable[..., Any]:
    media_type = MEDIA_TYPES[Path(name).suffix]

    async def serve(_request: Any) -> Response:
        return Response(content=_render(name), media_type=media_type)

    return serve


def _install() -> None:
    if UI_TITLE != UPSTREAM_TITLE:
        # Also renames it in the OpenAPI docs, so the two agree.
        app.title = UI_TITLE
    if variant() == "simplified":
        return
    for path, name in OVERRIDES.items():
        app.router.routes.insert(
            0, Route(path, _endpoint(name), methods=["GET", "HEAD"])
        )


_install()
