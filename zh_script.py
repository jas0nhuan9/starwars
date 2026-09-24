# coding=utf-8
"""Chinese script conversion at the edges of the pipeline.

Both models were trained on Simplified Chinese: R2T2 transcribes in it and
T3PO translates into it.  A Traditional-Chinese user wants to read Traditional,
but feeding Traditional back into the models puts them off the distribution
their streaming behaviour was tuned on -- T3PO's interleaved history in
particular is re-sent on every chunk.

So the script is converted at the boundaries instead, and the models keep
seeing Simplified:

* :func:`to_display` -- on text leaving for a human (ASR increments, finished
  translations).  ``s2twp`` also applies Taiwanese vocabulary, so 软件 reads as
  軟體 rather than 軟件.
* :func:`to_model` -- on text going into a model (T3PO's prompt, which carries
  the history of everything already displayed).  ``tw2sp`` reverses the
  vocabulary mapping too, so the round trip is lossless.

``ZH_VARIANT=simplified`` turns both into no-ops.  If OpenCC is missing the
functions pass text through unchanged and say so once, so a missing optional
dependency degrades the script rather than breaking the pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("zh-script")

# OpenCC configuration names, chosen as a matched pair so display -> model ->
# display is lossless, including the vocabulary substitutions.
DISPLAY_CONFIG = "s2twp"   # Simplified -> Traditional, Taiwanese vocabulary
MODEL_CONFIG = "tw2sp"     # the exact inverse

_converters: dict[str, Any] = {}
_warned = False


def variant() -> str:
    """``traditional`` (the default) or ``simplified``."""

    value = os.getenv("ZH_VARIANT", "traditional").strip().lower()
    return "simplified" if value in {"simplified", "s", "zh-cn", "cn", "off"} else "traditional"


def _converter(config: str) -> Optional[Any]:
    global _warned
    if config in _converters:
        return _converters[config]
    try:
        from opencc import OpenCC

        _converters[config] = OpenCC(config)
    except Exception:
        if not _warned:
            logger.warning(
                "OpenCC is unavailable, so Chinese text is passed through in the "
                "models' own script; install opencc-python-reimplemented for %s",
                variant(),
            )
            _warned = True
        _converters[config] = None
    return _converters[config]


def _convert(text: str, config: str) -> str:
    if not text or variant() == "simplified":
        return text
    converter = _converter(config)
    if converter is None:
        return text
    try:
        return converter.convert(text)
    except Exception:
        return text


def to_display(text: str) -> str:
    """Convert text that a person is about to read.

    Note for streaming callers: this is applied per increment, so a
    vocabulary mapping whose phrase straddles two increments is missed and the
    characters are converted individually.  That costs the occasional 軟件 for
    軟體 in the live transcript; the translation path converts whole segments
    and is unaffected.
    """

    return _convert(text, DISPLAY_CONFIG)


def to_model(text: str) -> str:
    """Convert text that a model is about to read."""

    return _convert(text, MODEL_CONFIG)
