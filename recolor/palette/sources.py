"""Reference-image search for prompt palettes.

Primary source is the Wikimedia Commons search API (no key needed, descriptive
User-Agent).  When Commons returns fewer than ``MIN_COMMONS_RESULTS`` usable hits the
optional ``ddgs`` package is tried as a second source.  Every public function here is
network-tolerant: a failed request, a timeout or a missing package yields fewer (or
zero) results, never an exception, so the palette stage can always fall back.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .. import imageio

log = logging.getLogger(__name__)

USER_AGENT = "chroma-studio/0.1 (https://github.com/Gilbert-Wang-30/chroma-studio)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
SEARCH_TIMEOUT_S = 8.0
FETCH_TIMEOUT_S = 10.0
MIN_COMMONS_RESULTS = 3
PREFERRED_MIN_SIDE = 800           # results at least this wide/tall are ranked first
MAX_FETCH_BYTES = 12_000_000       # hard cap per download (thumb or original)
MAX_ORIGINAL_PIXELS = 24_000_000   # skip the original-file fallback above this (w*h)
ACCEPTED_MIME = {"image/jpeg", "image/png"}
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(s: str) -> str:
    return html.unescape(_TAG_RE.sub("", s or "")).strip()


def _http_get(url: str, timeout: float, max_bytes: int | None = None) -> bytes:
    """GET ``url`` and return the body.  With ``max_bytes`` a response whose declared
    Content-Length or actual body exceeds the cap raises ``ValueError`` after reading at
    most ``max_bytes + 1`` bytes, so a multi-megabyte original can never stall a palette
    request for its whole download."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if max_bytes is None:
            return resp.read()
        declared = resp.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise ValueError(f"response too large: {declared} bytes > {max_bytes}")
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"response too large: > {max_bytes} bytes")
        return data


def _search_commons(prompt: str, limit: int, thumb_width: int) -> list[dict[str, Any]]:
    """Commons ``generator=search`` restricted to bitmap files with 640 px thumbs."""
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {prompt}",
        "gsrnamespace": 6,
        "gsrlimit": max(1, min(int(limit), 50)),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": int(thumb_width),
        "format": "json",
    }
    url = COMMONS_API + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    data = json.loads(_http_get(url, SEARCH_TIMEOUT_S).decode("utf-8"))
    pages = (data.get("query") or {}).get("pages") or {}
    items: list[dict[str, Any]] = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 1 << 30)):
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        mime = info.get("mime", "")
        if mime not in ACCEPTED_MIME:
            continue
        meta = info.get("extmetadata") or {}
        title = _strip_tags(page.get("title", "")).removeprefix("File:")
        title = os.path.splitext(title)[0]
        items.append({
            "url": info.get("url", ""),
            "thumb_url": info.get("thumburl") or info.get("url", ""),
            "page_url": info.get("descriptionurl", ""),
            "title": title,
            "license": _strip_tags((meta.get("LicenseShortName") or {}).get("value", "")) or "unknown",
            "artist": _strip_tags((meta.get("Artist") or {}).get("value", "")),
            "width": int(info.get("width") or 0),
            "height": int(info.get("height") or 0),
            "mime": mime,
            "source": "commons",
        })
    return items


def _search_ddgs(prompt: str, limit: int) -> list[dict[str, Any]]:
    """DuckDuckGo image search via the optional ``ddgs`` package."""
    try:
        from ddgs import DDGS  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return []
    results = DDGS(timeout=SEARCH_TIMEOUT_S).images(prompt, max_results=max(1, int(limit)))
    items: list[dict[str, Any]] = []
    for r in results or []:
        image = r.get("image") or ""
        if not image:
            continue

        def _int(v: Any) -> int:
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0

        items.append({
            "url": image,
            "thumb_url": r.get("thumbnail") or image,
            "page_url": r.get("url") or image,
            "title": _strip_tags(r.get("title") or ""),
            "license": "unknown (web image)",
            "artist": _strip_tags(r.get("source") or ""),
            "width": _int(r.get("width")),
            "height": _int(r.get("height")),
            "mime": "image/jpeg",
            "source": "ddgs",
        })
    return items


def search_images(prompt: str, limit: int, thumb_width: int = 640) -> list[dict[str, Any]]:
    """Find reference photographs for a prompt.

    Guarantees: returns at most ``limit`` dicts with at least the keys
    ``url, page_url, title, license, width, height`` (plus ``thumb_url, artist, mime,
    source``), large images (min side >= 800 px) first, duplicates by URL removed.  Never
    raises: network or parsing failures are logged and give fewer results, possibly an
    empty list.  Commons is queried first; ``ddgs`` is used only when Commons yields
    fewer than ``MIN_COMMONS_RESULTS`` usable hits and only if the package is installed.
    """
    prompt = " ".join(str(prompt).split())
    limit = max(0, int(limit))
    if not prompt or limit == 0:
        return []
    items: list[dict[str, Any]] = []
    try:
        items = _search_commons(prompt, limit * 2, thumb_width)
    except Exception as exc:  # network, JSON, HTTP errors
        log.warning("Commons search failed for %r: %s", prompt, exc)
    if len(items) < MIN_COMMONS_RESULTS:
        try:
            items.extend(_search_ddgs(prompt, limit))
        except Exception as exc:
            log.warning("ddgs search failed for %r: %s", prompt, exc)
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for it in items:
        key = it.get("url") or it.get("thumb_url")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(it)
    # Stable sort: preferred (large) images first, original ranking otherwise.
    unique.sort(key=lambda it: 0 if min(it["width"], it["height"]) >= PREFERRED_MIN_SIDE else 1)
    return unique[:limit]


def _fetch_one(item: dict[str, Any], width: int) -> np.ndarray | None:
    """Download one reference image: the thumbnail first, then the original as a
    fallback unless its declared size suggests a huge file (``MAX_ORIGINAL_PIXELS``).
    Both downloads are capped at ``MAX_FETCH_BYTES``; ``None`` when nothing works."""
    thumb, orig = item.get("thumb_url"), item.get("url")
    urls = [thumb] if thumb else []
    if orig and orig != thumb:
        w, h = int(item.get("width") or 0), int(item.get("height") or 0)
        if w * h <= MAX_ORIGINAL_PIXELS:
            urls.append(orig)
    for url in urls:
        try:
            raw = _http_get(url, FETCH_TIMEOUT_S, max_bytes=MAX_FETCH_BYTES)
            img = imageio.load_image(raw, max_long_side=width)
            if img.ndim == 3 and img.shape[0] >= 32 and img.shape[1] >= 32:
                return img
        except Exception as exc:
            log.info("thumb fetch failed %s: %s", url, exc)
    return None


def fetch_thumbs(items: list[dict[str, Any]], cache_dir: str | None, width: int = 640,
                 max_workers: int = 6) -> list[tuple[dict[str, Any], np.ndarray]]:
    """Download the thumbnails of ``items`` in parallel.

    Guarantees: returns ``(item, image)`` pairs in the order of ``items`` for every
    download that succeeded (failures are skipped, never raised); each image is uint8
    RGB with long side <= ``width``.  When ``cache_dir`` is given it is created and the
    i-th *returned* image is written to ``<cache_dir>/<i>.jpg`` so the indices match the
    palette's ``sources`` list.
    """
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(items)))) as pool:
        images = list(pool.map(lambda it: _fetch_one(it, width), items))
    out = [(it, img) for it, img in zip(items, images) if img is not None]
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for i, (_, img) in enumerate(out):
            imageio.save_image(os.path.join(cache_dir, f"{i}.jpg"), img, quality=88)
    return out
