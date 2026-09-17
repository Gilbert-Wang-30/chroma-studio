"""Prompt -> palette.

``generate_palette`` turns a free-text prompt ("hawaii sunset", "navy and gold",
"stealth matte") into exactly ``n_colors`` named colors with weights summing to 1, plus
the reference photographs the colors were distilled from.  Precedence of sources:

1. ``parsed``  - the prompt names two or more explicit colors (``colornames.parse_color_words``);
   they come first, padded with a matched theme's colors and then image colors (method
   becomes ``mixed`` as soon as anything is appended).
2. ``theme``   - a curated theme keyword matched (``themes.match_theme``); theme colors
   first, then distinct image colors if more are needed.
3. ``images``  - k-means over Wikimedia Commons (or ddgs) thumbnails.
4. ``fallback``- a neutral+accent set when nothing else is available (offline, no match).

Results are cached under ``config.PALETTE_CACHE_DIR/<pid>/`` (``palette.json`` and
``sources/<i>.jpg``); ``pid = sha1(normalized prompt | n_colors)``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import warnings
from typing import Any, Callable

import numpy as np

from .. import colornames, config, imageio
from ..types import Palette, PaletteColor, PaletteSource
from . import sources as _sources
from .extract import extract_palette
from .themes import THEMES, match_theme

__all__ = [
    "generate_palette", "load_palette", "palette_id", "palette_dir", "colors_from_hexes",
    "FALLBACK_COLORS", "THEMES", "match_theme", "extract_palette",
]

log = logging.getLogger(__name__)

# A tasteful neutral + accent set used when there is nothing to work with.
FALLBACK_COLORS: list[str] = [
    "#2b2f36", "#f2f0eb", "#8a9096", "#c8102e", "#d4af37", "#1f5fb6", "#4b5320", "#e07a9a",
]
DISTINCT_DELTA_E = 9.0     # an *image* color this close to an existing pick is a duplicate
CURATED_DELTA_E = 5.0      # curated (theme / parsed) colors may be closer: tonal steps are designed
Progress = Callable[[float, str], None] | None


def _normalize_prompt(prompt: str) -> str:
    return " ".join(str(prompt).strip().lower().split())


def palette_id(prompt: str, n_colors: int) -> str:
    """Stable cache id: ``sha1("<normalized prompt>|<n>")`` as 40 hex chars."""
    return hashlib.sha1(f"{_normalize_prompt(prompt)}|{int(n_colors)}".encode("utf-8")).hexdigest()


def palette_dir(pid: str) -> str:
    """Cache directory for a palette id (read from ``config`` at call time)."""
    return os.path.join(config.PALETTE_CACHE_DIR, pid)


def load_palette(pid: str) -> Palette | None:
    """Read a cached palette by id; ``None`` when it does not exist or is unreadable."""
    path = os.path.join(palette_dir(pid), "palette.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return Palette.from_dict(json.load(f))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("unreadable palette cache %s: %s", path, exc)
        return None


def colors_from_hexes(hexes: list[str], weights: list[float] | None = None) -> list[PaletteColor]:
    """Build ``PaletteColor`` records from hex strings.  Weights default to a descending
    linear ramp (first color heaviest) and are normalized to sum to 1; names come from
    ``colornames.nearest_name``."""
    if not hexes:
        return []
    if weights is None:
        weights = [float(len(hexes) - i) for i in range(len(hexes))]
    total = float(sum(weights)) or 1.0
    out: list[PaletteColor] = []
    for h, w in zip(hexes, weights):
        rgb = imageio.hex_to_rgb01(h)
        hx = imageio.rgb01_to_hex(rgb)
        lab = imageio.hex_to_lab(hx)
        out.append(PaletteColor(hex=hx, lab=lab, weight=float(w) / total, name=colornames.nearest_name(lab)))
    return out


def _is_distinct(color: PaletteColor, chosen: list[PaletteColor], delta_e: float = DISTINCT_DELTA_E) -> bool:
    if not chosen:
        return True
    d = imageio.delta_e(np.repeat(np.asarray(color.lab, np.float32)[None], len(chosen), 0),
                        np.asarray([c.lab for c in chosen], np.float32))
    return bool(d.min() >= delta_e)


def _append_distinct(chosen: list[PaletteColor], candidates: list[PaletteColor], n: int,
                     delta_e: float = DISTINCT_DELTA_E) -> int:
    """Append candidates that are not near-duplicates (CIEDE2000 < ``delta_e``) until
    ``chosen`` has ``n``; returns how many were added."""
    added = 0
    for c in candidates:
        if len(chosen) >= n:
            break
        if _is_distinct(c, chosen, delta_e):
            chosen.append(c)
            added += 1
    return added


def _tints_and_shades(chosen: list[PaletteColor], n: int) -> list[PaletteColor]:
    """Last-resort padding: lighter/darker variants of the chosen colors, then hue
    rotations of the chromatic ones, with the distinctness threshold relaxed in a
    second pass so a very large ``n`` still gets mostly different colors."""
    out: list[PaletteColor] = []

    def _candidates():
        for step in range(1, 9):                            # tints and shades
            for base in list(chosen):
                for sign in (1.0, -1.0):
                    L = min(98.0, max(4.0, base.lab[0] + sign * 11.0 * step))
                    yield (L, base.lab[1] * 0.85, base.lab[2] * 0.85)
        for deg in (30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):   # hue turns
            rad = math.radians(deg)
            for base in list(chosen):
                a, b = base.lab[1], base.lab[2]
                if math.hypot(a, b) < 12.0:
                    continue
                yield (base.lab[0], a * math.cos(rad) - b * math.sin(rad),
                       a * math.sin(rad) + b * math.cos(rad))

    for delta_e in (DISTINCT_DELTA_E, CURATED_DELTA_E, 2.5):
        for lab in _candidates():
            if len(chosen) + len(out) >= n:
                return out
            with warnings.catch_warnings():
                # A rotated/shifted Lab may sit outside sRGB; lab_to_hex clips it to
                # the gamut and the color is re-derived from the hex, so the skimage
                # clipping notice is noise here.
                warnings.simplefilter("ignore", UserWarning)
                hx = imageio.lab_to_hex(lab)
            cand = colors_from_hexes([hx])[0]
            if _is_distinct(cand, chosen + out, delta_e):
                out.append(cand)
    return out


def _demote_padding(chosen: list[PaletteColor], n_measured: int) -> list[PaletteColor]:
    """Give every color appended after the first ``n_measured`` (measured image colors)
    a descending ramp of weights strictly below the smallest measured weight, so padding
    never outranks a color that was actually seen in the reference images."""
    if n_measured <= 0 or len(chosen) <= n_measured:
        return chosen
    floor = min(float(c.weight) for c in chosen[:n_measured])
    pads = chosen[n_measured:]
    k = len(pads)
    demoted = [PaletteColor(hex=c.hex, lab=c.lab, weight=0.5 * floor * (k - i) / k, name=c.name)
               for i, c in enumerate(pads)]
    return chosen[:n_measured] + demoted


def _finalize(chosen: list[PaletteColor], n: int, ranked: bool) -> list[PaletteColor]:
    """Trim to ``n`` and normalize weights.  ``ranked`` colors (theme/parsed lists)
    get a descending ramp so position expresses importance; otherwise the measured
    image weights are kept."""
    chosen = chosen[:n]
    if ranked:
        ws = [float(n - i) for i in range(len(chosen))]
    else:
        ws = [max(float(c.weight), 1e-4) for c in chosen]
    total = sum(ws) or 1.0
    return [PaletteColor(hex=c.hex, lab=c.lab, weight=w / total, name=c.name) for c, w in zip(chosen, ws)]


def _report(progress: Progress, frac: float, msg: str) -> None:
    if progress is not None:
        try:
            progress(float(frac), msg)
        except Exception:  # a UI callback must never break palette generation
            log.debug("progress callback failed", exc_info=True)


def generate_palette(prompt: str, n_colors: int = 6, max_images: int = 6,
                     progress: Progress = None, use_cache: bool = True) -> Palette:
    """Build (or load from cache) the palette for a prompt.

    Guarantees: always returns a ``Palette`` with exactly ``n_colors`` colors whose
    weights sum to 1 (within float rounding), ``method`` in
    ``images | theme | parsed | mixed | fallback`` and ``id == palette_id(prompt, n)``.
    Never raises on network failure: with no reachable image source the result comes
    from explicit color words, a matched theme, or ``FALLBACK_COLORS``.  Reference
    thumbnails are written to ``<cache>/sources/<i>.jpg`` and listed in ``sources``
    with ``thumb = /api/palettes/<pid>/sources/<i>.jpg``.  ``progress(frac, message)``
    is called with increasing ``frac`` in [0, 1].  A cached ``palette.json`` is
    returned as-is unless ``use_cache`` is false.  ``max_images = 0`` skips the network.
    """
    n = max(1, int(n_colors))
    pid = palette_id(prompt, n)
    if use_cache:
        cached = load_palette(pid)
        if cached is not None and len(cached.colors) == n:
            _report(progress, 1.0, "Loaded cached palette")
            return cached

    clean = " ".join(str(prompt).split())
    explicit_words = colornames.parse_color_words(clean)
    explicit: list[PaletteColor] = []
    for _, hx in explicit_words:
        cand = colors_from_hexes([hx])[0]
        if all(cand.hex != c.hex for c in explicit):
            explicit.append(cand)
    theme = match_theme(clean)
    theme_colors = colors_from_hexes(theme[1]) if theme else []

    # ---- reference images (skipped when explicit colors already fill the palette)
    cache = palette_dir(pid)
    src_dir = os.path.join(cache, "sources")
    fetched: list[tuple[dict[str, Any], np.ndarray]] = []
    want_images = int(max_images) > 0 and clean and not (len(explicit) >= 2 and len(explicit) >= n)
    if want_images:
        _report(progress, 0.05, "Searching Wikimedia Commons")
        try:
            items = _sources.search_images(clean, limit=int(max_images))
        except Exception as exc:
            log.warning("image search failed for %r: %s", clean, exc)
            items = []
        if items:
            _report(progress, 0.3, f"Downloading {len(items)} reference images")
            try:
                fetched = _sources.fetch_thumbs(items, src_dir, width=640)
            except Exception as exc:
                log.warning("thumb download failed for %r: %s", clean, exc)
                fetched = []
    image_colors: list[PaletteColor] = []
    if fetched:
        _report(progress, 0.7, f"Extracting colors from {len(fetched)} images")
        try:
            image_colors = extract_palette([img for _, img in fetched], n)
        except Exception as exc:
            log.warning("palette extraction failed for %r: %s", clean, exc)
            image_colors = []

    # ---- assemble by precedence
    chosen: list[PaletteColor] = []
    ranked = True
    if len(explicit) >= 2:
        method = "parsed"
        _append_distinct(chosen, explicit, n, CURATED_DELTA_E)
        added = _append_distinct(chosen, theme_colors, n, CURATED_DELTA_E)
        added += _append_distinct(chosen, image_colors, n)
        if added:
            method = "mixed"
    elif theme is not None:
        method = "theme"
        _append_distinct(chosen, theme_colors, n, CURATED_DELTA_E)
        if _append_distinct(chosen, image_colors, n):
            method = "mixed"
    elif image_colors:
        method = "images"
        ranked = False
        if explicit:                       # a single color word is honoured, then images
            _append_distinct(chosen, explicit, n, CURATED_DELTA_E)
            ranked = True
            method = "mixed"
        _append_distinct(chosen, image_colors, n)
    elif explicit:
        method = "parsed"
        _append_distinct(chosen, explicit, n, CURATED_DELTA_E)
    else:
        method = "fallback"

    # ---- pad to exactly n: theme, fallback set, then tints/shades/hue turns
    n_measured = len(chosen) if not ranked else 0     # image colors carry real weights
    if len(chosen) < n and theme_colors:
        _append_distinct(chosen, theme_colors, n, CURATED_DELTA_E)
    if len(chosen) < n:
        if not chosen:
            method = "fallback"
        _append_distinct(chosen, colors_from_hexes(FALLBACK_COLORS), n)
    if len(chosen) < n:
        chosen.extend(_tints_and_shades(chosen, n))
    if len(chosen) < n:                    # degenerate (n in the hundreds): Lab space is
        log.warning("palette %r: only %d distinct colors for n=%d; repeating", clean, len(chosen), n)
        while len(chosen) < n:             # exhausted, repeat the set rather than fail
            chosen.append(chosen[len(chosen) % max(1, len(chosen))])
    if not ranked:
        # Measured image weights stay; padding is demoted below the smallest of them so
        # the palette is weight-descending and `area` mapping ranks real colors first.
        chosen = _demote_padding(chosen, n_measured)
    colors = _finalize(chosen, n, ranked)

    srcs = [PaletteSource(url=it.get("page_url") or it.get("url", ""), title=it.get("title", ""),
                          license=it.get("license", "unknown"), thumb=f"/api/palettes/{pid}/sources/{i}.jpg")
            for i, (it, _) in enumerate(fetched)]
    pal = Palette(id=pid, prompt=clean, colors=colors, sources=srcs, method=method, created=time.time())
    try:
        os.makedirs(cache, exist_ok=True)
        with open(os.path.join(cache, "palette.json"), "w", encoding="utf-8") as f:
            json.dump(pal.to_dict(), f, indent=1)
    except OSError as exc:
        log.warning("could not write palette cache %s: %s", cache, exc)
    _report(progress, 1.0, f"Palette ready ({method})")
    return pal
