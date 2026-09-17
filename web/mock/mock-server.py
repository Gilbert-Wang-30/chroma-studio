#!/usr/bin/env python
"""Mock API server for Chroma Studio UI development.

Serves ``web/`` as static files and fakes every route of docs/ARCHITECTURE.md §3.7 with
stdlib ``http.server`` only (plus PIL/numpy/skimage for the fake analysis). Nothing here
touches the GPU or the real pipeline; state lives in memory and is lost on exit.

    .venv/bin/python web/mock/mock-server.py [--port 8791] [--tick 600]

What is faked and how:

* samples  - ``samples/*.jpg`` resized on demand with PIL (cached per width)
* analysis - a stepper that advances every ``--tick`` ms through the five stages;
             regions are SLIC superpixels merged by colour, groups are an agglomerative
             clustering of their Lab means, so hover/select/merge/split feel real
* layers   - albedo/shading/residual are cheap blur-based approximations
* ids      - ``ids/regions.png`` (id = R + 256 G + 65536 B) and ``ids/groups.png`` (R = gid)
* palettes - colour words from the prompt, a few curated themes, else a seeded pick from
             ``recolor.colornames.NAMED``; sources are sample images with their attribution
* render   - per-group tint of the preview that keeps the original luminance texture
* export   - the same render at work/original resolution, kept in memory
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
import uuid
from email.utils import formatdate
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB_DIR = os.path.join(ROOT, "web")
SAMPLES_DIR = os.path.join(ROOT, "samples")
sys.path.insert(0, ROOT)

from recolor.colornames import NAMED, hue_family, nearest_name, parse_color_words  # noqa: E402
from recolor.imageio import hex_to_lab, lab_to_hex, rgb_to_lab  # noqa: E402

WORK_LONG_SIDE = 1536
PREVIEW_LONG_SIDE = 1024
STAGES = ["ingest", "intrinsic", "segment", "regions", "groups"]
STAGE_MESSAGES: dict[str, list[str]] = {
    "ingest": ["Reading pixels", "Making working and preview images"],
    "intrinsic": ["Loading Intrinsic v2.1", "Separating albedo from shading", "Recovering speculars"],
    "segment": ["Finding parts with SAM 2 · 32 points/side", "Finding parts with SAM 2 · crop layer 1/2",
                "Finding parts with SAM 2 · crop layer 2/2", "Scoring 214 mask proposals"],
    "regions": ["Painting masks largest-first", "Splitting two-tone parts", "Filling gaps with superpixels",
                "Removing specks"],
    "groups": ["Measuring albedo per region", "Clustering by colour (ΔE 10)", "Naming groups"],
}
TICK_MS = 600
MIME = {
    ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8", ".json": "application/json",
    ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8",
    ".woff2": "font/woff2", ".map": "application/json",
}

THEMES: dict[str, list[str]] = {
    "hawaii sunset": ["#ff5e5b", "#ff9f1c", "#ffd166", "#ef476f", "#3a86ff", "#073b4c"],
    "stealth matte": ["#141414", "#2c3539", "#4a4a4a", "#6b7280", "#9aa3ad", "#b8c0c8"],
    "sakura": ["#f9c8d6", "#ffb7c5", "#f4a6c1", "#e8637a", "#fffdd0", "#8e4585"],
    "racing livery": ["#d62828", "#fafafa", "#101010", "#f7d51d", "#1f5fd6", "#c0c0c0"],
    "cyberpunk": ["#fcee0a", "#ff1dce", "#00e5e5", "#191970", "#0a0a0a", "#8f00ff"],
    "desert camo": ["#c19a6b", "#e2c290", "#5a6b3f", "#6b5433", "#c3b091", "#4b5320"],
    "ocean": ["#0f2c7c", "#1d6fa5", "#40e0d0", "#87ceeb", "#f4f1ea", "#2e8b57"],
    "forest": ["#004225", "#228b22", "#6b8e23", "#7b4a2d", "#c2b280", "#f2f0eb"],
    "candy": ["#ff69b4", "#7fdd2a", "#00e5e5", "#fff44f", "#b57edc", "#ff7f50"],
    "autumn": ["#b7410e", "#cd7f32", "#e1ad01", "#7a1f2b", "#5c3a21", "#f5f5dc"],
}


# ----------------------------------------------------------------------------- helpers

def _now() -> float:
    return time.time()


def fit_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    """(w, h) scaled so the longer side equals ``long_side``; never upscales."""
    s = min(1.0, long_side / max(w, h))
    return max(1, round(w * s)), max(1, round(h * s))


def encode_jpeg(arr: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def encode_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", compress_level=3)
    return buf.getvalue()


def resize_u8(arr: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    im = Image.fromarray(arr)
    im = im.resize(size, Image.NEAREST if nearest else Image.LANCZOS)
    return np.asarray(im)


def _hex_rgb01(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


# ----------------------------------------------------------------------------- samples

class Samples:
    """Sample images and their resized/cached encodings."""

    def __init__(self) -> None:
        with open(os.path.join(SAMPLES_DIR, "MANIFEST.json"), encoding="utf-8") as f:
            self.manifest: dict[str, dict] = json.load(f)
        self._cache: dict[tuple[str, int], bytes] = {}
        self._lock = threading.Lock()

    def names(self) -> list[str]:
        return [n for n in self.manifest if os.path.exists(os.path.join(SAMPLES_DIR, n))]

    def listing(self) -> list[dict]:
        out = []
        for n in self.names():
            m = self.manifest[n]
            with Image.open(os.path.join(SAMPLES_DIR, n)) as im:   # header only, no decode
                w, h = im.size
            out.append({
                "name": n, "url": f"/api/samples/{n}", "thumb": f"/api/samples/{n}?w=320",
                "width": w, "height": h,
                "title": re.sub(r"^File:", "", m.get("title", n)).rsplit(".", 1)[0],
                "license": m.get("license", ""), "artist": m.get("artist", ""),
                "source": m.get("source", ""),
            })
        return out

    def load(self, name: str, long_side: Optional[int] = None) -> np.ndarray:
        im = Image.open(os.path.join(SAMPLES_DIR, name)).convert("RGB")
        if long_side:
            im = im.resize(fit_size(*im.size, long_side), Image.LANCZOS)
        return np.asarray(im)

    def encoded(self, name: str, width: Optional[int]) -> bytes:
        key = (name, width or 0)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if width:
            data = encode_jpeg(self.load(name, width), 85)
        else:
            with open(os.path.join(SAMPLES_DIR, name), "rb") as f:
                data = f.read()
        with self._lock:
            self._cache[key] = data
        return data


# ----------------------------------------------------------------------------- fake analysis

def superpixels(work: np.ndarray, detail: str) -> tuple[np.ndarray, np.ndarray]:
    """SLIC superpixels of the working image.

    Returns ``(labels int32 HxW in 0..S-1, mean Lab per superpixel Sx3)``.
    """
    from skimage.segmentation import slic
    n = {"fast": 160, "balanced": 320, "max": 640}.get(detail, 320)
    labels = slic(work, n_segments=n, compactness=18, sigma=1, start_label=0, channel_axis=-1)
    labels = labels.astype(np.int32)
    # Compact the label range (SLIC can skip ids).
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(work.shape[:2]).astype(np.int32)
    s = int(labels.max()) + 1
    lab = rgb_to_lab(work.astype(np.float32) / 255.0).reshape(-1, 3)
    flat = labels.ravel()
    sums = np.zeros((s, 3), np.float64)
    np.add.at(sums, flat, lab)
    counts = np.bincount(flat, minlength=s).astype(np.float64)
    return labels, (sums / counts[:, None]).astype(np.float32)


def cluster_lab(lab: np.ndarray, weights: np.ndarray, delta_e: float, max_groups: Optional[int]) -> np.ndarray:
    """Agglomerative clustering of Lab points into group ids 0..G-1 (area desc)."""
    from sklearn.cluster import AgglomerativeClustering
    n = len(lab)
    if n == 1:
        return np.zeros(1, np.int32)
    ac = AgglomerativeClustering(n_clusters=None, distance_threshold=max(4.0, delta_e * 1.6), linkage="average")
    ids = ac.fit_predict(lab)
    g = int(ids.max()) + 1
    cap = max_groups or 12
    if g > cap:
        ac = AgglomerativeClustering(n_clusters=cap, linkage="average")
        ids = ac.fit_predict(lab)
    return relabel_by_area(ids.astype(np.int32), weights)


def relabel_by_area(ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    g = int(ids.max()) + 1
    area = np.bincount(ids, weights=weights, minlength=g)
    order = np.argsort(-area)
    remap = np.empty(g, np.int32)
    remap[order] = np.arange(g, dtype=np.int32)
    return remap[ids]


def adjacency_pairs(labels: np.ndarray) -> np.ndarray:
    """Unique (a, b) pairs of superpixel ids that touch (4-connectivity)."""
    h = np.stack([labels[:, :-1].ravel(), labels[:, 1:].ravel()], 1)
    v = np.stack([labels[:-1, :].ravel(), labels[1:, :].ravel()], 1)
    p = np.concatenate([h, v])
    p = p[p[:, 0] != p[:, 1]]
    p = np.sort(p, 1)
    return np.unique(p, axis=0)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = np.arange(n)

    def find(self, a: int) -> int:
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return int(a)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


class Analysis:
    """Everything derived from one image, mutable through the group edits."""

    def __init__(self, work: np.ndarray, detail: str, max_groups: Optional[int], delta_e: float) -> None:
        self.work = work
        self.h, self.w = work.shape[:2]
        self.detail = detail
        self.sp_labels, self.sp_lab = superpixels(work, detail)
        self.sp_area = np.bincount(self.sp_labels.ravel(), minlength=len(self.sp_lab)).astype(np.float64)
        self.pairs = adjacency_pairs(self.sp_labels)
        border = np.zeros(self.sp_labels.shape, bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        self.sp_border = np.bincount(self.sp_labels[border], minlength=len(self.sp_lab)).astype(np.float64)
        self.sp_group = cluster_lab(self.sp_lab, self.sp_area, delta_e, max_groups)
        self.sp_region = np.zeros(len(self.sp_lab), np.int32)
        self.regions: list[dict] = []
        self.groups: list[dict] = []
        self.meta_overrides: dict[int, dict] = {}   # gid -> {name?, locked?, is_background?}
        self.next_region_id = 0
        self.rebuild(first=True)
        # Cached rasters (invalidated on rebuild)
        self._cache: dict[str, bytes] = {}

    # ----- structure

    def rebuild(self, first: bool = False) -> None:
        """Recompute regions (connected same-group superpixels) and groups, keeping
        region ids stable where a new region mostly overlaps an old one."""
        s = len(self.sp_lab)
        uf = UnionFind(s)
        same = self.sp_group[self.pairs[:, 0]] == self.sp_group[self.pairs[:, 1]]
        for a, b in self.pairs[same]:
            uf.union(int(a), int(b))
        roots = np.array([uf.find(i) for i in range(s)])
        _, comp = np.unique(roots, return_inverse=True)
        n_comp = int(comp.max()) + 1

        # Stable ids: majority old region id per component, unless already claimed.
        new_ids = np.full(n_comp, -1, np.int32)
        if not first:
            claimed: set[int] = set()
            for c in np.argsort(-np.bincount(comp, weights=self.sp_area, minlength=n_comp)):
                members = np.nonzero(comp == c)[0]
                olds = self.sp_region[members]
                cand = np.bincount(olds, weights=self.sp_area[members])
                best = int(np.argmax(cand))
                if best not in claimed:
                    new_ids[c] = best
                    claimed.add(best)
        for c in range(n_comp):
            if new_ids[c] < 0:
                new_ids[c] = self.next_region_id
                self.next_region_id += 1
        self.next_region_id = max(self.next_region_id, int(new_ids.max()) + 1)
        self.sp_region = new_ids[comp]

        # Old group -> flags, carried to the new group with the largest overlap.
        old_flags = {g["id"]: g for g in self.groups}
        old_sp_group = getattr(self, "_prev_sp_group", None)

        self.sp_group = relabel_by_area(self.sp_group, self.sp_area)
        total = float(self.h * self.w)
        regions: list[dict] = []
        ys, xs = np.mgrid[0:self.h, 0:self.w]
        # Bboxes per superpixel then per region.
        sp_x0 = np.full(s, self.w, np.int64); sp_x1 = np.zeros(s, np.int64)
        sp_y0 = np.full(s, self.h, np.int64); sp_y1 = np.zeros(s, np.int64)
        np.minimum.at(sp_x0, self.sp_labels.ravel(), xs.ravel())
        np.maximum.at(sp_x1, self.sp_labels.ravel(), xs.ravel())
        np.minimum.at(sp_y0, self.sp_labels.ravel(), ys.ravel())
        np.maximum.at(sp_y1, self.sp_labels.ravel(), ys.ravel())
        for rid in np.unique(self.sp_region):
            members = np.nonzero(self.sp_region == rid)[0]
            area = float(self.sp_area[members].sum())
            lab = (self.sp_lab[members] * self.sp_area[members, None]).sum(0) / area
            regions.append({
                "id": int(rid), "area": int(area),
                "bbox": [int(sp_x0[members].min()), int(sp_y0[members].min()),
                         int(sp_x1[members].max()) + 1, int(sp_y1[members].max()) + 1],
                "albedo_lab": [round(float(v), 2) for v in lab], "albedo_hex": lab_to_hex(lab),
                "group_id": int(self.sp_group[members[0]]),
                "touches_border": bool(self.sp_border[members].sum() > 0),
                "source": "sam" if area > 0.004 * total else "superpixel",
                "confidence": round(0.78 + 0.2 * (hash((int(rid), 7)) % 100) / 100.0, 3),
            })
        regions.sort(key=lambda r: r["id"])
        self.regions = regions

        g_count = int(self.sp_group.max()) + 1
        groups: list[dict] = []
        border_total = float(2 * (self.w + self.h))
        border_frac = np.zeros(g_count)
        for gid in range(g_count):
            members = np.nonzero(self.sp_group == gid)[0]
            area = float(self.sp_area[members].sum())
            lab = (self.sp_lab[members] * self.sp_area[members, None]).sum(0) / area
            border_frac[gid] = self.sp_border[members].sum() / border_total
            groups.append({
                "id": gid, "name": nearest_name(lab),
                "albedo_lab": [round(float(v), 2) for v in lab], "albedo_hex": lab_to_hex(lab),
                "area": int(area), "area_frac": round(area / total, 5),
                "region_ids": sorted({r["id"] for r in regions if r["group_id"] == gid}),
                "hue_family": hue_family(lab), "locked": False, "is_background": False,
            })
        bg = int(np.argmax(border_frac))
        if border_frac[bg] > 0.35:
            groups[bg]["is_background"] = True
        # Carry user flags across renumbering by overlap; only one group stays background.
        if old_sp_group is not None and old_flags:
            for g in groups:
                members = self.sp_group == g["id"]
                olds = old_sp_group[members]
                best = int(np.argmax(np.bincount(olds, weights=self.sp_area[members])))
                o = old_flags.get(best)
                if o:
                    g["locked"] = o["locked"]
                    g["is_background"] = o["is_background"]
                    if o.get("_custom_name"):
                        g["name"] = o["name"]
                        g["_custom_name"] = True
            bgs = [g for g in groups if g["is_background"]]
            for g in bgs[1:]:
                g["is_background"] = False
        self.groups = groups
        self._prev_sp_group = self.sp_group.copy()
        self._cache = {}

    def public_groups(self) -> list[dict]:
        return [{k: v for k, v in g.items() if not k.startswith("_")} for g in self.groups]

    @property
    def group_map(self) -> np.ndarray:
        return self.sp_group[self.sp_labels]

    @property
    def region_map(self) -> np.ndarray:
        return self.sp_region[self.sp_labels]

    # ----- edits

    def merge(self, ids: list[int]) -> None:
        ids = sorted(set(int(i) for i in ids))
        if len(ids) < 2:
            return
        keep = ids[0]
        self.sp_group[np.isin(self.sp_group, ids)] = keep
        self.rebuild()

    def split(self, gid: int, k: int = 2) -> None:
        from sklearn.cluster import KMeans
        members = np.nonzero(self.sp_group == gid)[0]
        if len(members) < 2:
            return
        k = max(2, min(int(k), len(members)))
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(self.sp_lab[members])
        new = km.labels_
        base = int(self.sp_group.max()) + 1
        for j in range(1, k):
            self.sp_group[members[new == j]] = base + j - 1
        self.rebuild()

    def move(self, region_ids: list[int], gid: int) -> None:
        mask = np.isin(self.sp_region, [int(r) for r in region_ids])
        self.sp_group[mask] = int(gid)
        self.rebuild()

    def regroup(self, max_groups: Optional[int], delta_e: float) -> None:
        self.sp_group = cluster_lab(self.sp_lab, self.sp_area, delta_e, max_groups)
        self.rebuild()

    def patch(self, gid: int, patch: dict) -> None:
        for g in self.groups:
            if g["id"] == gid:
                if "name" in patch and patch["name"]:
                    g["name"] = str(patch["name"])[:40]
                    g["_custom_name"] = True
                if "locked" in patch:
                    g["locked"] = bool(patch["locked"])
                if "is_background" in patch:
                    g["is_background"] = bool(patch["is_background"])

    # ----- rasters

    def layer(self, name: str) -> bytes:
        if name in self._cache:
            return self._cache[name]
        data = self._make_layer(name)
        self._cache[name] = data
        return data

    def _make_layer(self, name: str) -> bytes:
        work = self.work
        img = work.astype(np.float32) / 255.0
        if name == "work":
            return encode_jpeg(work, 90)
        if name == "preview":
            return encode_jpeg(resize_u8(work, fit_size(self.w, self.h, PREVIEW_LONG_SIDE)), 88)
        lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        if name in ("albedo", "shading"):
            sh = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(24))) / 255.0
            sh = np.maximum(sh, 0.04).astype(np.float32)
            if name == "shading":
                out = np.clip(sh / np.percentile(sh, 99.5), 0, 1)
                return encode_jpeg((np.repeat(out[..., None], 3, -1) * 255).astype(np.uint8), 88)
            alb = np.clip(img / sh[..., None] * float(np.mean(sh)) * 1.15, 0, 1)
            return encode_jpeg((alb * 255).astype(np.uint8), 88)
        if name == "residual":
            res = np.clip((img - 0.82) * 4.0, 0, 1)
            return encode_jpeg((res * 255).astype(np.uint8), 85)
        if name == "regions":
            rng = np.random.default_rng(3)
            colors = rng.integers(40, 235, (self.next_region_id + 1, 3)).astype(np.uint8)
            return encode_png(colors[self.region_map])
        if name == "groups":
            colors = np.array([_hex_rgb01(g["albedo_hex"]) * 255 for g in self.groups], np.uint8)
            return encode_png(colors[self.group_map])
        if name == "edges":
            rm = self.region_map
            edge = np.zeros(rm.shape, bool)
            edge[:, 1:] |= rm[:, 1:] != rm[:, :-1]
            edge[1:, :] |= rm[1:, :] != rm[:-1, :]
            out = (img * 0.55 * 255).astype(np.uint8)
            out[edge] = (255, 236, 120)
            return encode_jpeg(out, 88)
        if name == "ids/regions":
            rm = self.region_map.astype(np.int64)
            out = np.stack([rm & 255, (rm >> 8) & 255, (rm >> 16) & 255], -1).astype(np.uint8)
            return encode_png(out)
        if name == "ids/groups":
            gm = self.group_map.astype(np.uint8)
            return encode_png(np.stack([gm, np.zeros_like(gm), np.zeros_like(gm)], -1))
        raise KeyError(name)

    # ----- render

    def render(self, mapping: dict, options: dict, long_side: int, source: Optional[np.ndarray] = None) -> np.ndarray:
        """Tint each mapped group toward its target while keeping the luminance texture."""
        base = self.work if source is None else source
        size = fit_size(base.shape[1], base.shape[0], long_side)
        img = resize_u8(base, size).astype(np.float32) / 255.0 if size != base.shape[1::-1] else base.astype(np.float32) / 255.0
        gmap = resize_u8(self.group_map.astype(np.uint8), size, nearest=True).astype(np.int32)
        opts = {"mode": "shift", "texture": 1.0, "feather_px": 1.5, "keep_residual": True,
                "residual_tint": 0.0, "shading_strength": 1.0, "saturation": 1.0}
        opts.update({k: v for k, v in (options or {}).items() if v is not None})
        sat = float(opts["saturation"]); tex = float(opts["texture"]); shade = float(opts["shading_strength"])
        feather = float(opts["feather_px"]) * size[0] / self.w
        lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        out = img.copy()
        for g in self.groups:
            hexv = (mapping or {}).get(str(g["id"]))
            if not hexv or g["locked"]:
                continue
            mask = gmap == g["id"]
            if not mask.any():
                continue
            t = _hex_rgb01(hexv)
            t_l = float(t @ np.array([0.2126, 0.7152, 0.0722], np.float32))
            t = np.clip(t_l + (t - t_l) * sat, 0, 1)
            a = _hex_rgb01(g["albedo_hex"])
            a = np.maximum(a, 0.02)
            m_l = float(lum[mask].mean())
            ratio = np.clip(lum / max(m_l, 1e-3), 1e-3, 4.0)         # brightness relative to the group mean
            shade_gain = np.power(ratio, shade - 1.0)[..., None]      # 1 when shading_strength == 1
            flat = t[None, None, :] * ratio[..., None] * shade_gain   # target colour with the original shading
            shift = np.clip(img * (t / a)[None, None, :], 0, 1) * shade_gain   # keeps per-pixel albedo texture
            tinted = np.clip(flat * (1 - tex) + shift * tex, 0, 1)
            if feather > 0.3:
                wmask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(feather))) / 255.0
                wmask = wmask.astype(np.float32)[..., None]
                out = out * (1 - wmask) + tinted * wmask
            else:
                out[mask] = tinted[mask]
        if opts.get("keep_residual", True):
            res = np.clip((img - 0.9) * 2.0, 0, 1)
            out = np.clip(out + res * (1.0 - float(opts["residual_tint"]) * 0.5), 0, 1)
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# ----------------------------------------------------------------------------- jobs

class LayerNotReady(Exception):
    """A raster was requested before the stage that produces it has run (real server: 404 `layer_not_ready`)."""


class Job:
    def __init__(self, name: str, original: np.ndarray, options: dict) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.created = _now()
        self.status = "queued"
        self.error: Optional[str] = None
        self.options = {"detail": options.get("detail") or "balanced", "intrinsic": options.get("intrinsic") or "auto",
                        "max_groups": options.get("max_groups"), "delta_e": float(options.get("delta_e") or 10.0)}
        self.original = original
        h, w = original.shape[:2]
        ww, wh = fit_size(w, h, WORK_LONG_SIDE)
        pw, ph = fit_size(w, h, PREVIEW_LONG_SIDE)
        self.image = {"width": w, "height": h, "work_width": ww, "work_height": wh,
                      "preview_width": pw, "preview_height": ph}
        self.stages = {s: {"state": "idle", "progress": 0.0, "message": "", "seconds": 0.0} for s in STAGES}
        self.timings = {"total_s": 0.0}
        self.intrinsic_method = "careaga" if self.options["intrinsic"] != "heuristic" else "heuristic"
        self.analysis: Optional[Analysis] = None
        self.palette_id: Optional[str] = None
        self.mapping: dict = {}
        self.render_options: dict = {}
        self.exports: dict[str, bytes] = {}
        self._original_jpeg: Optional[bytes] = None
        self.lock = threading.RLock()
        self._subs: list[queue.Queue] = []
        self._subs_lock = threading.Lock()

    # ----- meta

    def to_dict(self) -> dict:
        with self.lock:
            groups = self.analysis.public_groups() if self.analysis else []
            return {
                "id": self.id, "name": self.name, "created": self.created, "status": self.status,
                "error": self.error, "image": dict(self.image), "options": dict(self.options),
                "stages": {k: dict(v) for k, v in self.stages.items()},
                "timings": dict(self.timings), "intrinsic_method": self.intrinsic_method,
                "groups": groups, "regions_count": len(self.analysis.regions) if self.analysis else 0,
                "palette_id": self.palette_id, "mapping": dict(self.mapping),
                "render_options": dict(self.render_options),
            }

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "created": self.created, "status": self.status,
                "thumb": f"/api/jobs/{self.id}/layers/preview", "width": self.image["width"],
                "height": self.image["height"],
                "n_groups": len(self.analysis.groups) if self.analysis else 0}

    # ----- events

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._subs_lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._subs_lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict) -> None:
        with self._subs_lock:
            for q in list(self._subs):
                q.put(event)

    def replay(self) -> list[dict]:
        with self.lock:
            ev = [{"type": "stage", "stage": s, **self.stages[s]} for s in STAGES]
            ev.append({"type": "status", "status": self.status})
            if self.status == "ready" and self.analysis:
                ev.append({"type": "groups", "groups": self.analysis.public_groups()})
                ev.append({"type": "done"})
            if self.status == "error":
                ev.append({"type": "error", "message": self.error or "Analysis failed"})
            return ev

    def set_stage(self, stage: str, state: str, progress: Optional[float] = None, message: Optional[str] = None) -> None:
        with self.lock:
            st = self.stages[stage]
            st["state"] = state
            if progress is not None:
                st["progress"] = float(progress)
            if message is not None:
                st["message"] = message
            ev = {"type": "stage", "stage": stage, "state": state, "progress": st["progress"], "message": st["message"]}
        self.publish(ev)

    def set_status(self, status: str) -> None:
        with self.lock:
            self.status = status
        self.publish({"type": "status", "status": status})

    # ----- analysis

    def analyze(self, tick_ms: int, fail: bool = False) -> None:
        t0 = _now()
        self.set_status("analyzing")
        try:
            for stage in STAGES:
                st0 = _now()
                msgs = STAGE_MESSAGES[stage]
                self.set_stage(stage, "running", 0.0, msgs[0])
                if stage == "segment" and fail:
                    raise RuntimeError("CUDA out of memory while running SAM 2 (mock failure)")
                if stage == "regions":
                    work = resize_u8(self.original, (self.image["work_width"], self.image["work_height"]))
                    self.analysis = Analysis(work, self.options["detail"], self.options["max_groups"], self.options["delta_e"])
                    if self.original.shape[0] * self.original.shape[1] > 30_000_000:
                        # Keep the mock's memory modest: "full" exports of huge originals use the work image.
                        self.original = work
                steps = len(msgs)
                for i in range(1, steps + 1):
                    if tick_ms:
                        time.sleep(tick_ms / 1000.0)
                    self.set_stage(stage, "running", i / steps, msgs[min(i, steps - 1)])
                with self.lock:
                    self.stages[stage]["seconds"] = round(_now() - st0, 2)
                self.set_stage(stage, "done", 1.0, "")
            with self.lock:
                self.timings["total_s"] = round(_now() - t0, 2)
            self.set_status("ready")
            self.publish({"type": "groups", "groups": self.analysis.public_groups()})
            self.publish({"type": "done"})
        except Exception as e:  # noqa: BLE001 - the mock reports every failure through the job
            with self.lock:
                self.error = str(e)
                for s in STAGES:
                    if self.stages[s]["state"] == "running":
                        self.stages[s]["state"] = "error"
                        self.stages[s]["message"] = str(e)
            self.publish({"type": "stage", "stage": stage, "state": "error", "progress": self.stages[stage]["progress"], "message": str(e)})
            self.set_status("error")
            self.publish({"type": "error", "message": str(e)})

    # ----- rasters

    def layer(self, name: str) -> tuple[bytes, str]:
        if name == "original":
            if self._original_jpeg is None:
                self._original_jpeg = encode_jpeg(self.original, 92)
            return self._original_jpeg, "image/jpeg"
        if not self.analysis:
            raise LayerNotReady(f"layer {name!r} is not available yet (job is {self.status})")
        data = self.analysis.layer(name)
        return data, ("image/png" if name in ("regions", "groups", "ids/regions", "ids/groups") else "image/jpeg")


class Registry:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.id] = job

    def get(self, jid: str) -> Optional[Job]:
        with self.lock:
            return self.jobs.get(jid)

    def list(self) -> list[dict]:
        with self.lock:
            return [j.summary() for j in sorted(self.jobs.values(), key=lambda j: -j.created)]

    def delete(self, jid: str) -> bool:
        with self.lock:
            return self.jobs.pop(jid, None) is not None


# ----------------------------------------------------------------------------- palettes

class Palettes:
    def __init__(self, samples: Samples) -> None:
        self.samples = samples
        self.store: dict[str, dict] = {}
        self.lock = threading.Lock()

    def create(self, prompt: str, n: int) -> dict:
        prompt = (prompt or "").strip()
        n = max(2, min(int(n or 6), 12))
        pid = hashlib.sha1(f"{prompt}|{n}".encode()).hexdigest()[:12]
        with self.lock:
            if pid in self.store:
                return self.store[pid]
        rng = random.Random(pid)
        hexes: list[str] = []
        method = "images"
        parsed = [h for _, h in parse_color_words(prompt)]
        theme = next((v for k, v in THEMES.items() if k in prompt.lower()), None)
        if len(parsed) >= 2:
            hexes = parsed[:n]
            method = "parsed"
        elif theme:
            hexes = list(theme)
            method = "theme"
        if len(hexes) < n:
            pool = [h for h in NAMED.values() if h not in hexes]
            rng.shuffle(pool)
            # Prefer chromatic colours so the strip looks like an image palette.
            pool.sort(key=lambda h: -sum(abs(float(v)) for v in hex_to_lab(h)[1:]) + rng.random() * 60)
            hexes += pool[: n - len(hexes)]
            if method != "images":
                method = "mixed"
        elif not prompt:
            method = "fallback"
        hexes = hexes[:n]
        weights = np.array([rng.uniform(0.5, 1.0) * (0.85 ** i) for i in range(n)])
        weights = weights / weights.sum()
        colors = []
        for h, w in zip(hexes, weights):
            lab = hex_to_lab(h)
            colors.append({"hex": h, "lab": [round(v, 2) for v in lab], "weight": round(float(w), 4),
                           "name": nearest_name(lab)})
        names = self.samples.names()
        rng.shuffle(names)
        sources = []
        for i, name in enumerate(names[: (0 if method == "parsed" else rng.randint(3, 5))]):
            m = self.samples.manifest[name]
            sources.append({"url": m.get("source", ""), "title": re.sub(r"^File:", "", m.get("title", name)),
                            "license": m.get("license", ""), "thumb": f"/api/palettes/{pid}/sources/{i}.jpg",
                            "_sample": name})
        pal = {"id": pid, "prompt": prompt, "method": method, "created": _now(), "colors": colors, "sources": sources}
        with self.lock:
            self.store[pid] = pal
        return pal

    @staticmethod
    def public(pal: dict) -> dict:
        return {**pal, "sources": [{k: v for k, v in s.items() if not k.startswith("_")} for s in pal["sources"]]}


# ----------------------------------------------------------------------------- mapping

def suggest_mapping(groups: list[dict], colors: list[str], strategy: str) -> dict:
    """Small, deterministic stand-in for recolor.mapping.suggest_mapping."""
    if not colors:
        return {str(g["id"]): None for g in groups}
    cand = [g for g in groups if not g["locked"] and not g["is_background"]]
    labs = [hex_to_lab(c) for c in colors]
    result: dict[str, Optional[str]] = {str(g["id"]): None for g in groups}

    def by_l(items: list, key: Callable) -> list:
        return sorted(items, key=key)

    if strategy == "luminance" or strategy == "contrast":
        gs = by_l(cand, lambda g: g["albedo_lab"][0])
        cs = by_l(range(len(colors)), lambda i: labs[i][0])
        if len(gs) <= len(cs):
            # spread the palette across the group lightness order
            for k, g in enumerate(gs):
                result[str(g["id"])] = colors[cs[round(k * (len(cs) - 1) / max(1, len(gs) - 1))]]
        else:
            for k, g in enumerate(gs):
                result[str(g["id"])] = colors[cs[min(len(cs) - 1, k * len(cs) // len(gs))]]
        return result
    if strategy == "hue":
        used: set[int] = set()
        for g in cand:
            L, a, b = g["albedo_lab"]
            gh = math.atan2(b, a)
            best, bd = 0, 1e9
            for i, (Lc, ac, bc) in enumerate(labs):
                d = abs((math.atan2(bc, ac) - gh + math.pi) % (2 * math.pi) - math.pi)
                if i in used:
                    d += 1.0
                if d < bd:
                    best, bd = i, d
            used.add(best)
            result[str(g["id"])] = colors[best]
        return result
    # area / balanced: rank by area vs palette order (weights are already sorted)
    for k, g in enumerate(cand):
        if strategy == "balanced" and g["hue_family"] == "neutral":
            neutrals = [i for i, (L, a, b) in enumerate(labs) if math.hypot(a, b) < 12]
            if neutrals:
                result[str(g["id"])] = colors[neutrals[k % len(neutrals)]]
                continue
        result[str(g["id"])] = colors[k % len(colors)]
    return result


# ----------------------------------------------------------------------------- HTTP

class MockState:
    def __init__(self, tick_ms: int) -> None:
        self.tick_ms = tick_ms
        self.samples = Samples()
        self.registry = Registry()
        self.palettes = Palettes(self.samples)
        self.started = _now()
        self.models_ready_at = self.started + 6.0


STATE: MockState


def parse_multipart(body: bytes, content_type: str) -> dict[str, Any]:
    """Minimal multipart/form-data parser: returns {field: str | (filename, bytes)}."""
    m = re.search(r"boundary=\"?([^\";]+)\"?", content_type)
    if not m:
        return {}
    boundary = m.group(1).encode()
    out: dict[str, Any] = {}
    for part in body.split(b"--" + boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        headers = head.decode("utf-8", "replace")
        name = re.search(r'name="([^"]*)"', headers)
        fname = re.search(r'filename="([^"]*)"', headers)
        if not name:
            continue
        if fname:
            out[name.group(1)] = (fname.group(1), data)
        else:
            out[name.group(1)] = data.decode("utf-8", "replace")
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ChromaMock/0.1"

    # ----- plumbing

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter log
        first = str(args[0]) if args else ""
        if "/api/" in first or os.environ.get("MOCK_VERBOSE"):
            sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _send(self, status: int, body: bytes, ctype: str, extra: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200, extra: Optional[dict] = None) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json", extra)

    def _error(self, status: int, error: str, detail: str = "") -> None:
        self._json({"error": error, "detail": detail}, status)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _json_body(self) -> dict:
        raw = self._body()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON body: {e}") from e

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_PATCH(self) -> None:
        self._dispatch("PATCH")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        url = urlsplit(self.path)
        path = unquote(url.path)
        qs = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if path.startswith("/api/"):
                self._api(method, path, qs)
            elif method in ("GET", "HEAD"):
                self._static(path)
            else:
                self._error(405, "Method not allowed")
        except ValueError as e:
            self._error(400, "Bad request", str(e))
        except KeyError as e:
            self._error(404, "Not found", str(e).strip("'"))
        except LayerNotReady as e:
            self._error(404, "layer_not_ready", str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._error(500, "Mock server error", f"{type(e).__name__}: {e}")

    # ----- static

    def _static(self, path: str) -> None:
        if path == "/" or path.startswith("/#"):
            path = "/index.html"
        full = os.path.normpath(os.path.join(WEB_DIR, path.lstrip("/")))
        if not full.startswith(WEB_DIR) or not os.path.isfile(full):
            # SPA: unknown non-file paths get index.html
            if "." not in os.path.basename(path):
                full = os.path.join(WEB_DIR, "index.html")
            else:
                self._error(404, "Not found", path)
                return
        with open(full, "rb") as f:
            data = f.read()
        self._send(200, data, MIME.get(os.path.splitext(full)[1].lower(), "application/octet-stream"))

    # ----- api

    def _api(self, method: str, path: str, qs: dict) -> None:
        S = STATE
        parts = path.split("/")[2:]   # after /api/
        route = (method, *parts)

        # health / samples
        if route == ("GET", "health"):
            t = _now() - S.started
            model = "cold" if t < 1.5 else ("loading" if _now() < S.models_ready_at else "ready")
            used = 1800 + 2200 * (1 if model == "ready" else 0) + int(600 * math.sin(t / 3))
            self._json({"ok": True, "device": "cuda", "gpu": "NVIDIA GeForce RTX 5090 (mock)",
                        "vram_total_mb": 32607, "vram_used_mb": used,
                        "models": {"sam2": model, "intrinsic": model},
                        "jobs": len(S.registry.jobs), "version": "mock-0.1"})
            return
        if route == ("GET", "samples"):
            self._json(S.samples.listing())
            return
        if method == "GET" and len(parts) == 2 and parts[0] == "samples":
            name = parts[1]
            if name not in S.samples.manifest:
                raise KeyError(name)
            w = int(qs["w"]) if qs.get("w") else None
            self._send(200, S.samples.encoded(name, w), "image/jpeg", {"Cache-Control": "public, max-age=3600"})
            return

        # jobs
        if route == ("GET", "jobs"):
            self._json(S.registry.list())
            return
        if route == ("POST", "jobs"):
            self._create_job()
            return
        if parts[0] == "jobs" and len(parts) >= 2:
            job = S.registry.get(parts[1])
            if not job:
                raise KeyError(f"job {parts[1]}")
            sub = parts[2:]
            if not sub:
                if method == "GET":
                    self._json(job.to_dict())
                elif method == "DELETE":
                    S.registry.delete(job.id)
                    job.set_status("deleted")
                    self._json({"ok": True})
                else:
                    self._error(405, "Method not allowed")
                return
            if sub == ["events"] and method == "GET":
                self._events(job, qs)
                return
            if sub[0] == "layers" and len(sub) == 2 and method == "GET":
                if sub[1] not in ("original", "work", "preview", "albedo", "shading", "residual", "regions", "groups", "edges"):
                    raise KeyError(sub[1])
                data, ctype = job.layer(sub[1])
                self._send(200, data, ctype)
                return
            if sub[0] == "ids" and len(sub) == 2 and method == "GET":
                if sub[1] not in ("regions", "groups"):
                    raise KeyError(sub[1])
                data, ctype = job.layer("ids/" + sub[1])
                self._send(200, data, ctype)
                return
            if sub[0] == "groups":
                self._group_edit(job, method, sub[1:])
                return
            if sub == ["regroup"] and method == "POST":
                b = self._json_body()
                self._require_ready(job)
                with job.lock:
                    job.options["max_groups"] = b.get("max_groups")
                    if b.get("delta_e"):
                        job.options["delta_e"] = float(b["delta_e"])
                    job.analysis.regroup(b.get("max_groups"), job.options["delta_e"])
                    job.mapping = {}
                self._json(job.to_dict())
                return
            if sub == ["mapping", "suggest"] and method == "POST":
                b = self._json_body()
                self._require_ready(job)
                strategy = b.get("strategy") or "balanced"
                if strategy not in ("balanced", "area", "luminance", "hue", "contrast"):
                    raise ValueError(f"Unknown strategy {strategy!r}")
                with job.lock:
                    m = suggest_mapping(job.analysis.groups, list(b.get("colors") or []), strategy)
                self._json({"mapping": m})
                return
            if sub == ["render"] and method == "POST":
                b = self._json_body()
                self._require_ready(job)
                t0 = time.perf_counter()
                with job.lock:
                    out = job.analysis.render(b.get("mapping") or {}, b.get("options") or {}, PREVIEW_LONG_SIDE)
                data = encode_jpeg(out, 88)
                ms = (time.perf_counter() - t0) * 1000
                self._send(200, data, "image/jpeg", {"X-Render-Ms": f"{ms:.1f}", "Access-Control-Expose-Headers": "X-Render-Ms"})
                return
            if sub == ["export"] and method == "POST":
                b = self._json_body()
                self._require_ready(job)
                quality = b.get("quality") or "work"
                fmt = b.get("format") or "png"
                if fmt not in ("png", "jpg") or quality not in ("work", "full"):
                    raise ValueError("format must be png|jpg and quality work|full")
                t0 = time.perf_counter()
                with job.lock:
                    if quality == "full":
                        out = job.analysis.render(b.get("mapping") or {}, b.get("options") or {},
                                                  max(job.original.shape[:2]), source=job.original)
                    else:
                        out = job.analysis.render(b.get("mapping") or {}, b.get("options") or {}, WORK_LONG_SIDE)
                data = encode_png(out) if fmt == "png" else encode_jpeg(out, 94)
                fname = f"{os.path.splitext(job.name)[0]}-recolor-{quality}-{int(_now())}.{fmt}"
                job.exports[fname] = data
                ms = (time.perf_counter() - t0) * 1000
                self._json({"url": f"/api/jobs/{job.id}/exports/{fname}", "width": int(out.shape[1]),
                            "height": int(out.shape[0]), "ms": round(ms, 1)})
                return
            if sub[0] == "exports" and len(sub) == 2 and method == "GET":
                data = job.exports.get(sub[1])
                if data is None:
                    raise KeyError(sub[1])
                self._send(200, data, "image/png" if sub[1].endswith(".png") else "image/jpeg",
                           {"Content-Disposition": f'attachment; filename="{sub[1]}"'})
                return
            if sub == ["state"] and method == "PUT":
                b = self._json_body()
                with job.lock:
                    if "mapping" in b and isinstance(b["mapping"], dict):
                        job.mapping = {str(k): (v or None) for k, v in b["mapping"].items()}
                    if "render_options" in b and isinstance(b["render_options"], dict):
                        job.render_options = dict(b["render_options"])
                    if "palette_id" in b:
                        job.palette_id = b["palette_id"] or None
                self._json(job.to_dict())
                return
            raise KeyError(path)

        # palettes
        if route == ("POST", "palettes"):
            b = self._json_body()
            time.sleep(0.9)   # pretend to search the web
            pal = S.palettes.create(b.get("prompt", ""), b.get("n_colors", 6))
            self._json(Palettes.public(pal))
            return
        if parts[0] == "palettes" and len(parts) == 2 and method == "GET":
            pal = S.palettes.store.get(parts[1])
            if not pal:
                raise KeyError(parts[1])
            self._json(Palettes.public(pal))
            return
        if parts[0] == "palettes" and len(parts) == 4 and parts[2] == "sources" and method == "GET":
            pal = S.palettes.store.get(parts[1])
            idx = int(parts[3].split(".")[0])
            if not pal or idx >= len(pal["sources"]):
                raise KeyError(path)
            self._send(200, S.samples.encoded(pal["sources"][idx]["_sample"], 640), "image/jpeg",
                       {"Cache-Control": "public, max-age=3600"})
            return
        raise KeyError(path)

    def _require_ready(self, job: Job) -> None:
        if job.status != "ready" or not job.analysis:
            raise ValueError("Job is not ready yet")

    def _create_job(self) -> None:
        S = STATE
        ctype = self.headers.get("Content-Type", "")
        fields: dict[str, Any] = {}
        image: Optional[np.ndarray] = None
        name = "upload.jpg"
        if ctype.startswith("multipart/form-data"):
            form = parse_multipart(self._body(), ctype)
            f = form.get("file")
            if not f or not isinstance(f, tuple):
                raise ValueError("multipart field 'file' is required")
            name, data = f
            try:
                im = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"Could not decode image: {e}") from e
            if max(im.size) > 6000:
                im = im.resize(fit_size(*im.size, 6000), Image.LANCZOS)
            image = np.asarray(im)
            fields = {k: v for k, v in form.items() if k != "file"}
        else:
            b = self._json_body()
            sample = b.get("sample")
            if not sample or sample not in S.samples.manifest:
                raise ValueError("JSON body needs {sample: <name>} or a multipart 'file'")
            name = sample
            image = S.samples.load(sample)
            fields = b
        opts = {
            "detail": fields.get("detail") or "balanced",
            "intrinsic": fields.get("intrinsic") or "auto",
            "max_groups": int(fields["max_groups"]) if fields.get("max_groups") not in (None, "", "null") else None,
            "delta_e": float(fields.get("delta_e") or 10.0),
        }
        if opts["detail"] not in ("fast", "balanced", "max"):
            raise ValueError("detail must be fast|balanced|max")
        job = Job(os.path.basename(name), image, opts)
        S.registry.add(job)
        fail = "fail" in job.name.lower()
        created = job.to_dict()  # snapshot while still 'queued', like the real server
        threading.Thread(target=job.analyze, args=(S.tick_ms, fail), daemon=True).start()
        self._json(created, 201)

    def _group_edit(self, job: Job, method: str, sub: list[str]) -> None:
        self._require_ready(job)
        b = self._json_body()
        with job.lock:
            an = job.analysis
            if method == "POST" and sub == ["merge"]:
                ids = [int(i) for i in b.get("group_ids") or []]
                if len(ids) < 2:
                    raise ValueError("group_ids needs at least two ids")
                an.merge(ids)
            elif method == "POST" and sub == ["split"]:
                an.split(int(b["group_id"]), int(b.get("k") or 2))
            elif method == "POST" and sub == ["move"]:
                an.move([int(r) for r in b.get("region_ids") or []], int(b["group_id"]))
            elif method == "PATCH" and len(sub) == 1:
                an.patch(int(sub[0]), b)
                self._json(job.to_dict())
                return
            else:
                raise KeyError("/".join(sub))
            # Group ids may have been renumbered; keep only mappings that still point somewhere.
            valid = {str(g["id"]) for g in an.groups}
            job.mapping = {k: v for k, v in job.mapping.items() if k in valid}
        job.publish({"type": "groups", "groups": an.public_groups()})
        self._json(job.to_dict())

    def _events(self, job: Job, qs: Optional[dict] = None) -> None:
        """SSE like the real server: replay, then live events with a keepalive every 15 s.
        `?once=1` closes after the replay; `?timeout=S` closes after S seconds without an event."""
        qs = qs or {}
        once = qs.get("once") not in (None, "", "0", "false")
        try:
            idle_limit = float(qs.get("timeout") or 0) or None
        except ValueError:
            idle_limit = None
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        q = job.subscribe()
        try:
            for ev in job.replay():
                self._sse(ev)
            if once:
                return
            last_beat = last_event = _now()
            while True:
                try:
                    ev = q.get(timeout=1.0)
                    self._sse(ev)
                    last_beat = last_event = _now()
                    if ev.get("type") == "status" and ev.get("status") == "deleted":
                        break
                except queue.Empty:
                    if idle_limit is not None and _now() - last_event >= idle_limit:
                        break
                    if _now() - last_beat > 15:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        last_beat = _now()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            job.unsubscribe(q)
            try:
                self.close_connection = True
            except Exception:  # noqa: BLE001
                pass

    def _sse(self, ev: dict) -> None:
        self.wfile.write(f"event: {ev.get('type', 'message')}\ndata: {json.dumps(ev)}\n\n".encode())
        self.wfile.flush()


# ----------------------------------------------------------------------------- main

def seed_jobs(state: MockState, names: list[str]) -> None:
    """Pre-analyze a few samples (no stepper delay) so Home and Gallery are populated."""
    for i, name in enumerate(names):
        job = Job(name, state.samples.load(name), {"detail": "balanced", "delta_e": 10.0})
        job.created -= 3600 * (i + 1)
        state.registry.add(job)
        job.analyze(0)


def lan_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main() -> None:
    global STATE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("MOCK_PORT", "8791")))
    ap.add_argument("--tick", type=int, default=TICK_MS, help="stepper tick in ms (0 = instant)")
    ap.add_argument("--no-seed", action="store_true", help="do not pre-analyze sample jobs")
    args = ap.parse_args()
    STATE = MockState(args.tick)
    if not args.no_seed:
        threading.Thread(target=seed_jobs, args=(STATE, ["motorcycle_1.jpg", "sneakers_1.jpg"]), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.daemon_threads = True
    print(f"Chroma Studio mock  ·  http://{lan_ip()}:{args.port}  ·  http://localhost:{args.port}  (tick {args.tick} ms)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
