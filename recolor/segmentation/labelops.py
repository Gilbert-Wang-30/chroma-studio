"""Vectorised helpers on int32 label maps.

Shared by the region hierarchy and the grouping code: per-region medians of a Lab
image, region adjacency, union-find merging and compact relabelling. Everything here
is pure numpy/torch and safe to run on the CPU (tests) or the GPU (pipeline).
"""
from __future__ import annotations

import numpy as np
import torch

_QUANT = 100.0          # Lab is quantised to 0.01 units for the sort-based median
_OFFSET = 200.0         # shift so every Lab channel is positive before quantising
_SPAN = 1 << 16         # per-channel key span; (128 + 200) * 100 < 65536


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def region_medians(labels: np.ndarray, lab: np.ndarray, n: int) -> np.ndarray:
    """Per-region lower median of each Lab channel.

    `labels` is int32 HxW (values < 0 are ignored), `lab` float32 HxWx3, `n` the number
    of region ids to report. Returns float32 [n, 3]; rows of regions with no pixels are
    NaN. Exact to 0.01 Lab units; runs as one sort per channel on the GPU when available.
    """
    out = np.full((n, 3), np.nan, dtype=np.float32)
    if n <= 0:
        return out
    dev = _device()
    flat = torch.from_numpy(np.ascontiguousarray(labels).astype(np.int64).ravel()).to(dev)
    valid = flat >= 0
    flat = flat[valid]
    if flat.numel() == 0:
        return out
    labt = torch.from_numpy(np.ascontiguousarray(lab, dtype=np.float32).reshape(-1, 3)).to(dev)[valid]
    counts = torch.bincount(flat, minlength=n)
    starts = torch.cumsum(counts, 0) - counts
    mid = (starts + (counts - 1).clamp_min(0) // 2).clamp_max(flat.numel() - 1)
    res = torch.empty((n, 3), device=dev, dtype=torch.float32)
    for c in range(3):
        q = ((labt[:, c] + _OFFSET) * _QUANT).round().clamp(0, _SPAN - 1).to(torch.int64)
        key, _ = torch.sort(flat * _SPAN + q)
        res[:, c] = (key[mid] % _SPAN).float() / _QUANT - _OFFSET
    res[counts == 0] = float("nan")
    return res.cpu().numpy().astype(np.float32)


def region_areas(labels: np.ndarray, n: int) -> np.ndarray:
    """Pixel count per region id (int64 [n]); ids < 0 are ignored."""
    v = labels[labels >= 0]
    return np.bincount(v.ravel(), minlength=n).astype(np.int64)


def adjacency(labels: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """4-connected touching pairs of distinct region ids.

    Returns (pairs int64 [M, 2] with pairs[:, 0] < pairs[:, 1], shared boundary length
    int64 [M]). Pixels labelled < 0 never form pairs.
    """
    p = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()]).astype(np.int64)
    q = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()]).astype(np.int64)
    m = (p != q) & (p >= 0) & (q >= 0)
    p, q = p[m], q[m]
    if p.size == 0:
        return np.zeros((0, 2), np.int64), np.zeros((0,), np.int64)
    lo = np.minimum(p, q)
    hi = np.maximum(p, q)
    key, cnt = np.unique(lo * n + hi, return_counts=True)
    return np.stack([key // n, key % n], axis=1), cnt.astype(np.int64)


def find_root(parent: np.ndarray, i: int) -> int:
    """Union-find root with path halving; `parent` is modified in place."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return int(i)


def roots_of(parent: np.ndarray) -> np.ndarray:
    """Root for every index of a union-find parent array (int64 [n])."""
    out = np.empty(len(parent), np.int64)
    for i in range(len(parent)):
        out[i] = find_root(parent, i)
    return out


def compact(labels: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Renumber the non-empty ids of `labels` to 0..K-1 in ascending order of old id.

    Returns (labels int32, mapping int64 [n] old -> new, -1 for empty ids). Pixels with
    a negative label stay negative.
    """
    areas = region_areas(labels, n)
    keep = np.flatnonzero(areas > 0)
    mapping = np.full(n, -1, np.int64)
    mapping[keep] = np.arange(len(keep))
    out = np.where(labels >= 0, mapping[np.clip(labels, 0, max(n - 1, 0))], -1).astype(np.int32)
    return out, mapping


def bboxes(labels: np.ndarray, n: int) -> np.ndarray:
    """Bounding boxes (x0, y0, x1, y1 exclusive) as int64 [n, 4]; zeros for empty ids."""
    from scipy import ndimage
    out = np.zeros((n, 4), np.int64)
    if n == 0:
        return out
    objs = ndimage.find_objects(labels + 1, max_label=n)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        out[i] = (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
    return out


def border_counts(labels: np.ndarray, n: int) -> np.ndarray:
    """How many image-border pixels each region id owns (int64 [n])."""
    b = np.concatenate([labels[0, :], labels[-1, :], labels[1:-1, 0], labels[1:-1, -1]])
    b = b[b >= 0]
    return np.bincount(b, minlength=n).astype(np.int64)
