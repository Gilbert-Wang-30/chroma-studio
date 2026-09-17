"""Distill a small, designed palette from reference photographs.

Pixels from every image are pooled in CIE Lab, weighted by chroma
(``1 + (C/40)**1.5``) so a gray sky or asphalt does not dominate a "hawaii sunset"
palette, clustered with a weighted k-means (``k = n + 18``: many tight clusters keep a
vivid koi orange from being averaged into bronze before the merge step), and the
centroids are merged (CIEDE2000 < 9), pruned (< 2 % of the weight) and ranked.
The k-means runs in torch on the GPU when one is available; the memory footprint is a
few megabytes and is released before returning.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .. import colornames, imageio
from ..types import PaletteColor

MAX_PIXELS_PER_IMAGE = 150_000
MERGE_DELTA_E = 9.0
MIN_CLUSTER_WEIGHT = 0.02
MIN_LIGHTNESS_SPAN = 25.0
EXTRA_CLUSTERS = 18
CHROMA_SCALE = 40.0
CHROMA_POWER = 1.5
KMEANS_ITERS = 30
SEED = 1234


def pool_pixels(images: list[np.ndarray], max_per_image: int = MAX_PIXELS_PER_IMAGE,
                seed: int = SEED) -> np.ndarray:
    """Sample at most ``max_per_image`` pixels from each uint8 RGB image and return them
    stacked as float32 Lab ``(P, 3)``.  Sampling is deterministic for a given seed.

    Tolerates inputs outside the uint8 RGB contract: a floating array is taken as
    sRGB in [0, 1] (clipped, not divided by 255), a 2-D array is treated as grayscale
    (replicated to three channels), an RGBA image drops its alpha; anything else
    (empty, 1-D, two-channel) is skipped."""
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for img in images:
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] < 3:
            continue
        flat = arr[..., :3].reshape(-1, 3)
        if flat.shape[0] == 0:
            continue
        if flat.shape[0] > max_per_image:
            idx = rng.choice(flat.shape[0], size=max_per_image, replace=False)
            flat = flat[idx]
        if np.issubdtype(flat.dtype, np.floating):
            rgb01 = np.clip(flat.astype(np.float32), 0.0, 1.0)
        else:
            rgb01 = np.clip(flat.astype(np.float32), 0.0, 255.0) / 255.0
        chunks.append(imageio.rgb_to_lab(rgb01))
    if not chunks:
        return np.zeros((0, 3), np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def chroma_weights(lab: np.ndarray) -> np.ndarray:
    """``1 + (C/CHROMA_SCALE)**CHROMA_POWER`` per pixel, C being the Lab chroma: a neutral
    pixel weighs 1, a vivid one (C = 80) about 3.8, so colorful subjects win over large
    gray backgrounds without a lone saturated speck taking over."""
    c = np.hypot(lab[:, 1], lab[:, 2])
    return (1.0 + (c / CHROMA_SCALE) ** CHROMA_POWER).astype(np.float32)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighted_kmeans(points: np.ndarray, weights: np.ndarray, k: int, iters: int = KMEANS_ITERS,
                    seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Weighted k-means (k-means++ seeding, Lloyd iterations) in torch.

    Guarantees: returns ``(centroids (k', 3), weights (k',))`` with ``k' <= k`` (empty
    clusters are dropped) and cluster weights summing to the total input weight; the
    result is deterministic for a fixed seed and the CUDA cache is emptied afterwards.
    """
    n = points.shape[0]
    if n == 0 or k <= 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
    k = min(k, n)
    dev = _device()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)).to(dev)
    w = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32)).to(dev)
    try:
        # k-means++ seeding on the weighted distribution.
        probs = (w / w.sum()).cpu()
        first = int(torch.multinomial(probs, 1, generator=gen))
        centers = [x[first]]
        d2 = ((x - centers[0]) ** 2).sum(1)
        for _ in range(1, k):
            p = (d2 * w).cpu()
            if float(p.sum()) <= 0:
                break
            idx = int(torch.multinomial(p / p.sum(), 1, generator=gen))
            centers.append(x[idx])
            d2 = torch.minimum(d2, ((x - x[idx]) ** 2).sum(1))
        c = torch.stack(centers)

        def _accumulate(cent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # One-hot matmul instead of index_add_: no atomics, so the reduction order
            # (and therefore the result) is identical from run to run on the GPU.
            assign = torch.cdist(x, cent).argmin(1)
            a = torch.nn.functional.one_hot(assign, cent.shape[0]).to(x.dtype) * w[:, None]  # (n, k)
            return a.sum(0), a.T @ x                       # counts (k,), sums (k, 3)

        for _ in range(iters):
            counts, sums = _accumulate(c)
            keep = counts > 0
            new_c = c.clone()
            new_c[keep] = sums[keep] / counts[keep][:, None]
            shift = float((new_c - c).abs().max())
            c = new_c
            if shift < 1e-3:
                break
        counts, _ = _accumulate(c)
        keep = counts > 0
        cents = c[keep].cpu().numpy().astype(np.float32)
        cw = counts[keep].cpu().numpy().astype(np.float32)
    finally:
        del x, w
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return cents, cw


def merge_centroids(cents: np.ndarray, weights: np.ndarray, delta_e: float = MERGE_DELTA_E
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Greedily merge centroids closer than ``delta_e`` (CIEDE2000), heaviest first,
    weight-averaging the merged Lab.  Returns ``(centroids, weights)`` sorted by weight
    descending."""
    if cents.shape[0] == 0:
        return cents, weights
    order = np.argsort(-weights, kind="stable")
    cents, weights = cents[order], weights[order]
    merged_c: list[np.ndarray] = []
    merged_w: list[float] = []
    for c, w in zip(cents, weights):
        if merged_c:
            d = imageio.delta_e(np.repeat(c[None], len(merged_c), 0), np.stack(merged_c))
            j = int(np.argmin(d))
            if float(d[j]) < delta_e:
                tot = merged_w[j] + float(w)
                merged_c[j] = (merged_c[j] * merged_w[j] + c * float(w)) / tot
                merged_w[j] = tot
                continue
        merged_c.append(c.astype(np.float32))
        merged_w.append(float(w))
    out_c = np.stack(merged_c).astype(np.float32)
    out_w = np.asarray(merged_w, np.float32)
    order = np.argsort(-out_w, kind="stable")
    return out_c[order], out_w[order]


def _to_palette_colors(cents: np.ndarray, weights: np.ndarray) -> list[PaletteColor]:
    total = float(weights.sum()) or 1.0
    out: list[PaletteColor] = []
    for c, w in zip(cents, weights):
        lab = (float(c[0]), float(c[1]), float(c[2]))
        out.append(PaletteColor(hex=imageio.lab_to_hex(lab), lab=lab, weight=float(w) / total,
                                name=colornames.nearest_name(lab)))
    return out


def extract_palette(images: list[np.ndarray], n_colors: int) -> list[PaletteColor]:
    """Extract up to ``n_colors`` representative colors from uint8 RGB images.

    Guarantees: returns at most ``n_colors`` ``PaletteColor`` records sorted by weight
    descending with weights summing to 1 (or an empty list when there are no usable
    pixels); no two colors are within CIEDE2000 ``MERGE_DELTA_E`` of each other; every
    color carries at least ``MIN_CLUSTER_WEIGHT`` of the (chroma-weighted) pixel mass
    unless it was re-added to guarantee a lightness span of at least
    ``MIN_LIGHTNESS_SPAN`` (the darkest and lightest merged centroids replace the two
    lightest-weighted picks when all picks are within 25 L of each other).  The result
    is deterministic for identical inputs.
    """
    n_colors = int(n_colors)
    if n_colors <= 0:
        return []
    lab = pool_pixels(images)
    if lab.shape[0] == 0:
        return []
    weights = chroma_weights(lab)
    cents, cw = weighted_kmeans(lab, weights, n_colors + EXTRA_CLUSTERS)
    cents, cw = merge_centroids(cents, cw)
    if cents.shape[0] == 0:
        return []
    total = float(cw.sum())
    frac = cw / total
    keep = frac >= MIN_CLUSTER_WEIGHT
    if not keep.any():
        keep[0] = True
    strong_c, strong_w = cents[keep], cw[keep]
    pick_c, pick_w = strong_c[:n_colors], strong_w[:n_colors]

    # Guarantee a value structure: a palette of six mid-tones looks flat on a product.
    span = float(pick_c[:, 0].max() - pick_c[:, 0].min())
    if span < MIN_LIGHTNESS_SPAN and cents.shape[0] > 1:
        darkest = int(np.argmin(cents[:, 0]))
        lightest = int(np.argmax(cents[:, 0]))
        extras = [i for i in (darkest, lightest)
                  if not any(np.array_equal(cents[i], p) for p in pick_c)]
        if extras:
            room = max(0, n_colors - len(pick_c))
            drop = max(0, len(extras) - room)
            if drop:
                # Drop the lightest-weighted picks to make room (picks are weight-sorted).
                pick_c, pick_w = pick_c[:len(pick_c) - drop], pick_w[:len(pick_w) - drop]
            pick_c = np.concatenate([pick_c, cents[extras]], 0)
            pick_w = np.concatenate([pick_w, cw[extras]], 0)
            order = np.argsort(-pick_w, kind="stable")
            pick_c, pick_w = pick_c[order], pick_w[order]
    return _to_palette_colors(pick_c, pick_w)


def lightness_span(colors: list[PaletteColor]) -> float:
    """Difference between the lightest and darkest color's L (0 for an empty list)."""
    if not colors:
        return 0.0
    ls = [c.lab[0] for c in colors]
    return float(max(ls) - min(ls))


def hue_angle(lab) -> float:
    """Lab hue angle in degrees, [0, 360)."""
    return (math.degrees(math.atan2(float(lab[2]), float(lab[1]))) + 360.0) % 360.0
