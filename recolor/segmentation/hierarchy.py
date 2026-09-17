"""From overlapping SAM proposals to a clean partition of the image into regions.

`build_regions` implements the four steps of docs/ARCHITECTURE.md §3.2:

1. paint proposals largest-first (parts override wholes), dropping tiny masks and
   duplicate proposals (same albedo as the region they sit in);
2. split regions whose albedo is clearly bimodal (two-tone parts SAM returns as one);
3. label what SAM missed with SLIC superpixels, merging each into the touching region
   with the nearest median albedo when the colors agree;
4. absorb specks into their most similar neighbour, then relabel to 0..N-1.

All color decisions are made on the *albedo* (shading removed) in CIE Lab, so a shadow
across a panel never splits it and two same-colored parts are never a "duplicate" just
because one is lit.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import cv2
import numpy as np

from .. import filters, imageio
from .labelops import adjacency, compact, find_root, region_areas, region_medians, roots_of
from .superpixels import slic_labels

Progress = Optional[Callable[[float, str], None]]

# Per-preset region parameters. `min_area_frac` is the fraction of the image a region must
# cover to be created; `superpixel_px` the target pixels per SLIC superpixel for the fill.
REGION_PRESETS: dict[str, dict[str, float]] = {
    "fast": {"min_area_frac": 0.0015, "superpixel_px": 600},
    "balanced": {"min_area_frac": 0.0006, "superpixel_px": 400},
    "max": {"min_area_frac": 0.00025, "superpixel_px": 300},
}

DUPLICATE_OVERLAP = 0.85   # a proposal sitting this much inside one region is a candidate duplicate
DUPLICATE_DELTA_E = 4.0    # ... and is dropped when its median albedo is this close to that region's
SPLIT_DELTA_E = 14.0       # 2-means centroids must differ by this much for a bimodal split
SPLIT_VALLEY = 0.5         # ... and the density between the modes must drop below this
                           #     fraction of the lower mode (rejects gradients and blobs)
SPLIT_MIN_FRACTION = 0.08  # the smaller mode must hold this fraction of the region's pixels
FILL_DELTA_E = 8.0         # superpixels merge into a touching region closer than this
MAX_MERGE_PASSES = 12
REFINE_RADIUS = 2          # guided-filter snap of region edges to the image (0 disables)
_MEDIAN_SAMPLE = 20_000    # pixels used for a per-mask median
_KMEANS_SAMPLE = 4_000     # pixels used to fit the 2-means of a region


# ---------------------------------------------------------------------- small helpers

def _sub(pix: np.ndarray, n: int) -> np.ndarray:
    if len(pix) <= n:
        return pix
    return pix[:: int(np.ceil(len(pix) / n))]


def _median(pix: np.ndarray) -> np.ndarray:
    return np.median(_sub(pix, _MEDIAN_SAMPLE), axis=0).astype(np.float32)


def _de(a: np.ndarray, b: np.ndarray) -> float:
    return float(imageio.delta_e(np.asarray(a, np.float32)[None], np.asarray(b, np.float32)[None])[0])


def _mask_crop(bbox: list, shape: tuple[int, int]) -> tuple[slice, slice]:
    h, w = shape
    x, y, bw, bh = (int(round(v)) for v in bbox)
    return slice(max(0, y), min(h, y + bh + 1)), slice(max(0, x), min(w, x + bw + 1))


def two_means(sample: np.ndarray, iters: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-means on rows of `sample` (float32 [n, 3]) initialised along the principal axis.
    Returns (c1, c2, assignment bool [n], True = c2). Deterministic."""
    x = sample.astype(np.float32)
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / max(1, len(x) - 1)
    _, vecs = np.linalg.eigh(cov)
    proj = xc @ vecs[:, -1]
    order = np.argsort(proj)
    k = max(1, len(x) // 3)
    c1 = x[order[:k]].mean(0)
    c2 = x[order[-k:]].mean(0)
    assign = np.zeros(len(x), bool)
    for _ in range(iters):
        d1 = ((x - c1) ** 2).sum(1)
        d2 = ((x - c2) ** 2).sum(1)
        new = d2 < d1
        if np.array_equal(new, assign) and _ > 0:
            break
        assign = new
        if assign.all() or not assign.any():
            break
        c1 = x[~assign].mean(0)
        c2 = x[assign].mean(0)
    return c1.astype(np.float32), c2.astype(np.float32), assign


def is_bimodal(sample: np.ndarray, c1: np.ndarray, c2: np.ndarray, valley: float = SPLIT_VALLEY) -> bool:
    """True when the density of `sample` projected onto the c1->c2 axis has a clear valley
    between the two modes: the minimum bin count between 25 % and 75 % of the way is below
    `valley` times the smaller of the two side peaks. A linear gradient (flat histogram)
    or a single blob (peak in the middle) is rejected."""
    axis = (c2 - c1).astype(np.float64)
    l2 = float(axis @ axis)
    if l2 < 1e-6 or len(sample) < 32:
        return False
    t = ((sample.astype(np.float64) - c1) @ axis) / l2
    lo, hi = np.percentile(t, [1, 99])
    if hi - lo < 1e-6:
        return False
    hist, edges = np.histogram(t, bins=16, range=(float(lo), float(hi)))
    centers = 0.5 * (edges[1:] + edges[:-1])
    left = hist[centers < 0.25]
    mid = hist[(centers >= 0.25) & (centers <= 0.75)]
    right = hist[centers > 0.75]
    if not (len(left) and len(mid) and len(right)):
        return False
    return float(mid.min()) < valley * float(min(left.max(), right.max()))


# ---------------------------------------------------------------------- step 1: paint

def _paint_masks(masks: list[dict], lab: np.ndarray, min_px: int, progress: Progress
                 ) -> tuple[np.ndarray, list[dict], list[np.ndarray]]:
    h, w = lab.shape[:2]
    labels = np.full((h, w), -1, np.int32)
    info: list[dict] = []
    medians: list[np.ndarray] = []
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i]["area"]))
    dropped_dup = 0
    for k, i in enumerate(order):
        m = masks[i]
        if int(m["area"]) < min_px:
            break
        seg = np.asarray(m["segmentation"], dtype=bool)
        if seg.shape != (h, w):
            raise ValueError("mask shape does not match the albedo")
        ys, xs = _mask_crop(m.get("bbox", [0, 0, w, h]), (h, w))
        crop_seg = seg[ys, xs]
        crop_lbl = labels[ys, xs]
        pix = lab[ys, xs][crop_seg]
        if len(pix) < min_px:
            continue
        med = _median(pix)
        under = crop_lbl[crop_seg]
        under = under[under >= 0]
        if under.size:
            cnt = np.bincount(under)
            c = int(cnt.argmax())
            if cnt[c] > DUPLICATE_OVERLAP * len(pix) and _de(med, medians[c]) < DUPLICATE_DELTA_E:
                dropped_dup += 1
                continue
        rid = len(info)
        crop_lbl[crop_seg] = rid
        info.append({"id": rid, "source": "sam", "confidence": float(m.get("predicted_iou", 0.0))})
        medians.append(med)
        if progress is not None and (k % 64 == 0):
            progress(0.05 + 0.30 * k / max(1, len(order)), f"Painting parts · {len(info)} regions")
    if progress is not None:
        progress(0.35, f"Painted {len(info)} regions ({dropped_dup} duplicates dropped)")
    return labels, info, medians


# ---------------------------------------------------------------------- step 2: split

def _split_bimodal(labels: np.ndarray, lab: np.ndarray, info: list[dict], medians: list[np.ndarray],
                   min_px: int, progress: Progress) -> int:
    from scipy import ndimage
    n = len(info)
    if n == 0:
        return 0
    objs = ndimage.find_objects(labels + 1, max_label=n)
    n_split = 0
    for rid in range(n):
        sl = objs[rid]
        if sl is None:
            continue
        crop = labels[sl] == rid
        area = int(crop.sum())
        if area < 2 * min_px:
            continue
        pix = lab[sl][crop]
        sample = _sub(pix, _KMEANS_SAMPLE)
        c1, c2, a_s = two_means(sample)
        frac = float(a_s.mean())
        if min(frac, 1.0 - frac) < SPLIT_MIN_FRACTION:
            continue
        if _de(c1, c2) < SPLIT_DELTA_E or not is_bimodal(sample, c1, c2):
            continue
        d1 = ((pix - c1) ** 2).sum(1)
        d2 = ((pix - c2) ** 2).sum(1)
        amap = np.zeros(crop.shape, np.float32)
        amap[crop] = (d2 < d1).astype(np.float32)
        num = cv2.blur(amap, (5, 5))
        den = cv2.blur(crop.astype(np.float32), (5, 5))
        second = crop & (num > 0.5 * den)
        first = crop & ~second
        n1, n2 = int(first.sum()), int(second.sum())
        if min(n1, n2) < min_px:
            continue
        if n2 > n1:  # the larger part keeps the id
            first, second = second, first
        new = len(info)
        view = labels[sl]
        view[second] = new
        info.append({"id": new, "source": "split", "confidence": info[rid]["confidence"]})
        medians[rid] = _median(lab[sl][first])
        medians.append(_median(lab[sl][second]))
        n_split += 1
    if progress is not None:
        progress(0.5, f"Split {n_split} two-tone regions")
    return n_split


# ---------------------------------------------------------------------- merging passes

def _merge_pass(labels: np.ndarray, lab: np.ndarray, info: list[dict], sources_mask: np.ndarray,
                target_ok: np.ndarray, max_de: float | None) -> tuple[np.ndarray, list[dict], int]:
    """One pass: every region flagged in `sources_mask` merges into its touching region
    (flagged in `target_ok`) with the nearest median albedo, when the CIEDE2000 distance is
    below `max_de` (None = always). Each source merges at most once per pass; the target's
    info survives. Returns (labels, info, number of merges) with ids compacted."""
    n = len(info)
    med = region_medians(labels, lab, n)
    pairs, blen = adjacency(labels, n)
    if len(pairs) == 0:
        return labels, info, 0
    a, b = pairs[:, 0], pairs[:, 1]
    src = np.concatenate([a, b])
    tgt = np.concatenate([b, a])
    keep = sources_mask[src] & target_ok[tgt]
    src, tgt = src[keep], tgt[keep]
    if src.size == 0:
        return labels, info, 0
    de = imageio.delta_e(med[src], med[tgt])
    de = np.where(np.isfinite(de), de, np.inf)
    order = np.lexsort((de, src))
    src, tgt, de = src[order], tgt[order], de[order]
    first = np.ones(len(src), bool)
    first[1:] = src[1:] != src[:-1]
    src, tgt, de = src[first], tgt[first], de[first]
    order = np.argsort(de, kind="stable")
    parent = np.arange(n)
    merges = 0
    for s, t, d in zip(src[order].tolist(), tgt[order].tolist(), de[order].tolist()):
        if max_de is not None and d >= max_de:
            break
        rt = find_root(parent, t)
        if rt == s:
            continue
        if rt != t:
            d = _de(med[s], med[rt])
            if max_de is not None and d >= max_de:
                continue
        parent[s] = rt
        merges += 1
    if merges == 0:
        return labels, info, 0
    roots = roots_of(parent)
    labels = np.where(labels >= 0, roots[np.clip(labels, 0, n - 1)], -1).astype(np.int32)
    labels, mapping = compact(labels, n)
    new_info = [None] * int(mapping.max() + 1)
    for old, new in enumerate(mapping.tolist()):
        if new >= 0 and roots[old] == old:
            new_info[new] = dict(info[old], id=new)
    return labels, [d for d in new_info if d is not None], merges


# ---------------------------------------------------------------------- step 3: fill

def _fill_unlabeled(labels: np.ndarray, image: np.ndarray, lab: np.ndarray, info: list[dict],
                    superpixel_px: float, progress: Progress) -> tuple[np.ndarray, list[dict]]:
    unl = labels < 0
    if not unl.any():
        return labels, info
    h, w = labels.shape
    sp = slic_labels(image, max(16, int(h * w / superpixel_px)))
    base = len(info)
    uniq, inv = np.unique(sp[unl], return_inverse=True)
    labels = labels.copy()
    labels[unl] = base + inv.astype(np.int32)
    info = list(info) + [{"id": base + k, "source": "superpixel", "confidence": 0.0} for k in range(len(uniq))]
    if progress is not None:
        progress(0.55, f"Filling gaps · {len(uniq)} superpixels")
    # Pass 0 lets superpixels join the SAM regions they touch; later passes also let the
    # leftover superpixels coalesce with each other (an unmasked background becomes one
    # region instead of a mosaic).
    for p in range(MAX_MERGE_PASSES):
        src = np.array([d["source"] == "superpixel" for d in info], bool)
        if not src.any():
            break
        tgt = ~src if p == 0 else np.ones(len(info), bool)
        labels, info, merges = _merge_pass(labels, lab, info, src, tgt, FILL_DELTA_E)
        if merges == 0 and p > 0:
            break
    return labels, info


# ---------------------------------------------------------------------- step 4: specks

def _remove_specks(labels: np.ndarray, lab: np.ndarray, info: list[dict], speck_px: int,
                   progress: Progress) -> tuple[np.ndarray, list[dict]]:
    for _ in range(MAX_MERGE_PASSES):
        areas = region_areas(labels, len(info))
        small = areas < speck_px
        if not small.any():
            break
        labels, info, merges = _merge_pass(labels, lab, info, small, np.ones(len(info), bool), None)
        if merges == 0:
            break
    return labels, info


def _refine_edges(labels: np.ndarray, image: np.ndarray, radius: int) -> np.ndarray:
    """Snap region boundaries to image edges with the guided filter (GPU).

    Skipped (labels returned unchanged) when the image is too thin for the filter
    window or when the shared filter cannot handle the shape: `filters._to_t` reads a
    1xWx3 guide as a [1, W, 3] channel-first tensor, so degenerate 1-pixel-tall images
    would raise inside the guided filter. Refinement is cosmetic, so the unrefined
    partition is always an acceptable result."""
    h, w = labels.shape
    if min(h, w) <= 2 * radius + 1:
        return labels
    try:
        return filters.refine_labels_with_guide(labels, imageio.to_float(image), radius=radius)
    except RuntimeError:
        return labels


# ---------------------------------------------------------------------- public

def build_regions(image_rgb_u8: np.ndarray, albedo_lin: np.ndarray, masks: list[dict],
                  detail: str = "balanced", progress: Progress = None) -> tuple[np.ndarray, list[dict]]:
    """Partition the image into regions from SAM proposals.

    Returns `(labels, info)`: `labels` is int32 HxW with every pixel in 0..N-1 (no -1
    survives, ids are contiguous), `info[i]` describes region `i` with keys `id`,
    `source` ('sam' | 'superpixel' | 'split'), `confidence` (SAM predicted IoU, 0 for
    superpixels), `area` (pixels) and `albedo_lab` (median albedo, CIE Lab tuple).
    Works with an empty proposal list (the image is then partitioned by superpixels).
    Regions may be disconnected but never overlap. `albedo_lin` is float32 linear HxWx3.
    """
    if detail not in REGION_PRESETS:
        raise ValueError(f"unknown detail preset {detail!r}; choose from {sorted(REGION_PRESETS)}")
    if image_rgb_u8.shape[:2] != albedo_lin.shape[:2]:
        raise ValueError("image and albedo must have the same height and width")
    preset = REGION_PRESETS[detail]
    h, w = image_rgb_u8.shape[:2]
    npx = h * w
    min_px = max(16, int(round(preset["min_area_frac"] * npx)))
    lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))

    labels, info, medians = _paint_masks(masks, lab, min_px, progress)
    _split_bimodal(labels, lab, info, medians, min_px, progress)
    labels, info = _fill_unlabeled(labels, image_rgb_u8, lab, info, preset["superpixel_px"], progress)
    if REFINE_RADIUS > 0 and len(info) > 1:
        labels = _refine_edges(labels, image_rgb_u8, REFINE_RADIUS)
    if progress is not None:
        progress(0.85, "Removing specks")
    labels, info = _remove_specks(labels, lab, info, max(4, min_px // 4), progress)
    labels, mapping = compact(labels, len(info))
    info = [dict(info[old], id=int(new)) for old, new in enumerate(mapping.tolist()) if new >= 0]
    n = len(info)
    if n == 0 or (labels < 0).any():
        raise AssertionError("build_regions left unlabeled pixels")
    areas = region_areas(labels, n)
    meds = region_medians(labels, lab, n)
    for d in info:
        i = d["id"]
        d["area"] = int(areas[i])
        d["albedo_lab"] = tuple(float(v) for v in meds[i])
    if progress is not None:
        progress(1.0, f"{n} regions")
    return labels, info
