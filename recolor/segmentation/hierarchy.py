"""From overlapping SAM proposals to a clean partition of the image into regions.

`build_regions` implements the four steps of docs/ARCHITECTURE.md §3.2:

1. paint proposals largest-first (parts override wholes), dropping tiny masks and
   duplicate proposals (same albedo as the region they sit in);
2. split regions whose albedo is clearly bimodal (two-tone parts SAM returns as one);
3. label what SAM missed with SLIC superpixels, merging each into the touching region
   with the nearest median albedo when the colors agree;
4. absorb specks into their most similar neighbour, then relabel to 0..N-1;
5. re-examine the regions step 2 produced: the smaller half of a split is never looked at
   again by step 2, so a mixed region survived (the yellow BMW's nose and every blown
   specular of the bike in one khaki region that no colour rule could group with the
   paint). Step 2's own test runs again on those regions for up to RESPLIT_ROUNDS rounds.

Two additions keep the small and thin parts the steps above lose:

- a *distinct small* proposal (below the preset's minimum area but at least SMALL_MIN_PX
  and more than SMALL_RING_DE from its own SMALL_RING_PX outer ring: an indicator lens, a
  decal on a sprue) is painted anyway in step 1 (source 'small'). It is *exempt*: never
  split, never merged away as a speck and never eroded by the edge snap. A remnant that a
  later, smaller proposal or a stamped extra leaves below SMALL_MIN_PX loses the exemption
  (the slivers of a word mask between its letters merged into the paint they are).
- *extra masks* from :mod:`smallparts` (lettering from Florence-2's OCR, wheels split into
  tyre and rim, named parts) are stamped after step 3 (:func:`_stamp_extras`): text over
  everything but other text (exempt too); a wheel's tyre and rim only over regions that
  reach outside the wheel in the wheel's own colour or hold both tyre and rim (the disc,
  the sprocket, a gold caliper are kept); a named part only where it splits a region into
  two parts whose albedo or photo lightness differ.

:func:`cut_on_matte` (the regions stage runs it after part recovery) cuts a region along the
foreground matte's contour when both sides are substantial, so the backdrop side and the
object side become separate regions and no group has to mix them; a compact object piece
well inside the silhouette (a reservoir the matte sees and SAM's backdrop mask swallowed)
is cut out of a backdrop region even when it is a small share of it.

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
RESPLIT_ROUNDS = 3         # step 5: passes of the bimodal split over the regions step 2 produced
_MEDIAN_SAMPLE = 20_000    # pixels used for a per-mask median
_KMEANS_SAMPLE = 4_000     # pixels used to fit the 2-means of a region
# Distinct small proposals (step 1): a proposal below the preset's min_px is still painted when
# it has at least SMALL_MIN_PX pixels and its median albedo is more than SMALL_RING_DE (CIEDE2000)
# from the median of its SMALL_RING_PX outer ring. No duplicate test: the BMW's rear indicator
# is 65 dE from its ring and 4 dE from the machinery mask it sits in.
SMALL_MIN_PX = 60
SMALL_RING_PX = 3
SMALL_RING_DE = 15.0
#: Sources of the regions the exemption applies to (never split, merged as a speck or eroded by
#: the snap) and the minimum area at which each keeps it.
EXEMPT_FLOOR = {"small": SMALL_MIN_PX, "text": 12}
# Stamped extras: a wheel part takes pixels only from a region that reaches outside the wheel by
# WHEEL_OUTSIDE of its area in the wheel's own colour (within WHEEL_DE of the tyre's or the
# rim's median) or that holds at least WHEEL_BOTH of both parts; a named part is stamped only
# where it leaves at least NAMED_MIN_FRAC of a region (and NAMED_MIN_PX) on each side and the
# sides differ by more than NAMED_DE in albedo or NAMED_DL in photo lightness. A stamped part
# below STAMP_MIN_PX is dropped.
WHEEL_OUTSIDE = 0.1
WHEEL_DE = 15.0
WHEEL_BOTH = 0.2
NAMED_MIN_FRAC = 0.1
NAMED_MIN_PX = 64
NAMED_DE = 8.0
NAMED_DL = 12.0
STAMP_MIN_PX = 150
#: The matte cut: a region with at least MATTE_SHARE of its pixels on each side of the matte's
#: MATTE_THRESHOLD contour is cut along it. A region mostly on the backdrop side also gives up
#: every compact object piece the matte finds well inside the silhouette: an 8-connected
#: component of at least MATTE_CC_MIN_PX px that holds a MATTE_CC_CORE_PX-wide disk (the
#: yellow BMW's brake-fluid reservoir, 1.4k px in a 620k px backdrop region, could never reach
#: 20 % of it and was locked with the backdrop); a sliver along the silhouette, where the
#: matte and SAM disagree by a few px, has no such core and stays. At most MATTE_CC_MAX pieces
#: per region, the largest first.
MATTE_SHARE = 0.2
MATTE_THRESHOLD = 0.5
MATTE_CC_MIN_PX = 150
MATTE_CC_CORE_PX = 7
MATTE_CC_MAX = 16


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


def _ring_median(lab: np.ndarray, seg: np.ndarray, ys: slice, xs: slice, ring: int) -> Optional[np.ndarray]:
    """Median Lab of the ``ring``-px outer ring of a full-size mask around its bbox crop."""
    h, w = seg.shape
    y0, y1 = max(0, ys.start - ring - 1), min(h, ys.stop + ring + 1)
    x0, x1 = max(0, xs.start - ring - 1), min(w, xs.stop + ring + 1)
    s = seg[y0:y1, x0:x1].astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    r = cv2.dilate(s, k).astype(bool) & ~s.astype(bool)
    if r.sum() < 4:
        return None
    return _median(lab[y0:y1, x0:x1][r])


def _exempt(info: list[dict], areas: Optional[np.ndarray] = None) -> np.ndarray:
    """bool [N]: the regions that keep the exemption (source in EXEMPT_FLOOR and, when
    ``areas`` is given, at least their floor's pixels)."""
    out = np.zeros(len(info), bool)
    for i, d in enumerate(info):
        floor = EXEMPT_FLOOR.get(d.get("source"))
        if floor is not None and d.get("exempt", True):
            out[i] = areas is None or areas[i] >= floor
    return out


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
    n_small = 0
    for k, i in enumerate(order):
        m = masks[i]
        small = int(m["area"]) < min_px
        if small and int(m["area"]) < SMALL_MIN_PX:
            break                                   # sorted by area: everything after is smaller
        seg = np.asarray(m["segmentation"], dtype=bool)
        if seg.shape != (h, w):
            raise ValueError("mask shape does not match the albedo")
        ys, xs = _mask_crop(m.get("bbox", [0, 0, w, h]), (h, w))
        crop_seg = seg[ys, xs]
        crop_lbl = labels[ys, xs]
        pix = lab[ys, xs][crop_seg]
        if small:
            # a distinct small proposal: painted when it stands out from its own surroundings
            if len(pix) < SMALL_MIN_PX:
                continue
            rmed = _ring_median(lab, seg, ys, xs, SMALL_RING_PX)
            if rmed is None:
                continue
            med = _median(pix)
            paint_seg = crop_seg
            if len(pix) >= 2 * SMALL_MIN_PX:
                # two tones, one of them the surroundings (thin lettering on the paint with the
                # paint showing between the strokes): only the other tone is the part. No valley
                # test: lettering a few px tall is anti-aliased into a continuum, the two
                # centroids and the ring tell the case apart on their own.
                sample = _sub(pix, _KMEANS_SAMPLE)
                c1, c2, _ = two_means(sample)
                if _de(c1, c2) >= SPLIT_DELTA_E:
                    near, far = (c1, c2) if _de(c1, rmed) <= _de(c2, rmed) else (c2, c1)
                    if _de(near, rmed) <= SMALL_RING_DE < _de(far, rmed):
                        keep = ((pix - far) ** 2).sum(1) < ((pix - near) ** 2).sum(1)
                        if keep.sum() < SMALL_MIN_PX:
                            continue
                        paint_seg = crop_seg.copy()
                        paint_seg[crop_seg] = keep
                        med = _median(pix[keep])
            if _de(med, rmed) <= SMALL_RING_DE:
                continue
            rid = len(info)
            crop_lbl[paint_seg] = rid
            info.append({"id": rid, "source": "small", "exempt": True, "confidence": float(m.get("predicted_iou", 0.0))})
            medians.append(med)
            n_small += 1
            continue
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
        small_note = f", {n_small} small distinct parts" if n_small else ""
        progress(0.35, f"Painted {len(info)} regions ({dropped_dup} duplicates dropped{small_note})")
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
        if sl is None or info[rid].get("source") in EXEMPT_FLOOR:
            continue                                # a small distinct or stamped region is one colour by construction
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
        # an exempt region is never merged away; a remnant below its floor is a speck
        small = (areas < speck_px) & ~_exempt(info, areas)
        for i, d in enumerate(info):
            if d.get("source") in EXEMPT_FLOOR and areas[i] < EXEMPT_FLOOR[d["source"]]:
                small[i] = True
        if not small.any():
            break
        labels, info, merges = _merge_pass(labels, lab, info, small, np.ones(len(info), bool), None)
        if merges == 0:
            break
    return labels, info


def _resplit(labels: np.ndarray, lab: np.ndarray, info: list[dict], min_px: int) -> int:
    """Step 5: run step 2's bimodal split again on the regions it produced (source
    'split', including those this step adds), up to RESPLIT_ROUNDS rounds, in place on the
    contiguous `labels` / `info`. New ids are appended; a split parent keeps its id and at
    least `min_px` pixels, so the partition stays contiguous. Returns the regions added."""
    added = 0
    for _ in range(RESPLIT_ROUNDS):
        cand = np.array([i for i, d in enumerate(info) if d.get("source") == "split"], np.int64)
        if cand.size == 0:
            break
        work = np.where(np.isin(labels, cand), labels, -1).astype(np.int32)
        medians = [np.zeros(3, np.float32) for _ in info]   # step 2 does not read them
        n_split = _split_bimodal(work, lab, info, medians, min_px, None)
        if n_split == 0:
            break
        moved = work >= 0
        labels[moved] = work[moved]
        added += n_split
    return added


# ---------------------------------------------------------------------- extras: lettering, wheels, named parts

def _photo_lightness(image_rgb_u8: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.ascontiguousarray(image_rgb_u8), cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[..., 0] * (100.0 / 255.0)


def _named_accept(mask: np.ndarray, labels: np.ndarray, info: list[dict], lab: np.ndarray,
                  photo_l: np.ndarray) -> np.ndarray:
    """The pixels of ``mask`` that split an existing (non-exempt) region into an inside and
    an outside of at least NAMED_MIN_FRAC of it each (and NAMED_MIN_PX) whose albedo medians
    differ by more than NAMED_DE or whose photo lightness medians differ by more than
    NAMED_DL: a real part of that region, not a phrase box on a uniform surface."""
    new = np.zeros(mask.shape, bool)
    ids, cnt = np.unique(labels[mask], return_counts=True)
    for r, c in zip(ids.tolist(), cnt.tolist()):
        if r < 0 or info[r].get("source") in EXEMPT_FLOOR or c < NAMED_MIN_PX:
            continue
        region = labels == r
        inside = region & mask
        n_r = int(region.sum())
        n_out = n_r - int(c)
        if n_out < NAMED_MIN_PX or min(int(c), n_out) < NAMED_MIN_FRAC * n_r:
            continue
        outside = region & ~mask
        d_alb = _de(_median(lab[inside]), _median(lab[outside]))
        d_l = abs(float(np.median(photo_l[inside])) - float(np.median(photo_l[outside])))
        if d_alb > NAMED_DE or d_l > NAMED_DL:
            new |= inside
    return new


def _stamp_extras(labels: np.ndarray, info: list[dict], lab: np.ndarray, image_rgb_u8: np.ndarray,
                  extra: list) -> tuple[np.ndarray, list[dict]]:
    """Stamp the extra masks onto a filled partition (ids stay contiguous; see the module
    docstring for the three rules). ``extra`` records carry ``mask``, ``source`` ('text' |
    'wheel' | 'named') and, for a wheel, ``parts`` = (tyre, rim)."""
    labels = labels.copy()
    info = [dict(d) for d in info]
    # 1. lettering: over everything but other text, larger words first
    texts = sorted((e for e in extra if e.source == "text"), key=lambda e: -int(e.mask.sum()))
    for e in texts:
        is_text = np.array([d.get("source") == "text" for d in info] + [False], bool)
        m = e.mask & ~is_text[np.clip(labels, 0, len(info) - 1)]
        if m.sum() < EXEMPT_FLOOR["text"]:
            continue
        rid = len(info)
        labels[m] = rid
        info.append({"id": rid, "source": "text", "exempt": True, "confidence": 1.0})
    # 2. wheels: tyre and rim from the regions the wheel rule allows
    for e in (e for e in extra if e.source == "wheel" and e.parts is not None):
        tyre, rim = e.parts
        wheel = e.mask
        med_t = _median(lab[tyre]) if tyre.any() else None
        med_m = _median(lab[rim]) if rim.any() else None
        take = np.zeros(len(info), bool)
        for r in np.unique(labels[wheel]).tolist():
            if info[r].get("source") in EXEMPT_FLOOR:
                continue
            region = labels == r
            nr = int(region.sum())
            n_w = int((region & wheel).sum())
            n_t, n_m = int((region & tyre).sum()), int((region & rim).sum())
            mr = _median(lab[region])
            close = min(_de(mr, med_t) if med_t is not None else 99.0,
                        _de(mr, med_m) if med_m is not None else 99.0) < WHEEL_DE
            # a region reaching outside the wheel in the wheel's own colour (the BMW's black
            # machinery) gives up its wheel pixels; a region inside the wheel (disc, sprocket,
            # an existing tyre) or of another colour (a gold caliper) is kept, unless it holds
            # both tyre and rim
            if (nr - n_w >= WHEEL_OUTSIDE * nr and close) or (n_t >= WHEEL_BOTH * nr and n_m >= WHEEL_BOTH * nr):
                take[r] = True
        for part in (tyre, rim):
            new = part & take[np.clip(labels, 0, len(take) - 1)]
            if new.sum() >= STAMP_MIN_PX:
                rid = len(info)
                labels[new] = rid
                info.append({"id": rid, "source": "wheel", "confidence": 1.0})
                take = np.append(take, False)
    # 3. named parts: only where they split a region into two different parts
    named = sorted((e for e in extra if e.source == "named"), key=lambda e: -int(e.mask.sum()))
    if named:
        photo_l = _photo_lightness(image_rgb_u8)
        for e in named:
            new = _named_accept(e.mask, labels, info, lab, photo_l)
            if new.sum() < STAMP_MIN_PX:
                continue
            rid = len(info)
            labels[new] = rid
            info.append({"id": rid, "source": "named", "confidence": 1.0})
    # stamping can empty a region: keep the map contiguous for the later passes
    labels, mapping = compact(labels, len(info))
    info = [dict(info[old], id=int(new)) for old, new in enumerate(mapping.tolist()) if new >= 0]
    return labels, info


def _refine_edges(labels: np.ndarray, image: np.ndarray, radius: int,
                  frozen: Optional[np.ndarray] = None) -> np.ndarray:
    """Snap region boundaries to image edges with the guided filter (GPU).

    Skipped (labels returned unchanged) when the image is too thin for the filter
    window or when the shared filter cannot handle the shape: `filters._to_t` reads a
    1xWx3 guide as a [1, W, 3] channel-first tensor, so degenerate 1-pixel-tall images
    would raise inside the guided filter. Refinement is cosmetic, so the unrefined
    partition is always an acceptable result. Pixels of the ``frozen`` region ids (and
    pixels the snap would give to them) keep their label: the snap erased small and thin
    regions."""
    h, w = labels.shape
    if min(h, w) <= 2 * radius + 1:
        return labels
    try:
        out = filters.refine_labels_with_guide(labels, imageio.to_float(image), radius=radius)
    except RuntimeError:
        return labels
    if frozen is not None and len(frozen):
        fr = np.zeros(int(max(labels.max(), out.max())) + 1, bool)
        fr[frozen[frozen < len(fr)]] = True
        out = np.where(fr[labels] | fr[out], labels, out).astype(np.int32)
    return out


# ---------------------------------------------------------------------- public

def build_regions(image_rgb_u8: np.ndarray, albedo_lin: np.ndarray, masks: list[dict],
                  detail: str = "balanced", progress: Progress = None,
                  extra: Optional[list] = None) -> tuple[np.ndarray, list[dict]]:
    """Partition the image into regions from SAM proposals.

    Returns `(labels, info)`: `labels` is int32 HxW with every pixel in 0..N-1 (no -1
    survives, ids are contiguous), `info[i]` describes region `i` with keys `id`,
    `source` ('sam' | 'superpixel' | 'split' | 'small' | 'text' | 'wheel' | 'named'),
    `confidence` (SAM predicted IoU, 0 for superpixels), `area` (pixels) and `albedo_lab`
    (median albedo, CIE Lab tuple); 'small' and 'text' regions carry `exempt: True`.
    Works with an empty proposal list (the image is then partitioned by superpixels).
    Regions may be disconnected but never overlap. `albedo_lin` is float32 linear HxWx3.
    ``extra`` (records with ``mask`` (bool HxW), ``source`` and, for a wheel, ``parts`` =
    (tyre, rim): :class:`smallparts.Extra`) are stamped after the fill (:func:`_stamp_extras`).
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
    if extra:
        labels, info = _stamp_extras(labels, info, lab, image_rgb_u8, extra)
        if progress is not None:
            n_extra = sum(1 for d in info if d.get("source") in ("text", "wheel", "named"))
            progress(0.8, f"Stamped {n_extra} lettering and named-part regions")
    if REFINE_RADIUS > 0 and len(info) > 1:
        frozen = np.array([d["id"] for d in info if d.get("source") in EXEMPT_FLOOR], np.int64)
        labels = _refine_edges(labels, image_rgb_u8, REFINE_RADIUS, frozen)
    if progress is not None:
        progress(0.85, "Removing specks")
    labels, info = _remove_specks(labels, lab, info, max(4, min_px // 4), progress)
    labels, mapping = compact(labels, len(info))
    info = [dict(info[old], id=int(new)) for old, new in enumerate(mapping.tolist()) if new >= 0]
    if _resplit(labels, lab, info, min_px) and progress is not None:
        progress(0.95, "Re-split mixed regions")
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


# ---------------------------------------------------------------------- the matte cut

def _object_pieces(labels: np.ndarray, fgm: np.ndarray, dominant: np.ndarray, min_px: int, core_px: int,
                   max_per_region: int) -> list[tuple[int, np.ndarray]]:
    """The compact object pieces inside backdrop-dominant regions: 8-connected components of
    ``fgm`` within the regions ``dominant`` marks, per region, of at least ``min_px`` px that
    survive an erosion by a ``core_px`` disk (padded, so a piece's own bbox edge is not a
    core). Returns ``[(parent region id, bool HxW mask)]``, largest first."""
    cand = fgm & dominant[labels]
    if not cand.any():
        return []
    ncc, cc, stt, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_px, core_px))
    pad = core_px
    taken: dict[int, int] = {}
    pieces: list[tuple[int, np.ndarray]] = []
    order = np.argsort(-stt[1:, cv2.CC_STAT_AREA], kind="stable") + 1
    for c in order.tolist():
        if stt[c, cv2.CC_STAT_AREA] < min_px:
            break
        x, y = int(stt[c, cv2.CC_STAT_LEFT]), int(stt[c, cv2.CC_STAT_TOP])
        w, h = int(stt[c, cv2.CC_STAT_WIDTH]), int(stt[c, cv2.CC_STAT_HEIGHT])
        comp = cc[y:y + h, x:x + w] == c
        lab_crop = labels[y:y + h, x:x + w]
        for r in np.unique(lab_crop[comp]).tolist():
            piece = comp & (lab_crop == r)
            if int(piece.sum()) < min_px or taken.get(r, 0) >= max_per_region:
                continue
            padded = np.pad(piece.astype(np.uint8), pad)
            if not cv2.erode(padded, kernel).any():
                continue                                  # a sliver along the silhouette
            taken[r] = taken.get(r, 0) + 1
            full = np.zeros(labels.shape, bool)
            full[y:y + h, x:x + w] = piece
            pieces.append((int(r), full))
    return pieces


def cut_on_matte(labels: np.ndarray, info: list[dict], fg: np.ndarray, albedo_lin: Optional[np.ndarray] = None,
                 share: float = MATTE_SHARE, threshold: float = MATTE_THRESHOLD) -> tuple[np.ndarray, list[dict], int]:
    """Cut every region that has at least ``share`` of its pixels on each side of the
    foreground matte's ``threshold`` contour along it: the object side becomes a new region
    (appended ids; the parent's source for a recovered part, else 'split'), the backdrop
    side keeps the id. A region below ``share`` on the object side gives up its compact
    object pieces too (see MATTE_CC_MIN_PX: one new region per piece). Exempt regions
    (lettering, small distinct parts) are left whole. Returns ``(labels, info, n_cut)`` in
    :func:`build_regions`' format (``area`` and ``albedo_lab`` refreshed when ``albedo_lin``
    is given), ``n_cut`` counting the cut regions and the pieces; the inputs are not
    modified."""
    labels = np.ascontiguousarray(labels, np.int32)
    n = int(labels.max()) + 1
    if fg.shape != labels.shape:
        raise ValueError("the matte must have the label map's shape")
    fgm = np.asarray(fg, np.float32) >= threshold
    cnt = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    cf = np.bincount(labels[fgm].ravel(), minlength=n).astype(np.float64)
    frac = cf / np.maximum(cnt, 1.0)
    exempt = np.zeros(n, bool)
    for i, d in enumerate(info):
        if i < n and d.get("source") in EXEMPT_FLOOR:
            exempt[i] = True
    cut = (frac >= share) & (1.0 - frac >= share) & (cnt > 0) & ~exempt
    dominant = (frac < share) & (cf > 0) & (cnt > 0) & ~exempt
    pieces = _object_pieces(labels, fgm, dominant, MATTE_CC_MIN_PX, MATTE_CC_CORE_PX, MATTE_CC_MAX) if dominant.any() else []
    ids = np.flatnonzero(cut)
    if ids.size == 0 and not pieces:
        return labels.copy(), [dict(d) for d in info], 0
    lut = np.full(n, -1, np.int64)
    lut[ids] = np.arange(n, n + len(ids))
    out = labels.copy()
    m = fgm & cut[labels]
    out[m] = lut[labels[m]].astype(np.int32)
    new_info = [dict(d) for d in info]

    def child_of(r: int, new_id: int) -> dict:
        base = dict(info[r])
        base["id"] = new_id
        if base.get("source") not in (PART_SOURCE, "prompt"):
            base["source"] = "split"
        base.pop("exempt", None)
        return base

    for r in ids.tolist():
        new_info.append(child_of(r, int(lut[r])))
    for r, mask in pieces:
        new_id = len(new_info)
        out[mask] = new_id
        new_info.append(child_of(r, new_id))
    n_cut = int(ids.size) + len(pieces)
    out, mapping = compact(out, len(new_info))
    out = out.astype(np.int32)
    new_info = [dict(new_info[old], id=int(new)) for old, new in enumerate(mapping.tolist()) if new >= 0]
    if albedo_lin is not None:
        lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
        areas = region_areas(out, len(new_info))
        meds = region_medians(out, lab, len(new_info))
        for d in new_info:
            d["area"] = int(areas[d["id"]])
            d["albedo_lab"] = tuple(float(v) for v in meds[d["id"]])
    return out, new_info, n_cut


# ---------------------------------------------------------------------- part recovery

# SAM's automatic proposals at the balanced preset miss small parts of their own colour
# inside a large neutral mask: the yellow BMW's gold anodised fork tube (about 700 px) sits
# inside the one mask of the black machinery, no proposal of it survives `min_area` (and
# none was made), so the tube was grouped as black, tinted blue by the reflection stage, and
# its lit edge was labelled paint. `recover_parts` finds such chromatic *pockets* in the
# neutral regions of a finished partition and asks SAM for the object under each with one
# point prompt on a PART_CROP_PX crop (SAM's full input resolution on a small area, which is
# what the `max` preset's crop layers buy for the whole image at three times the cost). A
# reflection of the paint in a chrome part does not pass: SAM returns the chrome part,
# whose median is neutral.
#
# Pocket: pixels with albedo chroma >= POCKET_CHROMA and CIEDE2000 >= POCKET_DE from their
# host region's median, in a host of median chroma < POCKET_HOST_CHROMA and >=
# POCKET_HOST_MIN_PX, 2x2-opened, 8-connected, >= POCKET_MIN_PX; the MAX_POCKETS largest.
POCKET_CHROMA = 20.0
POCKET_DE = 15.0
POCKET_MIN_PX = 60
POCKET_HOST_CHROMA = 12.0
POCKET_HOST_MIN_PX = 4000
MAX_POCKETS = 24
# A returned mask is a part when SAM's score is >= PART_MIN_SCORE, it does not touch the
# crop border (a clipped mask is a larger object), it covers >= PART_MIN_COVER of the
# pocket, it has PART_MIN_PX .. PART_MAX_FACTOR x the pocket's pixels, a median chroma >=
# PART_MIN_CHROMA and a median >= PART_MIN_DE from the host's, and it is not a region that
# already exists (>= PART_DUP_REGION of it inside one other region within PART_MIN_DE of
# its colour). It may take pixels from several regions: the Exia's red chin crystal was
# half in the white face and half in the blue armour's region. Of two parts overlapping by
# IoU > PART_DUP_IOU the better scored one is kept.
# Colour pockets (a second kind, in chromatic regions): SAM paints a large mask's colour mode
# as one region even when it is scattered over the whole image (the yellow BMW's paint
# region also held the gold preload adjuster and red cap of the far fork top, 800 px away
# from the fairing, so they were painted navy with the paint). A pocket is a connected piece
# of a region of median chroma >= COLOUR_HOST_CHROMA and >= COLOUR_HOST_MIN_PX, other than
# its largest piece, of >= COLOUR_POCKET_MIN_PX, whose median is >= COLOUR_POCKET_DE from the
# region's; the COLOUR_MAX_POCKETS largest are prompted like the others. A part found there
# must be chromatic too (PART_MIN_CHROMA) and >= COLOUR_PART_MIN_DE from the host, with at
# least COLOUR_PART_MIN_PX: a neutral blob inside the paint is split_decals' business.
COLOUR_HOST_CHROMA = 18.0
COLOUR_HOST_MIN_PX = 2000
COLOUR_POCKET_MIN_PX = 25
COLOUR_POCKET_DE = 8.0
COLOUR_MAX_POCKETS = 16
COLOUR_PART_MIN_DE = 12.0
COLOUR_PART_MIN_PX = 40
#: Region source of a part cut out of a chromatic region (a neutral host's part is 'prompt'):
#: the groups stage keeps it out of the paint (`refine.isolate_parts`).
PART_SOURCE = "part"
PART_CROP_PX = 192
PART_MIN_SCORE = 0.7
PART_MIN_COVER = 0.5
PART_DUP_REGION = 0.8
PART_MIN_PX = 150
PART_MAX_FACTOR = 40
PART_MIN_CHROMA = 18.0
PART_MIN_DE = 15.0
PART_DUP_IOU = 0.5

#: ``prompter(image_u8, points) -> candidates``: for every (x, y) point a list of
#: ``{"mask": bool crop, "x0": int, "y0": int, "score": float, "clipped": bool}``, i.e.
#: :meth:`recolor.segmentation.sam_masks.SamMasker.prompt_parts`.
Prompter = Callable[[np.ndarray, list[tuple[int, int]]], list[list[dict]]]


def find_pockets(labels: np.ndarray, albedo_lab: np.ndarray, max_pockets: int = MAX_POCKETS) -> list[dict]:
    """Chromatic pockets inside neutral regions (see POCKET_*), largest first: dicts with
    ``region`` (host id), ``point`` (x, y) the pocket pixel farthest from its edge, ``px``,
    and ``mask`` / ``x0`` / ``y0`` (bool crop of the pocket and its offset)."""
    from scipy import ndimage
    labels = np.ascontiguousarray(labels, np.int32)
    n = int(labels.max()) + 1
    if n <= 0:
        return []
    meds = region_medians(labels, albedo_lab, n)
    areas = region_areas(labels, n)
    chroma = np.hypot(albedo_lab[..., 1], albedo_lab[..., 2])
    objs = ndimage.find_objects(labels + 1, max_label=n)
    kernel = np.ones((2, 2), np.uint8)
    out: list[dict] = []
    for rid in range(n):
        sl = objs[rid]
        med = meds[rid]
        if sl is None or areas[rid] < POCKET_HOST_MIN_PX or not np.isfinite(med).all():
            continue
        if float(np.hypot(med[1], med[2])) >= POCKET_HOST_CHROMA:
            continue
        cand = (labels[sl] == rid) & (chroma[sl] >= POCKET_CHROMA)
        if int(cand.sum()) < POCKET_MIN_PX:
            continue
        pix = albedo_lab[sl][cand]
        far = imageio.delta_e(pix, np.broadcast_to(med.astype(np.float32), pix.shape)) >= POCKET_DE
        cand[cand] = far
        cand = cv2.dilate(cv2.erode(cand.astype(np.uint8), kernel), kernel, anchor=(0, 0))
        nb, cc, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        for i in range(1, nb):
            px = int(stats[i, cv2.CC_STAT_AREA])
            if px < POCKET_MIN_PX:
                continue
            bx, by, bw, bh = (int(v) for v in stats[i, :4])
            blob = (cc[by:by + bh, bx:bx + bw] == i)
            dist = cv2.distanceTransform(np.pad(blob.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
            yy, xx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            y0, x0 = sl[0].start + by, sl[1].start + bx
            out.append({"region": rid, "point": (int(x0 + xx), int(y0 + yy)), "px": px,
                        "mask": blob, "x0": int(x0), "y0": int(y0)})
    out.sort(key=lambda p: -p["px"])
    return out[:max_pockets]


def find_colour_pockets(labels: np.ndarray, albedo_lab: np.ndarray,
                        max_pockets: int = COLOUR_MAX_POCKETS) -> list[dict]:
    """Off-colour pieces of chromatic regions (see COLOUR_*), largest first, in
    :func:`find_pockets`' format plus ``"kind": "colour"``: every 8-connected piece of a
    region except its largest, whose median albedo is far from the region's."""
    from scipy import ndimage
    labels = np.ascontiguousarray(labels, np.int32)
    n = int(labels.max()) + 1
    if n <= 0:
        return []
    meds = region_medians(labels, albedo_lab, n)
    areas = region_areas(labels, n)
    objs = ndimage.find_objects(labels + 1, max_label=n)
    out: list[dict] = []
    for rid in range(n):
        sl = objs[rid]
        med = meds[rid]
        if sl is None or areas[rid] < COLOUR_HOST_MIN_PX or not np.isfinite(med).all():
            continue
        if float(np.hypot(med[1], med[2])) < COLOUR_HOST_CHROMA:
            continue
        m = (labels[sl] == rid).astype(np.uint8)
        nb, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if nb <= 2:
            continue
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        for i in range(1, nb):
            px = int(stats[i, cv2.CC_STAT_AREA])
            if i == main or px < COLOUR_POCKET_MIN_PX:
                continue
            bx, by, bw, bh = (int(v) for v in stats[i, :4])
            blob = cc[by:by + bh, bx:bx + bw] == i
            y0, x0 = sl[0].start + by, sl[1].start + bx
            pmed = np.median(albedo_lab[y0:y0 + bh, x0:x0 + bw][blob], axis=0)
            if _de(pmed, med) < COLOUR_POCKET_DE:
                continue
            dist = cv2.distanceTransform(np.pad(blob.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
            yy, xx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            out.append({"region": rid, "point": (int(x0 + xx), int(y0 + yy)), "px": px,
                        "mask": blob, "x0": int(x0), "y0": int(y0), "kind": "colour"})
    out.sort(key=lambda p: -p["px"])
    return out[:max_pockets]


def _accept_part(cand: dict, pocket: dict, labels: np.ndarray, albedo_lab: np.ndarray,
                 host_med: np.ndarray) -> Optional[np.ndarray]:
    """The candidate's full-size bool mask when it is a part of its own (PART_*; COLOUR_*
    for a pocket of a chromatic region), else None."""
    colour = pocket.get("kind") == "colour"
    min_px = COLOUR_PART_MIN_PX if colour else PART_MIN_PX
    min_de = COLOUR_PART_MIN_DE if colour else PART_MIN_DE
    if cand["clipped"] or float(cand["score"]) < PART_MIN_SCORE:
        return None
    crop = np.asarray(cand["mask"], bool)
    area = int(crop.sum())
    if area < min_px or area > PART_MAX_FACTOR * pocket["px"]:
        return None
    h, w = labels.shape
    x0, y0 = int(cand["x0"]), int(cand["y0"])
    y1, x1 = min(h, y0 + crop.shape[0]), min(w, x0 + crop.shape[1])
    full = np.zeros((h, w), bool)
    full[y0:y1, x0:x1] = crop[:y1 - y0, :x1 - x0]
    pm = np.zeros((h, w), bool)
    py0, px0 = pocket["y0"], pocket["x0"]
    pm[py0:py0 + pocket["mask"].shape[0], px0:px0 + pocket["mask"].shape[1]] = pocket["mask"]
    if (full & pm).sum() < PART_MIN_COVER * pocket["px"]:
        return None
    med = np.median(albedo_lab[full], axis=0)
    if float(np.hypot(med[1], med[2])) < PART_MIN_CHROMA or _de(med, host_med) < min_de:
        return None
    under = labels[full]
    ids, cnt = np.unique(under, return_counts=True)
    top = int(np.argmax(cnt))
    if ids[top] != pocket["region"] and cnt[top] >= PART_DUP_REGION * area:
        other = np.median(albedo_lab[labels == ids[top]], axis=0)
        if _de(med, other) < min_de:
            return None                                     # the part already has its region
    return full


def recover_parts(image_rgb_u8: np.ndarray, albedo_lin: np.ndarray, labels: np.ndarray, info: list[dict],
                  prompter: Prompter, progress: Progress = None) -> tuple[np.ndarray, list[dict], int]:
    """Give small parts SAM's automatic proposals missed a region of their own.

    Chromatic pockets of neutral regions (:func:`find_pockets`) and off-colour pieces of
    chromatic regions (:func:`find_colour_pockets`) are prompted through ``prompter`` and
    every accepted mask (see PART_*, COLOUR_*) is stamped onto the label map as a new region
    (source 'prompt', or PART_SOURCE for a part of a chromatic region; confidence = SAM's
    score), smaller parts over larger ones.
    Returns ``(labels, info, n_added)`` in :func:`build_regions`' format: int32, contiguous
    0..N-1 with no -1, ``info`` refreshed (``area``, ``albedo_lab``). The inputs are not
    modified; with no pocket, or nothing accepted, the partition is returned unchanged."""
    labels = np.ascontiguousarray(labels, np.int32)
    lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
    pockets = find_pockets(labels, lab) + find_colour_pockets(labels, lab)
    if not pockets:
        return labels.copy(), [dict(d) for d in info], 0
    if progress is not None:
        progress(0.2, f"Looking closer at {len(pockets)} small coloured spots")
    cands = prompter(image_rgb_u8, [p["point"] for p in pockets])
    meds = region_medians(labels, lab, int(labels.max()) + 1)
    accepted: list[tuple[float, np.ndarray, str]] = []
    for pocket, options in zip(pockets, cands):
        best: Optional[tuple[float, np.ndarray, str]] = None
        for c in options or []:
            full = _accept_part(c, pocket, labels, lab, meds[pocket["region"]])
            if full is not None and (best is None or float(c["score"]) > best[0]):
                best = (float(c["score"]), full, pocket.get("kind", "neutral"))
        if best is not None:
            accepted.append(best)
    kept: list[tuple[float, np.ndarray, str]] = []
    for score, full, kind in sorted(accepted, key=lambda t: -t[0]):
        if any((full & other).sum() > PART_DUP_IOU * (full | other).sum() for _, other, _ in kept):
            continue
        kept.append((score, full, kind))
    if not kept:
        return labels.copy(), [dict(d) for d in info], 0
    out = labels.copy()
    new_info = [dict(d) for d in info]
    for score, full, kind in sorted(kept, key=lambda t: -int(t[1].sum())):
        rid = len(new_info)
        out[full] = rid
        new_info.append({"id": rid, "source": PART_SOURCE if kind == "colour" else "prompt",
                         "confidence": round(score, 4)})
    out, mapping = compact(out, len(new_info))
    out = out.astype(np.int32)
    new_info = [dict(new_info[old], id=int(new)) for old, new in enumerate(mapping.tolist()) if new >= 0]
    n = len(new_info)
    areas = region_areas(out, n)
    meds = region_medians(out, lab, n)
    for d in new_info:
        d["area"] = int(areas[d["id"]])
        d["albedo_lab"] = tuple(float(v) for v in meds[d["id"]])
    if progress is not None:
        progress(1.0, f"{len(kept)} small part{'s' if len(kept) != 1 else ''} recovered")
    return out, new_info, len(kept)
