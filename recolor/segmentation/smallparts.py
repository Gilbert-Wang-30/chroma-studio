"""Small, thin and named parts the automatic proposals lose: lettering and named parts from
Florence-2's answers, prompted as SAM boxes, as extra masks for :func:`hierarchy.build_regions`.

Two pieces (the third of the round, thin ridges, was measured and rejected: rim lights along
silhouettes doubled existing edges and recovered nothing):

* **Lettering** (:func:`text_masks`): every OCR quad with at least TEXT_MIN_ALNUM ASCII letters
  or digits (Florence reads texture as Arabic-Indic digits and dashes otherwise), quads that
  overlap merged, expanded by TEXT_EXPAND px and prompted as a SAM box on a crop. SAM's mask is
  kept when it stays inside the quad, covers at most 90 % of it and differs from the rest of
  the quad by more than TEXT_DE; otherwise the minority 2-means albedo cluster inside the quad
  (the letters; nearest-centroid assignment leaves the anti-aliased rim with the host), when it
  is bimodal and more than TEXT_DE from the majority. The Ducati's "DUCATI 748" fairing decal
  and tank logo go from lost to 0.91 / 0.69 achievable IoU.
* **Named parts** (:func:`part_masks`): every phrase box of the grounding caption prompted as a
  SAM box on a crop (box + 8 % + 8 px), the best unclipped candidate. A "tire" or "wheel rim"
  box is a whole wheel to SAM (tyre + rim as one mask): :func:`wheels.split_wheel` cuts it
  into tyre and rim, and the three masks are stamped by the wheel rule of the hierarchy (a
  wheel that cannot be split adds nothing). Any other phrase mask is stamped only where it
  splits an existing region into two parts that differ (the hierarchy's named-part test).

Both run at Balanced and Max on the work image; the pieces cost about 1 s per 1536-px image
on top of Florence-2 itself. Everything here is CPU numpy plus the SAM prompts the caller
passes in (``prompter``: :meth:`SamMasker.prompt_boxes`); nothing loads a model.

Bounded work: overlapping quads are merged on their bounding boxes (union-find, no pixel
work), at most TEXT_MAX_DOMAINS lettering domains and PART_MAX_BOXES phrase boxes (the
largest) are prompted, in chunks of PROMPT_CHUNK, and :func:`find_extras` takes a wall-clock
budget after which the remaining prompts are skipped (logged), so a text-dense photo (a label
sheet, a poster wall: Florence-2 returns 350 quads for one) cannot hold the analysis worker.

**Detected parts** (:func:`find_kind_parts`, :func:`stamp_parts`, after the matte cut): the
parts people personalise (a shock spring, a grip, rims, tyres, a seat, a grille, mirrors) get a
region of their own that the grouping keeps as one group per *kind*, whatever their colour
(grouping is colour-only otherwise, so the Ducati's yellow spring shared a group with the gold
frame tubes). The object class comes from Florence-2's caption (:func:`object_class`), the
class's vocabulary of kinds (:data:`VOCAB`, each with 1-2 bare-noun prompts and a size range per
instance as a share of the object) is sent to an open-vocabulary detector (OWLv2,
:mod:`partdetect`), each box above the detector gate is prompted as a SAM box and the answer is
kept only when it passes the gates of :class:`PartGates` (SAM's score, mask/box agreement, on
the object, the kind's size range, no duplicate); wheel boxes are split into tyre and rim by
:func:`wheels.split_wheel`. Measured on the ten-photo part reference set: isolated
must-be-separate parts 6 -> 20 of 194 (per instance 29), merged 185 -> 165, no new junk, the
region metrics flat on our set and better on PACO. The detector threshold is the sensitive
knob: at 0.2 the wrongly named parts went from 5 to 33-61 of the stamped masks.

**Brake calipers** (:func:`find_calipers`, the wheel second look): at the scale of the photo no
detector scores a caliper high enough to use (OWLv2 0.18-0.22 on the three reference bikes, below
0.12 on two cars with painted calipers), so every wheel found is looked at again: OWLv2 on a
crop 2.5x a candidate's size names a bike's caliper at 0.26-0.38 (the fork foot the BMW's front
caliper is bolted to scored 0.25 and loses to the caliper beside it, one caliper per wheel; every
other candidate scored below 0.19), and a painted caliper, which OWLv2 does not
recognise at any scale, is the one region inside the wheel whose paint is found nowhere else in
it nor around it. Measured on the reference bikes and two added cars: 6 of the 7 visible
calipers found (the bicycle's, on a thin motion-blurred wheel nothing detects, is not), none on
the four cars without visible calipers (the final numbers are in docs/ARCHITECTURE.md). The same
zoomed pass gives the brake disc inside a wheel split into tyre and rim (:func:`find_discs`: the
rim is everything inside its lip, so the Ducati's drilled steel disc was part of its gold rim).

A part mask built from regions of the partition (``PartMask.exact``) adopts all of them when it
is stamped (the Corvette's caliper, seen on both sides of a spoke), a wheel's rim keeps only its
pixels on the matte (the backdrop seen between the spokes), and a mirror's stalk, which OWLv2
may name anything (a "rear spoiler" on the Alpine), is folded into the mirror
(:func:`attach_parts`).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .. import imageio
from .hierarchy import is_bimodal, two_means
from .labelops import adjacency, compact, region_areas, region_medians
from .wheels import split_wheel

log = logging.getLogger("recolor.segmentation.smallparts")

# Lettering.
TEXT_MIN_ALNUM = 2         # OCR strings with fewer ASCII letters or digits are texture, not text
TEXT_EXPAND = 4            # px the quad is grown by before it is prompted
TEXT_MARGIN = 24           # px of context around the quad in the SAM crop ...
TEXT_MIN_CROP = 128        # ... which is at least this big
TEXT_INSIDE = 0.98         # SAM's mask must lie inside the expanded quad by this share ...
TEXT_MAX_FILL = 0.9        # ... and cover at most this share of it
TEXT_DE = 15.0             # CIEDE2000 between the letters and the rest of the quad
TEXT_MIN_PX = 12
TEXT_MERGE_IOU = 0.3       # quads overlapping by more than this (or half the smaller) are one
TEXT_MAX_DOMAINS = 48      # at most this many merged lettering domains (the largest) are prompted
#: Named parts.
PART_MARGIN = 0.08         # crop = box grown by this share + 8 px
PART_MIN_PX = 150          # a phrase mask (or a wheel part) below this is dropped
PART_DEDUP_IOU = 0.85      # boxes overlapping this much are prompted once
PART_MAX_AREA = 0.6        # a box covering more than this share of the image is not a part
PART_MAX_BOXES = 48        # at most this many phrase boxes (the largest) are prompted
WHEEL_LABELS = ("tire", "wheel rim")
#: Prompts go to SAM in chunks of this many boxes; the deadline is checked between chunks.
PROMPT_CHUNK = 16
#: Sources of the stamped regions.
SOURCE_TEXT = "text"
SOURCE_WHEEL = "wheel"
SOURCE_NAMED = "named"

#: ``prompter(image_u8, jobs) -> candidates`` as :meth:`SamMasker.prompt_boxes` returns them.
Prompter = Callable[[np.ndarray, list[dict]], list[list[dict]]]


@dataclass
class Extra:
    """One extra mask for :func:`hierarchy.build_regions`: ``source`` is SOURCE_TEXT (stamped
    over everything but other text, exempt from the snap and the speck merge), SOURCE_WHEEL
    (``parts`` = (tyre, rim): stamped by the wheel rule) or SOURCE_NAMED (stamped where it
    splits a region). ``labels`` are the phrases (or the OCR strings) behind it."""
    mask: np.ndarray
    source: str
    labels: list[str] = field(default_factory=list)
    parts: Optional[tuple[np.ndarray, np.ndarray]] = None


# ---------------------------------------------------------------------- helpers

def _de(a, b) -> float:
    return float(imageio.delta_e(np.asarray(a, np.float32)[None], np.asarray(b, np.float32)[None])[0])


def _med(x: np.ndarray) -> np.ndarray:
    if len(x) > 20000:
        x = x[:: int(np.ceil(len(x) / 20000))]
    return np.median(x, axis=0).astype(np.float32)


def _full_mask(c: dict, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), bool)
    cm = np.asarray(c["mask"], bool)
    y0, x0 = int(c["y0"]), int(c["x0"])
    m[y0:y0 + cm.shape[0], x0:x0 + cm.shape[1]] = cm[: h - y0, : w - x0]
    return m


def _window(cx: float, cy: float, size_w: int, size_h: int, h: int, w: int) -> tuple[int, int, int, int]:
    """A size_w x size_h window centred on (cx, cy), shifted inside the image."""
    size_w, size_h = int(min(size_w, w)), int(min(size_h, h))
    x0 = int(np.clip(round(cx - size_w / 2), 0, w - size_w))
    y0 = int(np.clip(round(cy - size_h / 2), 0, h - size_h))
    return x0, y0, x0 + size_w, y0 + size_h


def _alnum(t: str) -> int:
    return sum(1 for ch in t if ch.isascii() and ch.isalnum())


def _quad_mask(quad: list[float], shape: tuple[int, int], expand: int) -> np.ndarray:
    h, w = shape
    pts = np.asarray(quad, np.float32).reshape(4, 2)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    if expand > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand + 1, 2 * expand + 1))
        m = cv2.dilate(m, k)
    return m.astype(bool)


def _expired(deadline: Optional[float]) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _prompt_chunks(prompter: Prompter, image_rgb_u8: np.ndarray, jobs: list[dict], deadline: Optional[float],
                   stats: Optional[dict] = None) -> list[list[dict]]:
    """``prompter`` on ``jobs`` in chunks of PROMPT_CHUNK, stopping at the deadline: jobs not
    prompted get no candidates. ``stats["prompted"]`` counts the jobs that were."""
    res: list[list[dict]] = [[] for _ in jobs]
    done = 0
    for i0 in range(0, len(jobs), PROMPT_CHUNK):
        if _expired(deadline):
            break
        chunk = jobs[i0:i0 + PROMPT_CHUNK]
        out = prompter(image_rgb_u8, chunk)
        for k, cands in enumerate(out):
            res[i0 + k] = cands
        done += len(chunk)
    if stats is not None:
        stats["prompted"] = stats.get("prompted", 0) + done
        stats["total"] = stats.get("total", 0) + len(jobs)
    return res


def _merge_quads(ocr: list[dict], shape: tuple[int, int]) -> list[tuple[np.ndarray, list[str]]]:
    """Overlapping OCR quads merged into lettering domains (the full image and the tiles
    report the same words). The test is on the quads' bounding boxes grown by TEXT_EXPAND
    (intersection over the smaller box > 0.5 or IoU > TEXT_MERGE_IOU), transitive through a
    union-find, so no pixel work is done per pair; only the TEXT_MAX_DOMAINS largest domains
    (by box area) are rasterised, each the union of its quads grown by TEXT_EXPAND px. A
    malformed, non-finite, zero-area (a point, a line) or out-of-image quad is dropped first.
    Returns ``[(mask, texts)]``, texts in reading order without repeats."""
    h, w = shape
    quads: list[np.ndarray] = []
    texts: list[str] = []
    for q in ocr:
        try:
            pts = np.asarray(q["quad"], np.float32).reshape(4, 2)
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(pts).all():
            continue
        if pts[:, 0].max() <= pts[:, 0].min() or pts[:, 1].max() <= pts[:, 1].min():
            continue                   # a point or a line: no lettering has zero area
        if pts[:, 0].max() <= 0 or pts[:, 0].min() >= w or pts[:, 1].max() <= 0 or pts[:, 1].min() >= h:
            continue                   # entirely outside the image
        quads.append(pts)
        texts.append(str(q.get("text", "")))
    n = len(quads)
    if n == 0:
        return []
    P = np.stack(quads)
    e = float(TEXT_EXPAND)
    x0 = np.clip(P[:, :, 0].min(1) - e, 0, w)
    x1 = np.clip(P[:, :, 0].max(1) + e, 0, w)
    y0 = np.clip(P[:, :, 1].min(1) - e, 0, h)
    y1 = np.clip(P[:, :, 1].max(1) + e, 0, h)
    area = (x1 - x0) * (y1 - y0)
    iw = np.clip(np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :]), 0, None)
    ih = np.clip(np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :]), 0, None)
    inter = iw * ih
    smaller = np.maximum(np.minimum(area[:, None], area[None, :]), 1e-6)
    union = np.maximum(area[:, None] + area[None, :] - inter, 1e-6)
    link = (inter > 0) & ((inter / smaller > 0.5) | (inter / union > TEXT_MERGE_IOU))
    # union-find over the link graph (transitive, as the pairwise restart loop was)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.nonzero(np.triu(link, 1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(a) for a in range(n)])
    comps: dict[int, list[int]] = {}
    for k, r in enumerate(roots.tolist()):
        comps.setdefault(r, []).append(k)
    # the largest domains first: the box of the union of the members
    order = sorted(comps.values(), key=lambda m: -float((x1[m].max() - x0[m].min()) * (y1[m].max() - y0[m].min())))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TEXT_EXPAND + 1, 2 * TEXT_EXPAND + 1)) if TEXT_EXPAND > 0 else None
    out: list[tuple[np.ndarray, list[str]]] = []
    for members in order[:TEXT_MAX_DOMAINS]:
        members = sorted(members)
        bx0, by0 = int(np.floor(x0[members].min())), int(np.floor(y0[members].min()))
        bx1, by1 = int(np.ceil(x1[members].max())) + 1, int(np.ceil(y1[members].max())) + 1
        bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(w, bx1), min(h, by1)
        if bx1 <= bx0 or by1 <= by0:
            continue
        crop = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        for k in members:
            cv2.fillPoly(crop, [np.round(P[k] - (bx0, by0)).astype(np.int32)], 1)
        if kernel is not None:
            crop = cv2.dilate(crop, kernel)
        if int(crop.sum()) < 16:
            continue
        m = np.zeros((h, w), bool)
        m[by0:by1, bx0:bx1] = crop.astype(bool)
        seen: set[str] = set()
        words = [t for t in (texts[k] for k in members) if not (t in seen or seen.add(t))]
        out.append((m, words))
    return out


# ---------------------------------------------------------------------- lettering

def text_masks(image_rgb_u8: np.ndarray, albedo_lab: np.ndarray, ocr: list[dict], prompter: Optional[Prompter],
               deadline: Optional[float] = None, stats: Optional[dict] = None) -> list[Extra]:
    """Lettering masks (SOURCE_TEXT) from Florence's OCR quads (``{"quad": [8 floats], "text"}``);
    see the module docstring. Without a ``prompter`` only the 2-means path runs. Past
    ``deadline`` (``time.monotonic`` seconds) no further domain is prompted (the ones already
    prompted are still examined: that is cheap CPU work, the prompts are the cost); ``stats``
    (optional dict) gets the prompted / total counts."""
    h, w = image_rgb_u8.shape[:2]
    ocr = [q for q in (ocr or []) if _alnum(str(q.get("text", ""))) >= TEXT_MIN_ALNUM]
    doms = _merge_quads(ocr, (h, w))
    if not doms:
        return []
    jobs = []
    for m, _ in doms:
        ys, xs = np.nonzero(m)
        bx = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        cw = max(TEXT_MIN_CROP, int(bx[2] - bx[0]) + 2 * TEXT_MARGIN)
        ch = max(TEXT_MIN_CROP, int(bx[3] - bx[1]) + 2 * TEXT_MARGIN)
        jobs.append({"crop": _window(0.5 * (bx[0] + bx[2]), 0.5 * (bx[1] + bx[3]), cw, ch, h, w), "box": bx})
    res: list[list[dict]] = [[] for _ in jobs]
    if prompter is not None:
        res = _prompt_chunks(prompter, image_rgb_u8, jobs, deadline, stats)
    elif stats is not None:
        stats["total"] = stats.get("total", 0) + len(jobs)
    out: list[Extra] = []
    for (dom, texts), cands in zip(doms, res):       # every prompted domain is examined (cheap CPU work)
        chosen = None
        for c in sorted(cands or [], key=lambda c: -float(c["score"])):
            m = _full_mask(c, (h, w))
            a = int(m.sum())
            if a < TEXT_MIN_PX or (m & dom).sum() < TEXT_INSIDE * a:
                continue
            rest = dom & ~m
            if rest.sum() < TEXT_MIN_PX or a > TEXT_MAX_FILL * dom.sum():
                continue
            if _de(_med(albedo_lab[m]), _med(albedo_lab[rest])) <= TEXT_DE:
                continue
            chosen = m
            break
        if chosen is None:
            pix = albedo_lab[dom]
            samp = pix[:: max(1, len(pix) // 4000)]
            if len(samp) < 32:
                continue
            c1, c2, _ = two_means(samp)
            d1 = ((pix - c1) ** 2).sum(1)
            d2 = ((pix - c2) ** 2).sum(1)
            second = d2 < d1
            if second.mean() > 0.5:
                second = ~second
                c1, c2 = c2, c1
            if _de(c1, c2) > TEXT_DE and second.sum() >= TEXT_MIN_PX and is_bimodal(samp, c1, c2):
                m = np.zeros((h, w), bool)
                m[dom] = second
                n, cc, stt, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
                small = np.flatnonzero(stt[:, cv2.CC_STAT_AREA] < 4)      # 1-3 px crumbs, not strokes
                small = small[small > 0]
                if len(small):
                    m &= ~np.isin(cc, small)
                if m.sum() >= TEXT_MIN_PX:
                    chosen = m
        if chosen is not None:
            out.append(Extra(chosen, SOURCE_TEXT, list(texts)))
    return out


# ---------------------------------------------------------------------- named parts

def _dedup_boxes(boxes: list[dict], h: int, w: int) -> list[dict]:
    uniq: list[dict] = []
    for b in boxes or []:
        try:
            x0, y0, x1, y1 = (float(v) for v in b["box"])
        except (KeyError, TypeError, ValueError):
            continue
        if (x1 - x0) < 4 or (y1 - y0) < 4 or (x1 - x0) * (y1 - y0) > PART_MAX_AREA * w * h:
            continue
        dup = False
        for u in uniq:
            ux0, uy0, ux1, uy1 = u["box"]
            iw, ih = max(0.0, min(x1, ux1) - max(x0, ux0)), max(0.0, min(y1, uy1) - max(y0, uy0))
            inter = iw * ih
            union = (x1 - x0) * (y1 - y0) + (ux1 - ux0) * (uy1 - uy0) - inter
            if inter / max(1e-6, union) > PART_DEDUP_IOU:
                u["labels"].append(str(b.get("label", "")))
                dup = True
                break
        if not dup:
            uniq.append({"box": [x0, y0, x1, y1], "labels": [str(b.get("label", ""))]})
    return uniq


def part_masks(image_rgb_u8: np.ndarray, albedo_lab: np.ndarray, boxes: list[dict], prompter: Prompter,
               deadline: Optional[float] = None, stats: Optional[dict] = None) -> list[Extra]:
    """Named-part masks from Florence's phrase boxes (``{"box": [x0, y0, x1, y1], "label"}``):
    wheels split into tyre and rim (SOURCE_WHEEL, ``parts``), other phrases as SOURCE_NAMED.
    See the module docstring. At most PART_MAX_BOXES boxes (the largest) are prompted, none
    past ``deadline``; ``stats`` gets the prompted / total counts."""
    h, w = image_rgb_u8.shape[:2]
    uniq = _dedup_boxes(boxes, h, w)
    if not uniq:
        return []
    uniq.sort(key=lambda u: -(u["box"][2] - u["box"][0]) * (u["box"][3] - u["box"][1]))
    uniq = uniq[:PART_MAX_BOXES]
    jobs = []
    for u in uniq:
        x0, y0, x1, y1 = u["box"]
        mx, my = PART_MARGIN * (x1 - x0) + 8, PART_MARGIN * (y1 - y0) + 8
        crop = (int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my)))
        jobs.append({"crop": crop, "box": u["box"]})
    res = _prompt_chunks(prompter, image_rgb_u8, jobs, deadline, stats)
    out: list[Extra] = []
    for u, cands in zip(uniq, res):
        ok = [c for c in (cands or []) if not c["clipped"]]
        if not ok:
            continue
        c = max(ok, key=lambda c: float(c["score"]))
        m = _full_mask(c, (h, w))
        if m.sum() < PART_MIN_PX:
            continue
        if any(lbl in WHEEL_LABELS for lbl in u["labels"]):
            sp = split_wheel(m, image_rgb_u8, albedo_lab)
            if sp is not None:
                out.append(Extra(m, SOURCE_WHEEL, list(u["labels"]), parts=(sp[0], sp[1])))
            continue                                  # a wheel that cannot be split adds nothing
        out.append(Extra(m, SOURCE_NAMED, list(u["labels"])))
    return out


def find_extras(image_rgb_u8: np.ndarray, albedo_lin: np.ndarray, analysis: Optional[dict],
                prompter: Prompter, budget_s: Optional[float] = None) -> list[Extra]:
    """Every extra mask for :func:`hierarchy.build_regions` from Florence's answers
    (:func:`florence.analyse`; None or empty gives []): lettering first, then wheels and named
    parts. ``albedo_lin`` is float32 linear HxWx3. With ``budget_s`` the work stops at that
    wall-clock time and the masks found so far are returned (the shortfall is logged): the
    regions stage runs this with the GPU lock held, so no photo may hold it for long."""
    if not analysis:
        return []
    t0 = time.monotonic()
    deadline = None if budget_s is None else t0 + float(budget_s)
    lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
    st_text: dict = {}
    st_part: dict = {}
    out = text_masks(image_rgb_u8, lab, analysis.get("ocr") or [], prompter, deadline=deadline, stats=st_text)
    out += part_masks(image_rgb_u8, lab, analysis.get("grounding") or [], prompter, deadline=deadline, stats=st_part)
    if deadline is not None and _expired(deadline):
        log.warning("lettering and named parts cut short after %.1f s: %d of %d lettering domains and %d of %d "
                    "part boxes prompted", time.monotonic() - t0, st_text.get("prompted", 0), st_text.get("total", 0),
                    st_part.get("prompted", 0), st_part.get("total", 0))
    return out


# ---------------------------------------------------------------------- detected parts: vocabulary

#: Region source of a detected part (:func:`stamp_parts`); its ``part_*`` fields name the kind.
SOURCE_KIND = "kind"


@dataclass(frozen=True)
class Kind:
    """One part kind of a class vocabulary: one group of the part-aware grouping ("Shock
    springs" holds every spring instance). ``phrases`` are the detector prompts (bare nouns);
    one instance's SAM mask must hold at least ``min_px`` px and ``min_frac`` of the object
    silhouette (the matte) and at most ``max_frac`` of it (a "spring" the size of a wheel is a
    false detection); at most ``max_instances`` are kept. ``tier`` is 'accessory' (the default
    vocabulary) or 'panel' (body panels: tank, fairing, hood; off, because they split the one
    paint a user repaints in one click and cost region isolation when measured). ``wheel``
    ('tyre' | 'rim') marks the kinds whose mask comes from a wheel split."""
    key: str
    label: str
    plural: str
    phrases: tuple[str, ...]
    max_frac: float
    min_px: int = 60
    min_frac: float = 0.0
    max_instances: int = 4
    tier: str = "accessory"
    wheel: str = ""


MOTORCYCLE = (
    Kind("shock_spring", "Shock spring", "Shock springs", ("coil spring", "shock absorber"), 0.02, max_instances=2),
    Kind("brake_caliper", "Brake caliper", "Brake calipers", ("brake caliper",), 0.012, max_instances=3),
    Kind("brake_disc", "Brake disc", "Brake discs", ("brake disc",), 0.05, max_instances=3),
    Kind("rim", "Wheel rim", "Wheel rims", ("wheel rim",), 0.12, min_frac=0.004, max_instances=2, wheel="rim"),
    Kind("tyre", "Tyre", "Tyres", ("tire",), 0.12, min_frac=0.004, max_instances=2, wheel="tyre"),
    Kind("mirror", "Mirror", "Mirrors", ("side mirror",), 0.012, max_instances=2),
    Kind("exhaust", "Exhaust", "Exhausts", ("exhaust muffler", "exhaust pipe"), 0.06, max_instances=3),
    Kind("seat", "Seat", "Seats", ("motorcycle seat",), 0.05, max_instances=2),
    Kind("grip", "Grip", "Grips", ("handlebar grip",), 0.006, max_instances=2),
    Kind("lever", "Lever", "Levers", ("brake lever",), 0.006, max_instances=2),
    Kind("sprocket", "Sprocket", "Sprockets", ("sprocket",), 0.04, max_instances=2),
    Kind("fork", "Fork", "Forks", ("front fork",), 0.05, max_instances=2),
    Kind("footpeg", "Footpeg", "Footpegs", ("footpeg",), 0.006, max_instances=4),
    Kind("fender", "Fender", "Fenders", ("front fender",), 0.05, max_instances=2, tier="panel"),
    Kind("tank", "Fuel tank", "Fuel tanks", ("fuel tank",), 0.1, max_instances=1, tier="panel"),
    Kind("fairing", "Fairing", "Fairings", ("fairing",), 0.2, max_instances=3, tier="panel"),
)
CAR = (
    Kind("rim", "Wheel rim", "Wheel rims", ("wheel rim",), 0.1, min_frac=0.003, max_instances=4, wheel="rim"),
    Kind("tyre", "Tyre", "Tyres", ("tire",), 0.1, min_frac=0.003, max_instances=4, wheel="tyre"),
    Kind("brake_caliper", "Brake caliper", "Brake calipers", ("brake caliper",), 0.01, max_instances=4),
    Kind("grille", "Grille", "Grilles", ("front grille",), 0.15, max_instances=2),
    Kind("mirror", "Mirror", "Mirrors", ("side mirror",), 0.02, max_instances=2),
    Kind("badge", "Badge", "Badges", ("badge", "emblem"), 0.01, max_instances=3),
    Kind("exhaust", "Exhaust tip", "Exhaust tips", ("exhaust pipe",), 0.01, max_instances=4),
    Kind("door_handle", "Door handle", "Door handles", ("door handle",), 0.004, max_instances=4),
    Kind("bumper", "Bumper", "Bumpers", ("bumper",), 0.2, max_instances=2),
    Kind("spoiler", "Spoiler", "Spoilers", ("rear spoiler",), 0.06, max_instances=1),
    Kind("fog_lamp", "Fog lamp", "Fog lamps", ("fog light",), 0.02, max_instances=2),
    Kind("vent", "Vent", "Vents", ("air vent",), 0.02, max_instances=4),
    Kind("fuel_cap", "Fuel cap", "Fuel caps", ("fuel cap",), 0.005, max_instances=1),
    Kind("hood", "Hood", "Hoods", ("car hood",), 0.4, max_instances=1, tier="panel"),
    Kind("roof", "Roof", "Roofs", ("car roof",), 0.3, max_instances=1, tier="panel"),
)
BICYCLE = (
    Kind("rim", "Wheel rim", "Wheel rims", ("wheel rim",), 0.3, min_frac=0.01, max_instances=2, wheel="rim"),
    Kind("tyre", "Tyre", "Tyres", ("tire",), 0.3, min_frac=0.01, max_instances=2, wheel="tyre"),
    Kind("brake_caliper", "Brake caliper", "Brake calipers", ("brake caliper",), 0.02, max_instances=2),
    Kind("brake_disc", "Brake disc", "Brake discs", ("brake disc",), 0.05, max_instances=2),
    Kind("saddle", "Saddle", "Saddles", ("bicycle saddle",), 0.05, max_instances=1),
    Kind("handlebar", "Handlebar", "Handlebars", ("handlebar",), 0.08, max_instances=1),
    Kind("crankset", "Crankset", "Cranksets", ("crankset",), 0.06, max_instances=1),
    Kind("pedal", "Pedal", "Pedals", ("pedal",), 0.01, max_instances=2),
    Kind("bottle", "Bottle", "Bottles", ("water bottle",), 0.03, max_instances=2),
    Kind("fork", "Fork", "Forks", ("bicycle fork",), 0.08, max_instances=1),
    Kind("frame", "Frame", "Frames", ("bicycle frame",), 0.4, max_instances=1, tier="panel"),
)
SNEAKER = (
    Kind("sole", "Sole", "Soles", ("shoe sole",), 0.35, min_frac=0.01, max_instances=2),
    Kind("laces", "Laces", "Laces", ("shoelaces",), 0.12, max_instances=2),
    Kind("logo", "Logo", "Logos", ("logo",), 0.02, max_instances=4),
    Kind("tongue", "Tongue", "Tongues", ("shoe tongue",), 0.08, max_instances=2),
    Kind("heel", "Heel counter", "Heel counters", ("heel counter",), 0.15, max_instances=2),
    Kind("toe", "Toe cap", "Toe caps", ("toe cap",), 0.1, max_instances=2),
)
FIGURE = (
    Kind("head", "Head", "Heads", ("robot head",), 0.08, max_instances=1),
    Kind("antenna", "Antenna", "Antennas", ("antenna",), 0.01, max_instances=3),
    Kind("shoulder", "Shoulder armour", "Shoulder armour", ("shoulder armor",), 0.1, max_instances=2),
    Kind("shield", "Shield", "Shields", ("shield",), 0.3, max_instances=1),
    Kind("weapon", "Weapon", "Weapons", ("rifle",), 0.2, max_instances=2),
    Kind("saber", "Beam saber", "Beam sabers", ("beam saber",), 0.02, max_instances=2),
    Kind("hand", "Hand", "Hands", ("robot hand",), 0.05, max_instances=2),
    Kind("foot", "Foot", "Feet", ("robot foot",), 0.08, max_instances=2),
    Kind("ear", "Ear", "Ears", ("robot ear",), 0.01, max_instances=2),
    Kind("leg", "Leg", "Legs", ("robot leg",), 0.2, max_instances=2, tier="panel"),
    Kind("arm", "Arm", "Arms", ("robot arm",), 0.15, max_instances=2, tier="panel"),
)
GENERIC = (
    Kind("logo", "Logo", "Logos", ("logo",), 0.02, max_instances=4),
    Kind("handle", "Handle", "Handles", ("handle",), 0.05, max_instances=2),
    Kind("button", "Button", "Buttons", ("button",), 0.01, max_instances=4),
    Kind("strap", "Strap", "Straps", ("strap",), 0.1, max_instances=2),
)
#: Part kinds per object class (the classes :func:`object_class` reads from a caption).
VOCAB: dict[str, tuple[Kind, ...]] = {
    "motorcycle": MOTORCYCLE, "car": CAR, "bicycle": BICYCLE, "sneaker": SNEAKER, "figure": FIGURE,
    "generic": GENERIC,
}
_CLASS_WORDS = (
    ("motorcycle", r"\b(motorcycle|motorbike|motor bike|scooter|superbike|dirt bike)s?\b"),
    ("bicycle", r"\b(bicycle|bike|cyclist|cycling|road bike)s?\b"),
    ("car", r"\b(car|van|truck|jeep|suv|vehicle|coupe|convertible|sedan|hatchback|roadster)s?\b"),
    ("sneaker", r"\b(shoe|sneaker|trainer|boot|footwear|clog|sandal)s?\b"),
    ("figure", r"\b(robot|gundam|figure|figurine|toy|mech|action figure|model kit|statue)s?\b"),
)


def object_class(caption: Optional[str]) -> str:
    """The vocabulary class of a caption: the class whose words appear first in it ("a man
    riding a motorcycle" is a motorcycle), 'generic' when none does (or without a caption)."""
    caption = (caption or "").lower()
    best, pos = "generic", len(caption) + 1
    for cls, pat in _CLASS_WORDS:
        m = re.search(pat, caption)
        if m and m.start() < pos:
            best, pos = cls, m.start()
    return best


def kinds_for(cls: str, tiers: tuple[str, ...] = ("accessory",)) -> tuple[Kind, ...]:
    """The kinds of object class ``cls`` (``generic`` for an unknown class) in ``tiers``."""
    return tuple(k for k in VOCAB.get(cls, GENERIC) if k.tier in tiers)


# ---------------------------------------------------------------------- detected parts: gates and masks

@dataclass(frozen=True)
class PartGates:
    """What a detection must pass to become a part region (chosen on the ten-photo part
    reference set; the detector score is the knob that matters). ``det_min`` is keyed by
    detector name so that a box from any other detector never passes; the product runs one
    detector, so the cross-detector votes of :func:`part_jobs` and :func:`_part_rank` stay at 1
    (they were measured with OWLv2 and Grounding DINO together and are kept for experiments)."""
    det_min: tuple[tuple[str, float], ...] = (("owlv2", 0.3),)   # per detector; unknown detectors never pass
    sam_min: float = 0.85           # SAM's predicted IoU of the kept candidate
    box_iou_min: float = 0.45       # the mask's bounding box against the detection box ...
    fill_min: float = 0.1           # ... and the share of the box the mask fills
    inside_min: float = 0.8         # share of the mask on the object (the matte)
    dedup_iou: float = 0.5          # two masks of one kind overlapping more (or one 80 % inside the other) are one
    cross_iou: float = 0.6          # two masks of different kinds overlapping more are one object
    box_nms_iou: float = 0.6        # detection boxes of one kind overlapping more are one detection ...
    box_inside: float = 0.8         # ... and so is a box this much inside a better one
    box_max_factor: float = 4.0     # a box larger than this x the kind's largest mask is not the kind
    max_jobs_per_kind: int = 12     # SAM prompts per kind (the best boxes)
    wheel_min_frac: float = 0.015   # a wheel mask below this share of the object is not a wheel
    wheel_fill_min: float = 0.3     # a wheel's SAM mask fills at least this share of its box (a tyre ring
                                    # alone fills about 0.4; the BMW's front wheel got a speckled 0.14 mask)

    def det_threshold(self, det: str) -> float:
        return dict(self.det_min).get(det, 1.0 + 1e-9)


DEFAULT_PART_GATES = PartGates()
#: SAM crop around a detection box: the box grown by this share of its size + 8 px.
PART_BOX_MARGIN = 0.08


@dataclass
class PartMask:
    """One accepted part instance: the kind, its SAM mask (bool HxW) and the evidence."""
    kind: str
    label: str
    plural: str
    mask: np.ndarray
    score: float                     # detector score (the best of the merged boxes)
    sam_score: float
    box: list[float]
    phrase: str
    det: str
    votes: int = 1
    notes: dict = field(default_factory=dict)
    #: The mask is a union of regions of the partition (the wheel look's region and painted
    #: routes): every region inside it is a piece of the part (:func:`stamp_parts` adopts them all).
    exact: bool = False

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def _box_iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def _box_inside(a, b) -> float:
    """Share of box ``a`` inside box ``b``."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


def part_jobs(dets: list[dict], kinds: Sequence[Kind], shape: tuple[int, int], object_px: int,
              gates: PartGates = DEFAULT_PART_GATES) -> list[dict]:
    """SAM box-prompt jobs from detections (``{"box": [x0, y0, x1, y1], "phrase", "score",
    "det"}``): each detection of a vocabulary phrase above its detector's gate, clipped to the
    image, not larger than ``box_max_factor`` x its kind's largest mask, then per kind (the
    wheel kinds pooled as 'wheel') a box NMS by score (IoU ``box_nms_iou``, or ``box_inside``
    containment; a duplicate from another detector counts as a vote), at most
    ``max_jobs_per_kind`` per kind. Each job carries its SAM ``crop``."""
    h, w = shape
    phrase_kind: dict[str, Kind] = {}
    for k in kinds:
        for p in k.phrases:
            phrase_kind.setdefault(p, k)
    by_kind: dict[str, list[dict]] = {}
    for d in dets or []:
        k = phrase_kind.get(str(d.get("phrase", "")))
        try:
            score = float(d["score"])
            x0, y0, x1, y1 = (float(v) for v in d["box"])
        except (KeyError, TypeError, ValueError):
            continue
        if k is None or not np.isfinite([score, x0, y0, x1, y1]).all() or score < gates.det_threshold(str(d.get("det"))):
            continue
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        key = "wheel" if k.wheel else k.key
        if (x1 - x0) * (y1 - y0) > gates.box_max_factor * max(k.max_frac * object_px, k.min_px):
            continue
        by_kind.setdefault(key, []).append({"kind": key, "box": [x0, y0, x1, y1], "score": score,
                                            "phrase": str(d["phrase"]), "det": str(d.get("det", "")), "votes": 1})
    jobs: list[dict] = []
    for key, lst in by_kind.items():
        lst.sort(key=lambda d: -d["score"])
        kept: list[dict] = []
        for d in lst:
            dup = next((u for u in kept if _box_iou(d["box"], u["box"]) > gates.box_nms_iou
                        or _box_inside(d["box"], u["box"]) >= gates.box_inside), None)
            if dup is not None:
                if d["det"] != dup["det"]:
                    dup["votes"] += 1
                continue
            kept.append(d)
        for d in kept[:gates.max_jobs_per_kind]:
            x0, y0, x1, y1 = d["box"]
            mx, my = PART_BOX_MARGIN * (x1 - x0) + 8, PART_BOX_MARGIN * (y1 - y0) + 8
            d["crop"] = (int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my)))
            jobs.append(d)
    return jobs


def _bbox_of(m: np.ndarray) -> Optional[list[float]]:
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _best_candidate(cc: Sequence[dict], box: Sequence[float], shape: tuple[int, int], gates: PartGates,
                    fill_min: Optional[float] = None) -> Optional[tuple[np.ndarray, float, float, float]]:
    """The unclipped SAM candidate of score >= ``sam_min`` whose bounding box agrees with
    ``box`` (IoU >= ``box_iou_min``, filling >= ``fill_min`` of it: the gates' unless given)
    with the best score x agreement: ``(mask, sam score, box IoU, fill)``, or None."""
    best = None
    barea = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
    fmin = gates.fill_min if fill_min is None else float(fill_min)
    for c in cc or []:
        if c.get("clipped") or float(c["score"]) < gates.sam_min:
            continue
        m = _full_mask(c, shape)
        bb = _bbox_of(m)
        if bb is None:
            continue
        biou = _box_iou(bb, box)
        fill = float(m.sum()) / barea
        if biou < gates.box_iou_min or fill < fmin:
            continue
        key = float(c["score"]) * biou
        if best is None or key > best[0]:
            best = (key, m, float(c["score"]), biou, fill)
    return None if best is None else best[1:]


def select_parts(image_rgb_u8: np.ndarray, albedo_lab: np.ndarray, fg: Optional[np.ndarray], jobs: list[dict],
                 cands: list[list[dict]], kinds: Sequence[Kind], gates: PartGates = DEFAULT_PART_GATES,
                 log_out: Optional[list] = None, wheels_out: Optional[list] = None) -> list[PartMask]:
    """The gated part masks of the SAM candidates ``cands`` (one list per job, as
    :meth:`SamMasker.prompt_boxes` returns them): per job the unclipped candidate of score >=
    ``sam_min`` whose bounding box agrees with the detection box (IoU >= ``box_iou_min``,
    filling >= ``fill_min`` of it) with the best score x agreement, kept when at least
    ``inside_min`` of it is on the object (``fg`` >= 0.5; everything without a matte); a wheel
    job's mask (at least ``wheel_min_frac`` of the object) is split into tyre and rim, and a
    wheel that does not split adds nothing; each piece must fit its kind's size range. Then the
    duplicates go (:func:`dedup_parts`). ``log_out`` (a list) gets one record per job;
    ``wheels_out`` (a list) gets ``{"box", "mask", "split"}`` for every wheel whose mask passed
    the gates, split or not (the wheel second look, :func:`find_calipers`, searches them)."""
    h, w = image_rgb_u8.shape[:2]
    obj = (np.asarray(fg, np.float32) >= 0.5) if fg is not None else np.ones((h, w), bool)
    object_px = max(int(obj.sum()), 1)
    kd = {k.key: k for k in kinds}
    wheel_kinds = {k.wheel: k for k in kinds if k.wheel}
    out: list[PartMask] = []

    def note(j, why, **kw):
        if log_out is not None:
            log_out.append({"kind": j["kind"], "box": [round(v) for v in j["box"]], "score": round(j["score"], 3),
                            "det": j["det"], "phrase": j["phrase"], "result": why, **kw})

    for j, cc in zip(jobs, cands):
        best = _best_candidate(cc, j["box"], (h, w), gates,
                               fill_min=max(gates.fill_min, gates.wheel_fill_min) if j["kind"] == "wheel" else None)
        if best is None:
            note(j, "no SAM candidate passes (clipped, score or box agreement)")
            continue
        m, sscore, biou, fill = best
        inside = float(obj[m].mean()) if m.any() else 0.0
        if inside < gates.inside_min:
            note(j, "off the object", inside=round(inside, 3))
            continue
        pieces: list[tuple[Kind, np.ndarray]] = []
        if j["kind"] == "wheel":
            if m.sum() < gates.wheel_min_frac * object_px:
                note(j, "wheel too small", area=int(m.sum()))
                continue
            sp = split_wheel(m, image_rgb_u8, albedo_lab)
            if sp is not None:
                # the rim is everything inside its lip, which the matte sees through between the
                # spokes: the backdrop there (the BMW's white wall, the dark behind the Ducati's
                # three spokes) is no part of the rim
                sp = (sp[0], sp[1] & obj)
            if wheels_out is not None:
                wheels_out.append({"box": list(j["box"]), "mask": m, "split": sp is not None, "score": float(j["score"]),
                                   "rim": sp[1].astype(bool) if sp is not None else None})
            if sp is None:
                note(j, "wheel not splittable")
                continue
            for part_name, pm in (("tyre", sp[0]), ("rim", sp[1])):
                if part_name in wheel_kinds and pm.any():
                    pieces.append((wheel_kinds[part_name], pm.astype(bool)))
        elif j["kind"] in kd:
            pieces.append((kd[j["kind"]], m))
        for k, pm in pieces:
            a = int(pm.sum())
            lo = max(k.min_px, k.min_frac * object_px)
            hi = k.max_frac * object_px
            if a < lo or a > hi:
                note(j, f"size {k.key}", area=a, lo=round(lo), hi=round(hi))
                continue
            out.append(PartMask(k.key, k.label, k.plural, pm, float(j["score"]), sscore, list(j["box"]), j["phrase"],
                                j["det"], votes=int(j.get("votes", 1)),
                                notes={"box_iou": round(biou, 3), "fill": round(fill, 3), "inside": round(inside, 3)}))
            note(j, f"accepted {k.key}", area=a, sam=round(sscore, 3), box_iou=round(biou, 3))
    return dedup_parts(out, kd, gates)


def _part_rank(p: PartMask) -> float:
    return p.score * (1.0 + 0.25 * (p.votes - 1)) * p.sam_score


def dedup_parts(parts: list[PartMask], kd: dict[str, Kind], gates: PartGates = DEFAULT_PART_GATES) -> list[PartMask]:
    """Best first (detector score x votes x SAM score): a mask overlapping a kept mask of its
    kind by IoU > ``dedup_iou`` (or 80 % of the smaller) is the same instance, one overlapping
    a kept mask of another kind by IoU > ``cross_iou`` the same object; at most
    ``max_instances`` per kind."""
    kept: list[PartMask] = []
    for p in sorted(parts, key=lambda p: -_part_rank(p)):
        drop = False
        for q in kept:
            inter = int((p.mask & q.mask).sum())
            if inter == 0:
                continue
            iou = inter / max(int((p.mask | q.mask).sum()), 1)
            if (q.kind == p.kind and (iou > gates.dedup_iou or inter >= 0.8 * min(p.area, q.area))) or \
                    (q.kind != p.kind and iou > gates.cross_iou):
                drop = True
                break
        if drop:
            continue
        k = kd.get(p.kind)
        if k is not None and sum(1 for q in kept if q.kind == p.kind) >= k.max_instances:
            continue
        kept.append(p)
    return kept


#: ``detector(image_u8, phrases) -> [{"box", "phrase", "score", "det"}] | None`` (:func:`partdetect.detect`).
Detector = Callable[[np.ndarray, list[str]], Optional[list[dict]]]
#: ``zoom(image_u8, crops, phrases) -> [{"box", "phrase", "score", "scores", "src": "crop<k>"}] | None``
#: (:func:`partdetect.detect_in`: one detector pass per window, no tiles).
Zoom = Callable[[np.ndarray, list, list[str]], Optional[list[dict]]]


# ---------------------------------------------------------------------- detected parts: the wheel second look

#: The part kind the wheel second look finds (a kind of every vehicle vocabulary).
CALIPER = "brake_caliper"
#: The phrase of a brake disc in the zoomed pass (the disc look, :func:`find_discs`).
DISC_PHRASE = "brake disc"
#: The phrases of the zoomed passes. OWLv2 scores every phrase on its own, so the others never
#: change the caliper's score; they decide whether the caliper is a box's best phrase (a box that
#: is more a disc, a spoke, a lug nut or a fork than a caliper is not called one).
WHEEL_LOOK_PHRASES = ("brake caliper", "brake disc", "tire", "wheel rim", "wheel hub", "wheel spoke",
                      "front fork", "swingarm", "exhaust pipe", "sprocket", "lug nut", "valve stem", "bolt")


@dataclass(frozen=True)
class WheelLook:
    """Knobs of the wheel second look (:func:`find_calipers`), measured on the three reference
    bikes and two cars with visible calipers. At the scale of the whole photo OWLv2 scores no
    caliper above 0.22 (the BMW's two 0.21-0.22, the Ducati's 0.18, the cars' below 0.12); on a
    square crop 2.5x the caliper's size it scores the three bike calipers 0.26-0.38, the fork
    foot the BMW's front caliper is bolted to 0.25 (one caliper per wheel: the caliper wins) and
    every other compact region inside a wheel of the set below 0.19. Painted calipers (the blue
    and yellow ones of the two cars) it does not recognise at any scale (0.05-0.10), but their
    colour is found nowhere else in the wheel."""
    crop_margin: float = 0.12          # the wheel pass: the wheel's box grown by this share
    box_min: float = 0.1               # its caliper boxes (the caliper the best phrase) at least this ...
    rad: tuple = (0.2, 0.92)           # ... centred at this normalised radius of the wheel's ellipse ...
    area: tuple = (0.002, 0.08)        # ... of this share of the wheel's area are candidates
    max_boxes: int = 3                 # per wheel
    region_min_px: int = 60            # a region of the partition with at least `inside` of it in the
    inside: float = 0.85               # wheel's outline, in the size and radius ranges above ...
    solidity: float = 0.6              # ... and compact (its area over its convex hull's) is a candidate
    painted_c: float = 20.0            # painted: CIELAB chroma at least this, its paint on at most
    painted_share: float = 0.05        # `painted_share` of the rest of the wheel and of a band around it
    band: float = 0.25                 # (this share of the wheel's mean semi-axis wide); one paint: the
    paint_cast: float = 15.0           # lightness-normalised (a, b) within `paint_cast`, a pixel's own
    paint_min_c: float = 10.0          # chroma at least `paint_min_c` (a neutral is no paint)
    verify_k: float = 2.5              # the verification crop: a square this many times the candidate's size
    verify_min: float = 0.22           # verified: a box agreeing with the candidate (IoU >= verify_iou) whose
    verify_iou: float = 0.4            # best phrase is the caliper, at this score or more
    max_verify: int = 6                # verification crops per wheel
    group_de: float = 12.0             # accepted pieces of one wheel within this dE are one caliper
    # the brake disc (:func:`find_discs`), on a wheel split into tyre and rim
    disc_min: float = 0.3              # a box of the wheel pass whose best phrase is the disc, at this score ...
    disc_rad: float = 0.3              # ... centred within this normalised radius of the wheel's ellipse ...
    disc_size: tuple = (0.4, 0.95)     # ... the larger side of its box this share of the wheel box's
    disc_in_rim: float = 0.85          # its SAM mask at least this much inside the rim ...
    disc_rim_max: float = 0.7          # ... at most this share of the rim's area ...
    disc_lip_band: float = 0.15        # ... and at most `disc_lip_max` of it in the rim's outer band
    disc_lip_max: float = 0.25         #     (this share of the rim's radius wide: the rim's own lip)
    disc_clean_px: int = 25            # holes and specks of the disc mask below this many px are closed / dropped


DEFAULT_WHEEL_LOOK = WheelLook()


@dataclass
class _Wheel:
    box: list
    disk: np.ndarray                   # bool HxW: the wheel's outline, filled
    area: int
    centre: tuple
    axes: tuple                        # semi-axes of the fitted ellipse
    angle: float                       # radians


def _wheel_geometry(mask: np.ndarray, box) -> Optional[_Wheel]:
    """The wheel's outline: the convex hull of its mask (a swingarm, an exhaust or a rope in
    front of the wheel cuts notches into the mask, and a caliper sits in one), with the ellipse
    fitted to it; the ellipse inscribed in the detection ``box`` when the mask covers less than
    half of that (SAM answered a wheel box with a sliver of the tyre)."""
    m = np.ascontiguousarray(mask, np.uint8)
    h, w = m.shape
    x0, y0, x1, y1 = (float(v) for v in box)
    box_ellipse = np.pi / 4.0 * max(x1 - x0, 1.0) * max(y1 - y0, 1.0)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    disk = np.zeros_like(m)
    geo = None
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        if len(hull) >= 5 and cv2.contourArea(hull) >= 0.5 * box_ellipse:
            (cx, cy), (ew, eh), ang = cv2.fitEllipse(hull)
            if np.isfinite([cx, cy, ew, eh]).all() and ew >= 8 and eh >= 8:
                cv2.fillPoly(disk, [hull], 1)
                geo = ((float(cx), float(cy)), (ew / 2.0, eh / 2.0), float(np.radians(ang)))
    if geo is None:
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        geo = (((x0 + x1) / 2.0, (y0 + y1) / 2.0), ((x1 - x0) / 2.0, (y1 - y0) / 2.0), 0.0)
        cv2.ellipse(disk, (int(round(geo[0][0])), int(round(geo[0][1]))),
                    (int(round(geo[1][0])), int(round(geo[1][1]))), 0.0, 0.0, 360.0, 1, -1)
    disk = disk[:h, :w].astype(bool)
    if not disk.any():
        return None
    return _Wheel(list(box), disk, int(disk.sum()), geo[0], geo[1], geo[2])


def _radius(wh: _Wheel, x: float, y: float) -> float:
    """Normalised radius of (x, y) in the wheel's ellipse (1 on its outline)."""
    dx, dy = x - wh.centre[0], y - wh.centre[1]
    c, s = float(np.cos(wh.angle)), float(np.sin(wh.angle))
    return float(np.hypot((dx * c + dy * s) / wh.axes[0], (-dx * s + dy * c) / wh.axes[1]))


def _norm_ab(lab: np.ndarray) -> np.ndarray:
    """Lightness-normalised (a, b) (:func:`grouping.normalise_lab` at L 50): what one paint keeps
    under different light."""
    lab = np.asarray(lab, np.float32).reshape(-1, 3)
    k = 66.0 / np.clip(lab[:, 0] + 16.0, 8.0, None)
    return lab[:, 1:] * k[:, None]


def _same_paint_share(pixels_lab: np.ndarray, col, lk: "WheelLook") -> float:
    """Share of ``pixels_lab`` (CIELAB [n, 3]) that are the paint ``col`` under some light: a
    chroma of at least ``paint_min_c`` and a lightness-normalised (a, b) within ``paint_cast``."""
    p = np.asarray(pixels_lab, np.float32).reshape(-1, 3)
    if len(p) == 0:
        return 0.0
    if len(p) > 30000:
        p = p[:: int(np.ceil(len(p) / 30000))]
    d = np.linalg.norm(_norm_ab(p) - _norm_ab(np.asarray(col, np.float32)), axis=1)
    chromatic = np.hypot(p[:, 1], p[:, 2]) >= lk.paint_min_c
    return float(((d <= lk.paint_cast) & chromatic).mean())


def _band_around(disk: np.ndarray, albedo_lab: np.ndarray, lk: "WheelLook") -> np.ndarray:
    """CIELAB pixels of the band just outside a wheel's outline (``band`` of its size wide)."""
    ys, xs = np.nonzero(disk)
    size = 0.5 * (float(xs.max() - xs.min()) + float(ys.max() - ys.min())) / 2.0
    r = max(3, int(round(lk.band * size)))
    h, w = disk.shape
    y0, y1 = max(0, int(ys.min()) - r), min(h, int(ys.max()) + r + 1)
    x0, x1 = max(0, int(xs.min()) - r), min(w, int(xs.max()) + r + 1)
    d = disk[y0:y1, x0:x1].astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ring = cv2.dilate(d, k).astype(bool) & ~d.astype(bool)
    return albedo_lab[y0:y1, x0:x1][ring]


def _square(box, k: float, shape: tuple[int, int], min_side: int = 48) -> tuple[int, int, int, int]:
    h, w = shape
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    side = int(round(max(min_side, k * max(box[2] - box[0], box[3] - box[1]))))
    return _window(cx, cy, side, side, h, w)


def _crop_of(d: dict) -> int:
    try:
        return int(str(d.get("src", ""))[4:])
    except ValueError:
        return -1


def _caliper_scores(zoom: Zoom, image: np.ndarray, boxes: list, lk: WheelLook) -> Optional[list[float]]:
    """Per box: the best score of a box of its verification crop that agrees with it (IoU >=
    ``verify_iou``) and whose best phrase is the caliper (0 when none); None without the model."""
    if not boxes:
        return []
    crops = [_square(b, lk.verify_k, image.shape[:2]) for b in boxes]
    dets = zoom(image, crops, list(WHEEL_LOOK_PHRASES))
    if dets is None:
        return None
    out = [0.0] * len(boxes)
    for d in dets:
        k = _crop_of(d)
        if not (0 <= k < len(boxes)) or d.get("phrase") != WHEEL_LOOK_PHRASES[0]:
            continue
        if _box_iou(d["box"], boxes[k]) >= lk.verify_iou:
            out[k] = max(out[k], float(d["score"]))
    return out


def find_calipers(image_rgb_u8: np.ndarray, albedo_lab: np.ndarray, fg: Optional[np.ndarray], labels: Optional[np.ndarray],
                  info: Optional[Sequence[dict]], wheels: Sequence[dict], zoom: Zoom, prompter: Prompter, kind: Kind,
                  object_px: int, gates: PartGates = DEFAULT_PART_GATES, look: WheelLook = DEFAULT_WHEEL_LOOK,
                  deadline: Optional[float] = None, log_out: Optional[list] = None,
                  pass_out: Optional[dict] = None) -> list[PartMask]:
    """The wheel second look: a brake caliper inside each wheel ``wheels`` (``{"box", "mask"}``,
    the SAM masks of the wheel boxes, split into tyre and rim or not). At the scale of the photo
    no detector scores a caliper high enough to use (:class:`WheelLook`), so each wheel is looked
    at again:

    * **candidates**: OWLv2 (``zoom``) on the wheel's crop, its boxes whose best phrase is the
      caliper (>= ``box_min``), and the regions of the partition ``labels`` (not lettering, not
      another part, on the matte) lying inside the wheel's outline; either must be centred off
      the hub and inside the rim (normalised radius ``rad``) and have a caliper's share of the
      wheel (``area``), a region must be compact (``solidity``);
    * **verified**: a candidate is the caliper when OWLv2, on a square crop ``verify_k`` times
      its size, names a box agreeing with it a caliper at ``verify_min`` or more (a rim-coloured
      caliper, the BMW's black one inside its black rim); a box candidate gets its mask from a
      SAM box prompt (``prompter``, the gates of :func:`select_parts`);
    * **painted**: a region of a clear colour (chroma >= ``painted_c``) whose paint is found
      nowhere else in the wheel nor in a band around it (at most ``painted_share`` of either)
      and that is no piece of a same-painted region going on outside the wheel is the caliper: a
      painted caliper is the one coloured thing inside a wheel, and OWLv2 does not recognise
      one at any scale.

    One caliper per wheel: the best piece (a verified one by score, then a painted one) and
    every accepted piece of the wheel of its paint (a caliper seen between two spokes). Each
    must fit ``kind``'s size range against ``object_px``. Returns the part masks (kind
    CALIPER); ``log_out`` gets one record per wheel; ``pass_out`` (a dict) gets the wheel pass
    (``geo``: the wheels' outlines by crop, ``dets``: the zoomed boxes) for :func:`find_discs`.
    Nothing is found without the model."""
    h, w = image_rgb_u8.shape[:2]
    out: list[PartMask] = []
    lk = look
    geo = [(i, _wheel_geometry(np.asarray(wd["mask"], bool), wd["box"])) for i, wd in enumerate(wheels)]
    geo = [(i, g) for i, g in geo if g is not None]
    if not geo or _expired(deadline):
        return out

    def note(i, why, **kw):
        if log_out is not None:
            log_out.append({"kind": CALIPER, "wheel": i, "result": why, **kw})

    # the wheel pass: one detector pass per wheel crop
    crops = []
    for _, g in geo:
        x0, y0, x1, y1 = g.box
        mx, my = lk.crop_margin * (x1 - x0), lk.crop_margin * (y1 - y0)
        crops.append((int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my))))
    dets = zoom(image_rgb_u8, crops, list(WHEEL_LOOK_PHRASES))
    if dets is None:
        note(-1, "detector unavailable")
        return out
    if pass_out is not None:
        pass_out.update(geo=geo, dets=dets)
    obj = (np.asarray(fg, np.float32) >= 0.5) if fg is not None else np.ones((h, w), bool)
    n_reg = int(labels.max()) + 1 if labels is not None else 0
    areas = region_areas(labels, n_reg) if labels is not None else None
    meds = region_medians(labels, albedo_lab, n_reg) if labels is not None else None
    ring_k = np.ones((5, 5), np.uint8)
    srcs = {int(d.get("id", k)): str(d.get("source", "")) for k, d in enumerate(info or [])}
    kinds_of = {int(d.get("id", k)) for k, d in enumerate(info or []) if d.get("part_kind")}
    lo = max(kind.min_px, kind.min_frac * object_px)
    hi = kind.max_frac * object_px
    for k, (wi, g) in enumerate(geo):
        if _expired(deadline):
            note(wi, "out of time")
            break
        # candidate boxes of the wheel pass
        boxes = []
        for d in sorted((d for d in dets if _crop_of(d) == k and d.get("phrase") == WHEEL_LOOK_PHRASES[0]
                         and float(d["score"]) >= lk.box_min),
                        key=lambda d: (-round(float(d["score"]), 3), (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))):
            b = [float(v) for v in d["box"]]
            a = (b[2] - b[0]) * (b[3] - b[1])
            rad = _radius(g, (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            if not (lk.rad[0] <= rad <= lk.rad[1] and lk.area[0] * g.area <= a <= lk.area[1] * g.area):
                continue
            if any(_box_iou(b, q["box"]) > 0.5 or _box_inside(q["box"], b) >= 0.8 or _box_inside(b, q["box"]) >= 0.8
                   for q in boxes):
                continue
            boxes.append({"box": b, "score": float(d["score"]), "route": "box"})
            if len(boxes) >= lk.max_boxes:
                break
        # candidate regions inside the wheel's outline
        regions = []
        if labels is not None:
            ys, xs = np.nonzero(g.disk)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            sub = labels[y0:y1, x0:x1]
            dsk = g.disk[y0:y1, x0:x1]
            ids, cnt = np.unique(sub[dsk], return_counts=True)
            in_disk = dict(zip(ids.tolist(), cnt.tolist()))
            rest_lab = albedo_lab[y0:y1, x0:x1][dsk]
            rest_ids = sub[dsk]
            around = None                                   # the band around the wheel, for the painted test
            for r, c in zip(ids.tolist(), cnt.tolist()):
                a = int(areas[r])
                if r in kinds_of or srcs.get(r) == SOURCE_TEXT or c < lk.inside * a:
                    continue
                if not (max(lk.region_min_px, lk.area[0] * g.area) <= a <= lk.area[1] * g.area):
                    continue
                rm = sub == r
                ry, rx = np.nonzero(rm)
                if float(obj[y0:y1, x0:x1][rm].mean()) < 0.5:
                    continue
                rad = _radius(g, float(rx.mean()) + x0, float(ry.mean()) + y0)
                if not (lk.rad[0] <= rad <= lk.rad[1]):
                    continue
                hull = cv2.convexHull(np.stack([rx, ry], 1).astype(np.int32))
                if a / max(float(cv2.contourArea(hull)), 1.0) < lk.solidity:
                    continue
                col = _med(albedo_lab[y0:y1, x0:x1][rm])
                same = around_share = 1.0
                painted = float(np.hypot(col[1], col[2])) >= lk.painted_c
                if painted:
                    # its paint (hue and lightness-normalised chroma) found nowhere else in the wheel ...
                    same = _same_paint_share(rest_lab[rest_ids != r], col, lk)
                    painted = same <= lk.painted_share
                if painted:
                    # ... nor around it (a fender's edge seen inside the wheel's outline) ...
                    if around is None:
                        around = _band_around(g.disk, albedo_lab, lk)
                    around_share = _same_paint_share(around, col, lk)
                    painted = around_share <= lk.painted_share
                if painted:
                    # ... and not a piece of something of its paint that goes on outside the wheel
                    qy0, qy1 = max(0, int(ry.min()) + y0 - 3), min(h, int(ry.max()) + y0 + 4)
                    qx0, qx1 = max(0, int(rx.min()) + x0 - 3), min(w, int(rx.max()) + x0 + 4)
                    loc = labels[qy0:qy1, qx0:qx1]
                    rl = loc == r
                    ring = cv2.dilate(rl.astype(np.uint8), ring_k).astype(bool) & ~rl
                    nb, nc = np.unique(loc[ring], return_counts=True)
                    for n, c2 in zip(nb.tolist(), nc.tolist()):
                        if c2 >= 8 and in_disk.get(n, 0) < 0.5 * areas[n] and \
                                _same_paint_share(np.asarray(meds[n], np.float32)[None], col, lk) > 0:
                            painted = False
                            break
                bb = [float(rx.min() + x0), float(ry.min() + y0), float(rx.max() + x0 + 1), float(ry.max() + y0 + 1)]
                regions.append({"box": bb, "region": int(r), "colour": col, "painted": painted, "px": a,
                                "same": round(same, 3), "around": round(around_share, 3),
                                "route": "painted" if painted else "region"})
        # a box that is a candidate region already is the region (exact pixels, no prompt)
        boxes = [b for b in boxes if not any(_box_iou(b["box"], r["box"]) >= 0.5 for r in regions)]
        # verification crops: the detector's boxes (best first), then the larger regions
        todo = boxes + sorted((r for r in regions if not r["painted"]), key=lambda r: (-r["px"], r["region"]))
        todo = todo[:lk.max_verify]
        scores = _caliper_scores(zoom, image_rgb_u8, [c["box"] for c in todo], lk) if todo else []
        if scores is None:
            note(wi, "detector unavailable")
            break
        for c, s in zip(todo, scores):
            c["verified"] = round(float(s), 3)
        # a caliper OWLv2 names beats a painted one (a fender's edge can pass the colour tests,
        # the BMW's real caliper is the one OWLv2 names)
        accepted = [dict(c, rank=c["verified"]) for c in todo if c["verified"] >= lk.verify_min]
        accepted += [dict(r, rank=lk.verify_min - 0.01) for r in regions if r["painted"]]
        if not accepted:
            note(wi, "no caliper", candidates=[{k: v for k, v in c.items() if k not in ("colour",)} for c in regions + boxes])
            continue
        # masks: a region's own pixels, a box's SAM answer
        jobs = [c for c in accepted if c["route"] == "box"]
        if jobs:
            pj = []
            for c in jobs:
                b = c["box"]
                mx, my = PART_BOX_MARGIN * (b[2] - b[0]) + 8, PART_BOX_MARGIN * (b[3] - b[1]) + 8
                pj.append({"crop": (int(max(0, b[0] - mx)), int(max(0, b[1] - my)), int(min(w, b[2] + mx)),
                                    int(min(h, b[3] + my))), "box": b})
            res = _prompt_chunks(prompter, image_rgb_u8, pj, deadline)
            for c, cc in zip(jobs, res):
                best = _best_candidate(cc, c["box"], (h, w), gates)
                if best is not None and float(g.disk[best[0]].mean()) >= 0.6:
                    c["mask"], c["sam"] = best[0], best[1]
                    c["colour"] = _med(albedo_lab[best[0]])
        for c in accepted:
            if c["route"] != "box":
                x0, y0, x1, y1 = (int(v) for v in c["box"])
                m = np.zeros((h, w), bool)
                m[y0:y1, x0:x1] = labels[y0:y1, x0:x1] == c["region"]
                c["mask"], c["sam"] = m, 1.0
        accepted = [c for c in accepted if "mask" in c]
        if not accepted:
            note(wi, "no SAM mask for the caliper box")
            continue
        accepted.sort(key=lambda c: -c["rank"])
        first = accepted[0]
        mask = first["mask"].copy()
        for c in accepted[1:]:
            if _de(c["colour"], first["colour"]) <= lk.group_de or \
                    _same_paint_share(np.asarray(c["colour"], np.float32)[None], first["colour"], lk) > 0:
                mask |= c["mask"]
        a = int(mask.sum())
        if a < lo or a > hi:
            note(wi, "size", area=a, lo=round(lo), hi=round(hi))
            continue
        bb = _bbox_of(mask)
        score = 1.0 if first["route"] == "painted" else float(first.get("verified", 0.0) or 0.0)
        out.append(PartMask(CALIPER, kind.label, kind.plural, mask, score, float(first["sam"]), bb, WHEEL_LOOK_PHRASES[0],
                            "wheel-colour" if first["route"] == "painted" else "owlv2-zoom",
                            notes={"route": first["route"], "verified": first.get("verified"), "pieces": len(accepted)},
                            exact=all(c["route"] != "box" for c in accepted)))
        note(wi, f"accepted {CALIPER}", route=first["route"], area=a, verified=first.get("verified"),
             candidates=[{k: v for k, v in c.items() if k not in ("colour", "mask")} for c in regions + boxes])
    return out


#: The part kind the disc look finds (motorcycle and bicycle vocabularies).
DISC = "brake_disc"

#: A part that touches a detected mirror (ATTACH_TOUCH_PX), is at most ATTACH_MAX of its size and
#: lies inside the mirror's box grown by ATTACH_GROW of its size on each side is the mirror's
#: stalk or base, and is folded into the mirror: OWLv2 called the Alpine's mirror stalk a "rear
#: spoiler", a 633 px part row in the body's colour beside the 1925 px mirror head (the reference
#: holds head and stalk as one mirror). The handlebar's own parts (a grip next to a bar-end
#: mirror) are never folded.
ATTACH_HOSTS = ("mirror",)
ATTACH_NEVER = ("grip", "lever", "handlebar")
ATTACH_MAX = 0.5
ATTACH_GROW = 1.0
ATTACH_TOUCH_PX = 3


def attach_parts(parts: list[PartMask], log_out: Optional[list] = None) -> list[PartMask]:
    """The parts with every mirror's stalk folded into its mirror (ATTACH_*: a smaller part
    touching the mirror inside its grown box); the mirror keeps its kind, score and box."""
    hosts = [p for p in parts if p.kind in ATTACH_HOSTS]
    if not hosts:
        return list(parts)
    k = np.ones((2 * ATTACH_TOUCH_PX + 1, 2 * ATTACH_TOUCH_PX + 1), np.uint8)
    out: list[PartMask] = []
    for p in parts:
        if p.kind in ATTACH_HOSTS or p.kind in ATTACH_NEVER:
            out.append(p)
            continue
        pb = _bbox_of(p.mask)
        home = None
        for q in hosts:
            qb = _bbox_of(q.mask)
            if pb is None or qb is None or p.area > ATTACH_MAX * q.area:
                continue
            gw, gh = ATTACH_GROW * (qb[2] - qb[0]), ATTACH_GROW * (qb[3] - qb[1])
            if not (pb[0] >= qb[0] - gw and pb[2] <= qb[2] + gw and pb[1] >= qb[1] - gh and pb[3] <= qb[3] + gh):
                continue
            y0, y1 = int(max(0, min(pb[1], qb[1]) - ATTACH_TOUCH_PX)), int(max(pb[3], qb[3]) + ATTACH_TOUCH_PX)
            x0, x1 = int(max(0, min(pb[0], qb[0]) - ATTACH_TOUCH_PX)), int(max(pb[2], qb[2]) + ATTACH_TOUCH_PX)
            grown = cv2.dilate(p.mask[y0:y1, x0:x1].astype(np.uint8), k).astype(bool)
            if (grown & q.mask[y0:y1, x0:x1]).any():
                home = q
                break
        if home is None:
            out.append(p)
            continue
        home.mask = home.mask | p.mask
        home.notes.setdefault("attached", []).append({"kind": p.kind, "px": p.area})
        if log_out is not None:
            log_out.append({"kind": p.kind, "result": f"attached to {home.kind}", "area": p.area,
                            "box": [round(v) for v in p.box]})
    return out


def _clean_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """``mask`` with its holes and islands below ``min_px`` px filled / dropped (worked on the
    mask's box: cheap)."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or min_px <= 0:
        return mask
    y0, y1 = max(0, int(ys.min()) - 1), min(mask.shape[0], int(ys.max()) + 2)
    x0, x1 = max(0, int(xs.min()) - 1), min(mask.shape[1], int(xs.max()) + 2)
    sub = mask[y0:y1, x0:x1].copy()
    for value in (False, True):                      # the holes, then the islands
        n, cc, st, _ = cv2.connectedComponentsWithStats((sub == value).astype(np.uint8), connectivity=8)
        small = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < min_px]
        if value is False:
            border = set(np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]])).tolist())
            small = [i for i in small if i not in border]      # the outside is no hole
        if small:
            sub[np.isin(cc, small)] = not value
    out = mask.copy()
    out[y0:y1, x0:x1] = sub
    return out


def find_discs(image_rgb_u8: np.ndarray, fg: Optional[np.ndarray], wheels: Sequence[dict], wheel_pass: dict,
               prompter: Prompter, kind: Kind, object_px: int, gates: PartGates = DEFAULT_PART_GATES,
               look: WheelLook = DEFAULT_WHEEL_LOOK, deadline: Optional[float] = None,
               log_out: Optional[list] = None) -> list[PartMask]:
    """The brake disc inside each wheel split into tyre and rim (``wheels[i]["rim"]``), from the
    wheel second look's zoomed pass (``wheel_pass`` from :func:`find_calipers`): the rim is
    everything inside the rim's lip, so the Ducati's drilled steel disc was part of its gold
    "Wheel rim" and a rim repaint painted the disc. A box of the pass whose best phrase is the
    disc (>= ``disc_min``), centred on the hub (``disc_rad``) and of a disc's size (``disc_size``),
    is prompted as a SAM box; of SAM's answers (the gates' score and box agreement, on the object)
    the disc is the one inside the rim (``disc_in_rim``), smaller than it (``disc_rim_max``) and
    off the rim's lip (``disc_lip_max`` of it in the outer ``disc_lip_band`` of the rim): SAM's
    other answers to a disc box are the whole rim. Stamped over the rim like every smaller part,
    kind DISC. At most one per wheel; nothing without a split wheel or the pass."""
    out: list[PartMask] = []
    geo, dets = wheel_pass.get("geo") or [], wheel_pass.get("dets") or []
    if not geo or not dets or _expired(deadline):
        return out
    lk = look
    h, w = image_rgb_u8.shape[:2]
    obj = (np.asarray(fg, np.float32) >= 0.5) if fg is not None else np.ones((h, w), bool)
    lo = max(kind.min_px, kind.min_frac * object_px)
    hi = kind.max_frac * object_px
    jobs, meta = [], []
    for k, (wi, g) in enumerate(geo):
        rim = wheels[wi].get("rim") if wi < len(wheels) else None
        if rim is None or not np.asarray(rim).any():
            continue
        wside = max(g.box[2] - g.box[0], g.box[3] - g.box[1], 1.0)
        best = None
        for d in dets:
            if _crop_of(d) != k or d.get("phrase") != DISC_PHRASE or float(d["score"]) < lk.disc_min:
                continue
            b = [float(v) for v in d["box"]]
            side = max(b[2] - b[0], b[3] - b[1]) / wside
            if not (lk.disc_size[0] <= side <= lk.disc_size[1]) or \
                    _radius(g, (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) > lk.disc_rad:
                continue
            if best is None or float(d["score"]) > best[0]:
                best = (float(d["score"]), b)
        if best is None:
            continue
        b = best[1]
        mx, my = PART_BOX_MARGIN * (b[2] - b[0]) + 8, PART_BOX_MARGIN * (b[3] - b[1]) + 8
        jobs.append({"crop": (int(max(0, b[0] - mx)), int(max(0, b[1] - my)), int(min(w, b[2] + mx)),
                              int(min(h, b[3] + my))), "box": b})
        meta.append((wi, np.asarray(rim, bool), best[0]))
    if not jobs:
        return out
    res = _prompt_chunks(prompter, image_rgb_u8, jobs, deadline)
    for (wi, rim, score), j, cc in zip(meta, jobs, res):
        rim_px = int(rim.sum())
        ys, xs = np.nonzero(rim)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        sub = rim[y0:y1, x0:x1].astype(np.uint8)
        hull = cv2.convexHull(np.stack([xs - x0, ys - y0], 1).astype(np.int32))
        disk = np.zeros_like(sub)
        cv2.fillPoly(disk, [hull], 1)
        dt = cv2.distanceTransform(disk, cv2.DIST_L2, 3)
        band = np.zeros((h, w), bool)
        band[y0:y1, x0:x1] = (dt <= lk.disc_lip_band * 0.5 * min(y1 - y0, x1 - x0)) & sub.astype(bool)
        pick = None
        for c in cc or []:
            if c.get("clipped") or float(c["score"]) < gates.sam_min:
                continue
            m = _full_mask(c, (h, w))
            a = int(m.sum())
            bb = _bbox_of(m)
            if bb is None or a == 0 or _box_iou(bb, j["box"]) < gates.box_iou_min or a < gates.fill_min * \
                    max((j["box"][2] - j["box"][0]) * (j["box"][3] - j["box"][1]), 1.0):
                continue
            inside = float((m & rim).sum()) / a
            lip = float((m & band).sum()) / a
            on_obj = float(obj[m].mean())
            if inside < lk.disc_in_rim or a > lk.disc_rim_max * rim_px or lip > lk.disc_lip_max or on_obj < gates.inside_min:
                continue
            key = (lip, -float(c["score"]))
            if pick is None or key < pick[0]:
                pick = (key, m, float(c["score"]), lip, inside)
        if log_out is not None and pick is None:
            log_out.append({"kind": DISC, "wheel": wi, "result": "no disc mask", "score": round(score, 3)})
        if pick is None:
            continue
        m = _clean_small(pick[1], lk.disc_clean_px)      # SAM draws a drilled disc speckled: holes of a few px
        a = int(m.sum())
        if a < lo or a > hi:
            if log_out is not None:
                log_out.append({"kind": DISC, "wheel": wi, "result": "size", "area": a, "lo": round(lo), "hi": round(hi)})
            continue
        out.append(PartMask(DISC, kind.label, kind.plural, m, score, pick[2], _bbox_of(m), DISC_PHRASE, "owlv2-zoom",
                            notes={"lip": round(pick[3], 3), "in_rim": round(pick[4], 3)}))
        if log_out is not None:
            log_out.append({"kind": DISC, "wheel": wi, "result": f"accepted {DISC}", "area": a, "score": round(score, 3),
                            "sam": round(pick[2], 3), "lip": round(pick[3], 3)})
    return out


def find_kind_parts(image_rgb_u8: np.ndarray, albedo_lin: np.ndarray, fg: Optional[np.ndarray], caption: Optional[str],
                    detector: Detector, prompter: Prompter, budget_s: Optional[float] = None,
                    gates: PartGates = DEFAULT_PART_GATES, tiers: tuple[str, ...] = ("accessory",),
                    labels: Optional[np.ndarray] = None, info: Optional[Sequence[dict]] = None,
                    zoom: Optional[Zoom] = None, look: Optional[WheelLook] = DEFAULT_WHEEL_LOOK
                    ) -> tuple[list[PartMask], dict]:
    """The part instances of one work image (module docstring): the class of ``caption``, its
    vocabulary's phrases to ``detector`` (every tier is asked, the boxes of ``tiers`` are
    kept), SAM box prompts (``prompter``, in chunks, none past ``budget_s`` of wall clock) and
    the gates. With ``zoom`` (:func:`partdetect.detect_in`) and a vocabulary with a brake
    caliper, every wheel found is looked at again for its caliper (:func:`find_calipers`, on
    the partition ``labels`` / ``info`` when given; ``look`` None turns it off). Returns
    ``(parts, report)``; ``report`` has the class, the counts and the gate log. A detector that
    returns None (model unavailable) gives no parts."""
    t0 = time.monotonic()
    deadline = None if budget_s is None else t0 + float(budget_s)
    cls = object_class(caption)
    kinds = kinds_for(cls, tiers)
    report: dict = {"class": cls, "caption": caption or "", "detections": 0, "jobs": 0, "parts": 0, "gate_log": []}
    if not kinds:
        return [], report
    # Every phrase of the class is asked (the panel tier too) and a box is kept for the phrase it
    # scores best on: a fuel tank's box answers "fuel tank", not its runner-up "seat". The panel
    # phrases are there as distractors only: their boxes are never kept.
    asked = sorted({p for k in VOCAB.get(cls, GENERIC) for p in k.phrases})
    use = {p for k in kinds for p in k.phrases}
    dets = detector(image_rgb_u8, asked)
    if dets is None:
        report["detector"] = "unavailable"
        return [], report
    dets = [d for d in dets if d.get("phrase") in use]
    report["detections"] = len(dets)
    h, w = image_rgb_u8.shape[:2]
    object_px = int((np.asarray(fg, np.float32) >= 0.5).sum()) if fg is not None else h * w
    jobs = part_jobs(dets, kinds, (h, w), max(object_px, 1), gates)
    report["jobs"] = len(jobs)
    if not jobs:
        return [], report
    stats: dict = {}
    cands = _prompt_chunks(prompter, image_rgb_u8, [{"crop": j["crop"], "box": j["box"]} for j in jobs], deadline,
                           stats)
    if stats.get("prompted", 0) < len(jobs):
        log.warning("detected parts cut short after %.1f s: %d of %d part boxes prompted", time.monotonic() - t0,
                    stats.get("prompted", 0), len(jobs))
    lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
    wheels: list[dict] = []
    parts = select_parts(image_rgb_u8, lab, fg, jobs, cands, kinds, gates, log_out=report["gate_log"],
                         wheels_out=wheels)
    kd = {k.key: k for k in kinds}
    if zoom is not None and look is not None and CALIPER in kd and wheels:
        t1 = time.monotonic()
        wheel_pass: dict = {}
        cal = find_calipers(image_rgb_u8, lab, fg, labels, info, wheels, zoom, prompter, kd[CALIPER],
                            max(object_px, 1), gates, look, deadline, log_out=report["gate_log"], pass_out=wheel_pass)
        report["calipers"] = len(cal)
        if DISC in kd:
            discs = find_discs(image_rgb_u8, fg, wheels, wheel_pass, prompter, kd[DISC], max(object_px, 1), gates, look,
                               deadline, log_out=report["gate_log"])
            report["discs"] = len(discs)
            cal = cal + discs
        report["wheel_look_s"] = round(time.monotonic() - t1, 3)
        if cal:
            parts = dedup_parts(parts + cal, kd, gates)
    parts = attach_parts(parts, log_out=report["gate_log"])
    report["parts"] = len(parts)
    report["seconds"] = round(time.monotonic() - t0, 3)
    return parts, report


# ---------------------------------------------------------------------- detected parts: stamping

#: A stamped mask (or what is left of it) below this many px adds nothing.
STAMP_MIN_PX = 40
#: An existing region whose IoU with a part mask is at least this is the part: it is tagged
#: whole instead of being re-drawn (the partition, its snapped edges and its remnants stay; a
#: spring's or a mirror's region is often a better outline than SAM's box answer). A tyre or a
#: rim adopts only a region lying ADOPT_INSIDE_WHEEL inside its mask, and is cut to the mask
#: otherwise: the regions stage's own wheel split drew the BMW's front tyre region over the fork
#: stanchion in front of it (11.5k px beyond the tyre's mask), and adopted whole it made "Tyres"
#: paint the fork.
ADOPT_IOU = 0.6
ADOPT_INSIDE_WHEEL = 0.9
WHEEL_PART_KINDS = ("tyre", "rim")
#: A host region more than this x the mask is always cut by it.
HOST_FACTOR = 2.0
#: A region less than this share of which lies inside the mask is only nicked: kept whole.
NICK_SHARE = 0.5
#: Distinct sub-parts kept inside a stamped mask: a region at least SUB_INSIDE inside it, at
#: most SUB_MAX of its area, at least SUB_MIN_PX and at least SUB_DE (CIEDE2000, albedo medians)
#: from the rest of the mask (a caliper on a rim, a decal on an exhaust).
SUB_INSIDE = 0.85
SUB_MAX = 0.35
SUB_MIN_PX = 60
SUB_DE = 15.0
#: Remnants: a region that lost at least REMNANT_LOST of its pixels to parts and keeps fewer
#: than REMNANT_MAX_PX (or less than REMNANT_REL of what it lost) joins the touching part it
#: lost them to when its median albedo is within REMNANT_DE (the 2-4 px ring SAM's automatic
#: mask leaves around a box-prompted one is the part's own material; a group of it is junk). A
#: wheel part takes a remnant of REMNANT_MAX_PX or more back only when it is a ring (no disk of
#: REMNANT_CORE_PX radius fits in it): a larger piece with a core stood in front of the wheel
#: (the fork stanchion in the BMW's tyre region).
REMNANT_LOST = 0.5
REMNANT_MAX_PX = 400
REMNANT_REL = 0.35
REMNANT_DE = 15.0
REMNANT_CORE_PX = 3
#: The other pieces of an adopted part whose mask is a union of regions (``PartMask.exact``: the
#: wheel look's region and painted routes): a region lying at least SUB_INSIDE inside the mask, of
#: at least SUB_MIN_PX, within COADOPT_DE of the adopted region's median, joins it (the Corvette's
#: caliper is seen above and below a spoke, two regions its mask held; adopting the larger alone
#: left the lower piece in a "Gold" colour group, a yellow blob when the caliper was painted red).
#: A SAM mask is not co-adopted: a wheel's mask holds the fork leg in front of it.
COADOPT_DE = 10.0


def _has_core(mask: np.ndarray, radius: int) -> bool:
    """True when a disk of ``radius`` px fits inside ``mask`` somewhere (worked on its box)."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return False
    sub = np.pad(mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8), radius)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return bool(cv2.erode(sub, k).any())


def stamp_parts(labels: np.ndarray, info: list[dict], parts: Sequence[PartMask], albedo_lab: np.ndarray
                ) -> tuple[np.ndarray, list[dict], dict]:
    """The regions stage's partition with one region per part instance (after the matte cut,
    so a part never meets the fill, the speck merge or the snap of ``build_regions`` and
    survives at any size). Larger parts first, smaller over them (the more specific part wins
    where two overlap). A region with IoU >= ADOPT_IOU with the mask is adopted whole (tagged,
    pixels unchanged); otherwise the mask is stamped pixel by pixel except over lettering
    (source 'text'), other part regions, regions it only nicks (NICK_SHARE) and distinct
    sub-parts inside it (SUB_*); a host region much larger than the mask is always cut. The
    remnants of the regions it cut join the part when they are its colour (REMNANT_*).
    A part region has source SOURCE_KIND and ``part_kind``, ``part_label``, ``part_plural``,
    ``part_instance`` (0.. per kind, larger instances first), ``part_score`` in its info.
    Returns ``(labels, info, report)``: ids contiguous 0..N-1, ``info[i]["id"] == i``, every
    region's ``area`` and ``albedo_lab`` refreshed, the other info fields kept."""
    labels = np.ascontiguousarray(labels, np.int32).copy()
    info = [dict(d, id=i) for i, d in enumerate(info)]
    n0 = len(info)
    before = region_areas(labels, n0)
    report: dict = {"stamped": [], "adopted": [], "kept_subparts": [], "skipped": [], "remnants_merged": 0}
    counter: dict[str, int] = {}
    for p in sorted(parts, key=lambda p: -p.area):
        is_text = np.array([d.get("source") == SOURCE_TEXT for d in info], bool)
        m = p.mask & ~is_text[labels]
        ma = int(m.sum())
        if ma < STAMP_MIN_PX:
            report["skipped"].append({"kind": p.kind, "px": ma})
            continue
        ids, cnt = np.unique(labels[m], return_counts=True)
        areas_now = region_areas(labels, len(info))
        inst = counter.get(p.kind, 0)
        rec = {"source": SOURCE_KIND, "part_kind": p.kind, "part_label": p.label, "part_plural": p.plural,
               "part_instance": inst, "part_score": round(float(p.score), 4), "confidence": round(float(p.sam_score), 4),
               "phrase": p.phrase, "det": p.det}
        adopt = None
        inside_min = ADOPT_INSIDE_WHEEL if p.kind in WHEEL_PART_KINDS else 0.0
        for r, c in zip(ids.tolist(), cnt.tolist()):
            if c / max(areas_now[r] + ma - c, 1) >= ADOPT_IOU and c >= inside_min * areas_now[r] \
                    and info[r].get("source") not in (SOURCE_KIND, SOURCE_TEXT):
                adopt = r
                break
        if adopt is not None:
            counter[p.kind] = inst + 1
            info[adopt] = dict(info[adopt], **rec, adopted_from=info[adopt].get("source", "sam"))
            px = int(areas_now[adopt])
            meds = None
            for r, c in zip(ids.tolist() if p.exact else [], cnt.tolist() if p.exact else []):
                if r == adopt or info[r].get("source") in (SOURCE_KIND, SOURCE_TEXT) or c < SUB_MIN_PX \
                        or c < SUB_INSIDE * int(areas_now[r]):
                    continue
                if meds is None:
                    meds = region_medians(labels, albedo_lab, len(info))
                if _de(meds[r], meds[adopt]) <= COADOPT_DE:
                    labels[labels == r] = adopt                # another piece of the part
                    px += int(areas_now[r])
                    report.setdefault("coadopted", []).append({"kind": p.kind, "region": int(r), "into": int(adopt),
                                                               "px": int(areas_now[r])})
            report["adopted"].append({"kind": p.kind, "region": int(adopt), "px": px})
            report["stamped"].append({"kind": p.kind, "instance": inst, "px": px, "adopted": True,
                                      "score": round(float(p.score), 3), "sam": round(float(p.sam_score), 3)})
            continue
        keep = np.zeros(len(info), bool)
        meds = None
        for r, c in zip(ids.tolist(), cnt.tolist()):
            if info[r].get("source") == SOURCE_KIND:
                continue
            ar = int(areas_now[r])
            if ar > HOST_FACTOR * ma:
                continue                                   # a host the part sits in: cut
            if c < NICK_SHARE * ar:
                keep[r] = True                             # a neighbour across the mask's edge
                report["kept_subparts"].append({"kind": p.kind, "region": int(r), "px": int(c), "why": "nicked"})
                continue
            if c < SUB_MIN_PX or c < SUB_INSIDE * ar or ar > SUB_MAX * ma:
                continue
            rest = m & (labels != r)
            if rest.sum() < STAMP_MIN_PX:
                continue
            if meds is None:
                meds = region_medians(labels, albedo_lab, len(info))
            rest_med = np.median(albedo_lab[rest][:: max(1, int(rest.sum()) // 20000)], axis=0)
            de = _de(meds[r], rest_med)
            if de >= SUB_DE:
                keep[r] = True
                report["kept_subparts"].append({"kind": p.kind, "region": int(r), "px": int(c), "de": round(de, 1),
                                                "why": "sub-part"})
        m2 = m & ~keep[labels]
        if int(m2.sum()) < STAMP_MIN_PX:
            report["skipped"].append({"kind": p.kind, "px": int(m2.sum())})
            continue
        counter[p.kind] = inst + 1
        rid = len(info)
        labels[m2] = rid
        info.append(dict(rec, id=rid))
        report["stamped"].append({"kind": p.kind, "instance": inst, "px": int(m2.sum()), "score": round(float(p.score), 3),
                                  "sam": round(float(p.sam_score), 3)})
    n = len(info)
    after = region_areas(labels, n)
    rem = [i for i in range(n0) if after[i] > 0 and before[i] - after[i] >= REMNANT_LOST * before[i]
           and (after[i] < REMNANT_MAX_PX or after[i] < REMNANT_REL * (before[i] - after[i]))
           and info[i].get("source") not in (SOURCE_TEXT, SOURCE_KIND)]
    if rem:
        meds = region_medians(labels, albedo_lab, n)
        pairs, cnt = adjacency(labels, n)
        nb: dict[int, dict[int, int]] = {}
        for (a, b), c in zip(pairs.tolist(), cnt.tolist()):
            nb.setdefault(a, {})[b] = c
            nb.setdefault(b, {})[a] = c
        target = np.arange(n)
        cored: dict[int, bool] = {}
        for i in rem:
            for j in sorted((j for j in nb.get(i, {}) if info[j].get("source") == SOURCE_KIND), key=lambda j: -nb[i][j]):
                if info[j].get("part_kind") in WHEEL_PART_KINDS and after[i] >= REMNANT_MAX_PX:
                    # a wheel part takes only a ring back: a piece with a core is what stood in
                    # front of the wheel (the fork stanchion in the BMW's tyre region)
                    if i not in cored:
                        cored[i] = _has_core(labels == i, REMNANT_CORE_PX)
                    if cored[i]:
                        continue
                de = _de(meds[i], meds[j])
                if np.isfinite(de) and de < REMNANT_DE:
                    target[i] = j
                    break
        moved = target != np.arange(n)
        if moved.any():
            labels = target[labels].astype(np.int32)
            report["remnants_merged"] = int(moved.sum())
    labels, mapping = compact(labels, n)
    labels = labels.astype(np.int32)
    info = [dict(info[old], id=int(new)) for old, new in enumerate(mapping.tolist()) if new >= 0]
    n = len(info)
    areas = region_areas(labels, n)
    meds = region_medians(labels, albedo_lab, n)
    for d in info:
        d["area"] = int(areas[d["id"]])
        d["albedo_lab"] = tuple(float(v) for v in meds[d["id"]])
    return labels, info, report


def part_info(info: Sequence[dict]) -> dict[int, dict]:
    """region id -> ``{"kind", "label", "plural", "instance"}`` of the part regions of a
    regions-stage info list (source SOURCE_KIND with a ``part_kind``)."""
    out = {}
    for k, d in enumerate(info or []):
        if d.get("part_kind"):
            out[int(d.get("id", k))] = {"kind": str(d["part_kind"]), "label": str(d.get("part_label", "")),
                                        "plural": str(d.get("part_plural", "")),
                                        "instance": int(d.get("part_instance", 0))}
    return out
