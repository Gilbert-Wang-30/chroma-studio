"""Analysis-time refinement of the colour groups: what the engine needs to know about a
photo beyond "which pixels share a paint".

Called by the pipeline's groups stage after :func:`grouping.group_regions`, in this order
(each step keeps a complete int32 partition, region ids contiguous, no -1):

1. :func:`absorb_washed` - a region whose albedo is the paint washed out by a blown
   highlight (same hue, lower chroma, higher lightness, chroma falling as lightness rises)
   joins the paint's group. SAM parts under a studio light cluster apart otherwise (the
   BMW's under-tail panel). (The lit and shadowed pieces of one paint were already merged
   at group level by :func:`grouping.absorb_lit`, inside the clustering call.)
   :func:`isolate_parts` then takes the parts the regions stage cut out of chromatic
   regions with SAM (source PART_SOURCE) out of the paint when clustering put them there:
   another material in a colour close to the paint's (a gold fork tube on a yellow bike).
   :func:`materials.absorb_highlights` moves a highlight of a touching paint that clustered
   as white (its unclipped remainder carries the paint's hue) into that paint, and
   :func:`materials.tag_regions` records every region's shininess and chrome advisory.
2. :func:`split_decals` - bright neutral blobs inside the main paint (white lettering)
   become their own regions, returned as the ``islands`` mask. Painted with the paint, a
   white DUCATI logo turned grey-blue. A bright neutral blob that continues into a
   neutral neighbour of the same colour is that neighbour's material (the BMW's clear
   tail-light lens against the white backdrop, inside the tail's SAM mask): it is carved
   too, even when it is only a little lighter than the paint. Lettering and small distinct
   parts the regions stage stamped (source 'text' / 'small') are islands too: the snap must
   not claim them and the paint's ramp must not enter them. Then :func:`fill_decal_gaps`
   gives the paint back the small gaps of its own colour inside a decal's region (the
   yellow between the letters of the BMW's "RR S1000", which SAM's mask of the decal held
   and a navy repaint left yellow). Then :func:`fill_letter_counters` gives the paint back the
   counters of the lettering printed on it (the holes of an "8": the paint mixed with the
   letter's edge, lighter and more orange, which a navy repaint of the paint left red).
3. the boundary snap (:mod:`recolor.segmentation.matting`), then :func:`relabel` to rebuild
   the records around the moved labels.
4. :func:`lock_materials` - a group in the main paint's hue that is much darker and less
   chromatic is another material (an anodised gold caliper, a fork cap, a mirror stalk
   tinted by the fairing's bounce) and is locked by default. The albedo carries no shading,
   so the same paint in shadow keeps the paint's chroma and is not locked. Background flags
   come from the regions' backdrop decisions (:func:`grouping._mark_background`), so they
   survive every rebuild here.
5. :func:`protect_mask` - a region outside the paint whose photo pixels are mostly the old
   paint's colour is an object of that colour (the BMW's brake-fluid reservoir), not a
   reflection: the engine's reflection stage never recolours it.
6. the junk pruning (:func:`junk.prune_junk`, with ``prune``): shadow slivers, the shadowed
   rims of detected parts and invisible colour variants join the group they belong to.
7. :func:`absorb_sheen` - a group that is the paint under a white sheen (the Torana's roof and
   boot lid facing the showroom's ceiling) joins that paint. Last, so the snap and the pruning
   see the groups the clustering made.

"The paint" is the main unlocked chromatic group (:func:`main_paint`) and its family,
every unlocked chromatic group within PAINT_HUE_TOL degrees of its hue
(:func:`paint_family`).

A regroup reproduces the analysis: :func:`refine_groups` records, for every final region,
the region of the regions stage it descends from (``Refined.origin``), and
:func:`regroup_refined` clusters those pre-snap regions again, with the same absorb rule,
before it carries the result over to the final label map. Clustering the snapped regions
instead split the BMW's tan undertray off its paint.

Every function is deterministic, CPU numpy (plus the GPU region medians of labelops), and
never touches a model.
"""
from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .. import imageio
from ..types import ColorGroup, Region
from . import grouping, materials
from .labelops import adjacency, bboxes, border_counts, compact, region_areas, region_medians

# The paint: the largest non-background, unlocked group above PAINT_CHROMA, and every unlocked
# group above it within PAINT_HUE_TOL degrees (CIELAB hue of the group albedo).
PAINT_CHROMA = 18.0
PAINT_HUE_TOL = 25.0

# Absorb (step 1). A region R outside a clearly chromatic group G (chroma >= ABSORB_GROUP_CHROMA,
# not background, not locked), sharing >= ABSORB_MIN_TOUCH boundary px with it, joins G when its
# hue is within ABSORB_HUE_TOL, its chroma is ABSORB_C_LO..ABSORB_C_HI of G's (and >= ABSORB_C_MIN),
# it is >= ABSORB_L_MARGIN lighter, not larger than G, and corr(L, C) over its pixels <= -ABSORB_RHO.
# Chosen on six analysed jobs: 2 correct moves, 0 wrong (the chroma-ratio bound does the work).
ABSORB_HUE_TOL = 12.0
ABSORB_C_LO, ABSORB_C_HI, ABSORB_C_MIN = 0.5, 0.95, 10.0
ABSORB_L_MARGIN = 3.0
ABSORB_RHO = 0.2
ABSORB_GROUP_CHROMA = 30.0
ABSORB_MIN_TOUCH = 20
ABSORB_MAX_AREA_RATIO = 1.0

# Decals (step 2), inside the regions of the main paint (chroma > DECAL_PAINT_CHROMA) of at least
# DECAL_MIN_REGION px: pixels > DECAL_DE (CIEDE2000) from the region median and > DECAL_DC lower
# in chroma, a 2x2 opening, 8-connected blobs >= DECAL_MIN_PX whose median is a bright neutral
# (L >= paint L + DECAL_L_MIN, chroma <= DECAL_C_MAX); specular or shadow blobs are not.
DECAL_PAINT_CHROMA = 30.0
DECAL_MIN_REGION = 500
DECAL_DE, DECAL_DC = 25.0, 30.0
DECAL_MIN_PX = 200
DECAL_L_MIN, DECAL_C_MAX = 10.0, 12.0
# ... or a bright neutral blob of at least SIL_MIN_PX, at least SIL_L_MIN lighter than the
# paint region, sharing >= SIL_TOUCH px of its outer ring with a neutral group (chroma <=
# DECAL_C_MAX) whose albedo is within SIL_DE (CIEDE2000) of the blob's: a clear or white
# part at the paint's edge, continuing into its neighbour (the BMW's tail-light lens).
SIL_MIN_PX = 40
SIL_L_MIN = 3.0
SIL_TOUCH = 10
SIL_DE = 6.0
#: A new region joins the group whose albedo is closer than this (CIEDE2000), else its own.
NEAREST_GROUP_DE = 10.0
# Decal gaps (step 2b): in a region outside the paint of at most GAP_MAX_REGION px that shares
# >= GAP_MIN_TOUCH boundary px with a main-paint region and whose median is > GAP_OWN_DE from
# the paint's, 8-connected blobs of at most GAP_MAX_PX pixels within GAP_PAINT_DE of that paint
# region's median and > GAP_OWN_DE from the region's own median go back to the paint region.
GAP_MAX_REGION = 4000
GAP_MIN_TOUCH = 20
GAP_PAINT_DE = 10.0
GAP_OWN_DE = 20.0
GAP_MAX_PX = 300
#: Regions the regions stage stamped as lettering or small distinct parts: islands by construction.
ISLAND_SOURCES = ("text", "small")

# Material lock (step 4): a paint-family group below MAT_C_RATIO of the main paint's chroma
# and more than MAT_L_MARGIN darker.
MAT_C_RATIO = 0.6
MAT_L_MARGIN = 10.0

#: Region source of a part the regions stage cut out of a chromatic region with SAM
#: (hierarchy.PART_SOURCE): not the paint, even when its colour is close to it.
PART_SOURCE = "part"

# Protected segments (step 5): photo pixels with CIELAB chroma > PROTECT_CHROMA within
# PROTECT_HUE degrees of the main paint's hue, at least PROTECT_FRAC of a region's pixels
# outside the paint and the islands, and at least PROTECT_MIN_PX of them.
PROTECT_CHROMA = 20.0
PROTECT_HUE = 30.0
PROTECT_FRAC = 0.4
PROTECT_MIN_PX = 100

GroupState = tuple[list[Region], list[ColorGroup], np.ndarray]


# ---------------------------------------------------------------------- small helpers

def _chroma(lab) -> float:
    return float(math.hypot(float(lab[1]), float(lab[2])))


def _hue(lab) -> float:
    return float((math.degrees(math.atan2(float(lab[2]), float(lab[1]))) + 360.0) % 360.0)


def _hue_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _rebuild(regions: list[Region], labels: np.ndarray, assignment: dict[int, int],
             carry: Optional[dict[int, dict]] = None) -> GroupState:
    """Groups from a region -> provisional group assignment, exactly as
    :func:`grouping.group_regions` builds them (ids 0..G-1 by area, background marked from
    the regions' backdrop decisions, or by the border rule without any)."""
    regions, groups, group_map = grouping._finalize(regions, labels, assignment, carry)
    grouping._mark_background(groups, group_map, regions)
    grouping._check_state(regions, groups)
    return regions, groups, group_map


def _check_partition(labels: np.ndarray, regions: Sequence[Region]) -> None:
    if labels.dtype != np.int32 or (labels < 0).any():
        raise AssertionError("labels must be an int32 partition with no -1")
    if int(labels.max()) + 1 != len(regions) or [r.id for r in regions] != list(range(len(regions))):
        raise AssertionError("region ids must be contiguous 0..N-1 and match the label map")


def refresh_regions(labels: np.ndarray, albedo_lab: np.ndarray, regions: Sequence[Region],
                    parents: Optional[dict[int, int]] = None) -> list[Region]:
    """Every Region record rebuilt from the label map (area, bbox, median albedo, border
    flag), keeping each region's group, source and confidence; a new id (in ``parents``)
    inherits its parent's record with source 'split'. Empty ids are dropped."""
    n = int(labels.max()) + 1
    meds = region_medians(labels, albedo_lab, n)
    areas = region_areas(labels, n)
    bb = bboxes(labels, n)
    border = border_counts(labels, n) > 0
    by_id = {r.id: r for r in regions}
    parents = parents or {}
    out: list[Region] = []
    for rid in range(n):
        if areas[rid] == 0:
            continue
        base = by_id.get(rid)
        if base is None:
            base = replace(by_id[parents[rid]], id=rid, source="split")
        lab = tuple(float(v) for v in meds[rid])
        out.append(replace(base, id=rid, area=int(areas[rid]), bbox=tuple(int(v) for v in bb[rid]),
                           albedo_lab=lab, albedo_hex=imageio.lab_to_hex(lab), touches_border=bool(border[rid])))
    return out


def _nearest_group(lab, groups: Sequence[ColorGroup]) -> Optional[int]:
    """Id of the colour group whose albedo is within NEAREST_GROUP_DE of ``lab``, else None
    (a part group never takes in a region by colour)."""
    groups = [g for g in groups if not g.part]
    if not groups:
        return None
    cents = np.array([g.albedo_lab for g in groups], np.float64)
    de = imageio.delta_e(np.repeat(np.asarray(lab, np.float64)[None], len(cents), 0), cents)
    j = int(np.argmin(de))
    return int(groups[j].id) if de[j] < NEAREST_GROUP_DE else None


# ---------------------------------------------------------------------- the paint

def main_paint(groups: Sequence[ColorGroup], chroma_min: float = PAINT_CHROMA) -> Optional[ColorGroup]:
    """The largest non-background, unlocked colour group with CIELAB chroma above
    ``chroma_min``, or None. A locked group is never the paint: after a merge that dulled the
    paint's median, the locked gold material became "the paint" and the protect mask covered
    the old paint's own colour. A detected part (a part group) is never the paint either."""
    cand = [g for g in groups if not g.is_background and not g.locked and not g.part
            and _chroma(g.albedo_lab) > chroma_min]
    return max(cand, key=lambda g: g.area) if cand else None


def paint_family(groups: Sequence[ColorGroup]) -> list[int]:
    """Ids of every non-background, unlocked colour group above PAINT_CHROMA within
    PAINT_HUE_TOL degrees of the main paint's hue (the main paint included), [] when there is
    no unlocked chromatic group. Part groups are never in it, and neither is a minor group
    whose parent is a part group (:func:`grouping.annotate_groups`: the part under other light,
    which goes with the part: painted with the paint, the robot's red feet got a navy rim)."""
    main = main_paint(groups)
    if main is None:
        return []
    h = _hue(main.albedo_lab)
    parts = {int(g.id) for g in groups if g.part}
    return [int(g.id) for g in groups if not g.is_background and not g.locked and not g.part
            and not (g.minor and g.parent in parts)
            and _chroma(g.albedo_lab) > PAINT_CHROMA and _hue_delta(_hue(g.albedo_lab), h) < PAINT_HUE_TOL]


# ---------------------------------------------------------------------- step 1: absorb

def _corr_l_c(albedo_lab: np.ndarray, labels: np.ndarray, r: Region) -> float:
    """corr(L, C) over a region's pixels (0 when it is too small or flat to tell)."""
    x0, y0, x1, y1 = r.bbox
    m = labels[y0:y1, x0:x1] == r.id
    p = albedo_lab[y0:y1, x0:x1][m]
    L = p[:, 0]
    C = np.hypot(p[:, 1], p[:, 2])
    if len(L) < 32 or L.std() < 1e-3 or C.std() < 1e-3:
        return 0.0
    return float(np.corrcoef(L, C)[0, 1])


def absorb_washed(regions: list[Region], groups: list[ColorGroup], labels: np.ndarray,
                  albedo_lab: np.ndarray) -> tuple[list[Region], list[ColorGroup], np.ndarray, list[dict]]:
    """Step 1: absorb regions that are the paint washed out by a highlight into it.

    Returns the rebuilt ``(regions, groups, group_map)`` (unchanged grouping, rebuilt the
    same way, when nothing moves) and a log of the moves. Locked, background, part and
    weakly chromatic groups never absorb, a detected part's region never moves; ties go to
    the closest hue."""
    pairs, cnt = adjacency(labels, len(regions))
    touch: dict[tuple[int, int], int] = {}
    for (a, b), c in zip(pairs.tolist(), cnt.tolist()):
        touch[(a, b)] = touch[(b, a)] = c
    g_of = {rid: g.id for g in groups for rid in g.region_ids}
    assignment = {r.id: g_of[r.id] for r in regions}
    cands = [g for g in groups if not g.is_background and not g.locked and not g.part
             and _chroma(g.albedo_lab) >= ABSORB_GROUP_CHROMA]
    moves: list[dict] = []
    for r in regions:
        if r.part_kind:
            continue
        best: Optional[dict] = None
        for g in cands:
            if g_of[r.id] == g.id:
                continue
            shared = sum(touch.get((r.id, o), 0) for o in g.region_ids)
            if shared < ABSORB_MIN_TOUCH:
                continue
            cg, cr = _chroma(g.albedo_lab), _chroma(r.albedo_lab)
            dh = _hue_delta(_hue(r.albedo_lab), _hue(g.albedo_lab))
            if not (dh <= ABSORB_HUE_TOL and ABSORB_C_LO * cg <= cr <= ABSORB_C_HI * cg and cr >= ABSORB_C_MIN
                    and r.albedo_lab[0] >= g.albedo_lab[0] + ABSORB_L_MARGIN
                    and r.area <= ABSORB_MAX_AREA_RATIO * g.area):
                continue
            corr = _corr_l_c(albedo_lab, labels, r)
            if corr <= -ABSORB_RHO and (best is None or dh < best["dh"]):
                best = {"region": r.id, "from_group": g_of[r.id], "to_group": g.id, "dh": round(dh, 1),
                        "c_ratio": round(cr / max(cg, 1e-6), 3), "corr": round(corr, 3), "area": r.area}
        if best is not None:
            assignment[r.id] = best["to_group"]
            moves.append(best)
    regions, groups, group_map = _rebuild(regions, labels, assignment)
    return regions, groups, group_map, moves


# ---------------------------------------------------------------------- step 1b: recovered parts

def isolate_parts(regions: list[Region], groups: list[ColorGroup], labels: np.ndarray,
                  parts: Sequence[int]) -> tuple[list[Region], list[ColorGroup], np.ndarray, list[int]]:
    """Step 1b: the regions stage cut ``parts`` (region ids) out of chromatic regions with SAM
    because their colour was not the region's: a gold fork tube between the yellow BMW's
    fairing and fender, a red sticker on the Ducati's rear shock, the tan console of a red
    car. When clustering puts one of them into the paint (its family is 25 deg wide, less the
    groups :func:`lock_materials` will lock) it is another material in the paint's colour: it
    leaves the paint for a group of its own (parts within NEAREST_GROUP_DE of each other
    share one), which is locked. Returns the rebuilt ``(regions, groups, group_map)`` and the
    ids of the regions that moved (their groups must stay locked through later rebuilds:
    :func:`lock_parts`)."""
    # the paint's family, less the groups step 4 will lock anyway (a part there stays with them)
    family = set(paint_family(groups)) - set(_material_ids(groups))
    part_ids = {int(p) for p in parts}
    moving = [r for r in regions if r.id in part_ids and r.group_id in family]
    if not moving:
        return regions, groups, grouping._group_map(regions, labels, len(groups)), []
    assignment = {r.id: r.group_id for r in regions}
    fresh: list[tuple[int, np.ndarray]] = []           # (provisional id, colour of its first part)
    for r in sorted(moving, key=lambda r: -r.area):
        lab = np.asarray(r.albedo_lab, np.float64)
        gid = None
        if fresh:
            de = imageio.delta_e(np.repeat(lab[None], len(fresh), 0), np.array([c for _, c in fresh], np.float64))
            j = int(np.argmin(de))
            if de[j] < NEAREST_GROUP_DE:
                gid = fresh[j][0]
        if gid is None:
            gid = len(groups) + len(fresh)
            fresh.append((gid, lab))
        assignment[r.id] = gid
    moved = sorted(r.id for r in moving)
    carry = {g.id: {"name": grouping._keep_name(g), "locked": g.locked, "is_background": g.is_background} for g in groups}
    regions, groups, group_map = grouping._finalize(regions, labels, assignment, carry)
    grouping._check_state(regions, groups)
    lock_parts(groups, moved)
    return regions, groups, group_map, moved


def lock_parts(groups: Sequence[ColorGroup], part_regions: Sequence[int],
               regions: Optional[Sequence[Region]] = None) -> list[int]:
    """Lock (in place) every group made mostly of ``part_regions`` (more than half of its
    area, or every region when ``regions`` is not given): the groups :func:`isolate_parts`
    gave the recovered parts, also when a decal or a piece a pixel-level split cut off
    joined one. Returns their ids."""
    parts = {int(p) for p in part_regions}
    if not parts:
        return []
    area = {r.id: max(int(r.area), 1) for r in regions} if regions is not None else None
    out = []
    for g in groups:
        if not g.region_ids:
            continue
        if area is None:
            ok = set(g.region_ids) <= parts
        else:
            tot = sum(area.get(rid, 1) for rid in g.region_ids)
            ok = 2 * sum(area.get(rid, 1) for rid in g.region_ids if rid in parts) > tot
        if ok:
            g.locked = True
            out.append(int(g.id))
    return out


# ---------------------------------------------------------------------- step 2: decals

def _continues_into_neutral(labels: np.ndarray, blob: np.ndarray, offset: tuple[int, int], rid: int,
                            group_of: np.ndarray, neutral: dict[int, np.ndarray], blob_lab) -> bool:
    """True when the outer ring of ``blob`` (a bool crop of region ``rid`` at ``offset``
    (x0, y0) in ``labels``) holds at least SIL_TOUCH pixels of one neutral group whose
    albedo is within SIL_DE of the blob's median ``blob_lab``."""
    if not neutral:
        return False
    x0, y0 = offset
    ys, xs = np.nonzero(blob)
    H, W = labels.shape
    cy0, cy1 = max(0, int(ys.min()) + y0 - 1), min(H, int(ys.max()) + y0 + 2)
    cx0, cx1 = max(0, int(xs.min()) + x0 - 1), min(W, int(xs.max()) + x0 + 2)
    sub = np.zeros((cy1 - cy0, cx1 - cx0), np.uint8)
    sub[ys + y0 - cy0, xs + x0 - cx0] = 1
    ring = cv2.dilate(sub, np.ones((3, 3), np.uint8)).astype(bool)
    lab_sub = labels[cy0:cy1, cx0:cx1]
    ids = lab_sub[ring & (lab_sub != rid)]
    ids = ids[ids < len(group_of)]                         # decals carved earlier in this pass
    if ids.size < SIL_TOUCH:
        return False
    gids, cnt = np.unique(group_of[ids], return_counts=True)
    ref = np.asarray(blob_lab, np.float64)[None]
    for g, c in zip(gids.tolist(), cnt.tolist()):
        if c >= SIL_TOUCH and g in neutral and float(imageio.delta_e(ref, neutral[g][None])[0]) < SIL_DE:
            return True
    return False


def split_decals(labels: np.ndarray, albedo_lab: np.ndarray, regions: list[Region], groups: list[ColorGroup]
                 ) -> tuple[np.ndarray, list[Region], list[ColorGroup], np.ndarray, np.ndarray]:
    """Step 2: carve bright neutral blobs (decals) out of the main paint's regions.

    A blob is carved when its median albedo is a bright neutral (chroma <= DECAL_C_MAX,
    at least DECAL_L_MIN lighter than its region, >= DECAL_MIN_PX), or when it is a neutral
    at least SIL_L_MIN lighter that continues into a neutral neighbour group of the same
    colour (see SIL_*): a clear or white part at the paint's edge, like a tail-light lens
    in front of a white backdrop. Every accepted blob of one paint region becomes one new
    region (appended ids, source 'split'), assigned to the nearest existing group or a
    group of its own. The letters' anti-aliased rim stays with the paint (no dilation).
    Returns ``(labels, regions, groups, group_map, islands)`` with ``islands`` the bool mask
    of the new regions."""
    labels = labels.copy()
    paint = main_paint(groups, DECAL_PAINT_CHROMA)
    islands = np.zeros(labels.shape, bool)
    if paint is None:
        return (labels,) + _rebuild(regions, labels, {r.id: r.group_id for r in regions}) + (islands,)
    by_id = {r.id: r for r in regions}
    next_rid = int(labels.max()) + 1
    group_of = np.zeros(next_rid, np.int64)
    for r in regions:
        group_of[r.id] = r.group_id
    neutral = {int(g.id): np.array(g.albedo_lab, np.float64) for g in groups if _chroma(g.albedo_lab) <= DECAL_C_MAX}
    parents: dict[int, int] = {}
    kernel = np.ones((2, 2), np.uint8)
    for rid in paint.region_ids:
        r = by_id[rid]
        x0, y0, x1, y1 = r.bbox
        view = labels[y0:y1, x0:x1]
        m = view == rid
        if m.sum() < DECAL_MIN_REGION:
            continue
        p = albedo_lab[y0:y1, x0:x1]
        med = np.array(r.albedo_lab, np.float32)
        flat = p.reshape(-1, 3)
        de = imageio.delta_e(flat, np.broadcast_to(med, flat.shape)).reshape(p.shape[:2])
        dc = np.hypot(med[1], med[2]) - np.hypot(p[..., 1], p[..., 2])
        out = m & (de > DECAL_DE) & (dc > DECAL_DC)
        # 2x2 opening (drops 1-px specks). Erode with the default anchor and dilate with the
        # reflected one: cv2.MORPH_OPEN with an even kernel moves the result one pixel
        # down-right, which put a band of paint into every letter's island.
        out = cv2.dilate(cv2.erode(out.astype(np.uint8), kernel), kernel, anchor=(0, 0)).astype(bool)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(out)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min(DECAL_MIN_PX, SIL_MIN_PX):
                continue
            blob = cc == i
            bm = np.median(p[blob], axis=0)
            if float(np.hypot(bm[1], bm[2])) > DECAL_C_MAX:
                continue
            if area >= DECAL_MIN_PX and bm[0] >= med[0] + DECAL_L_MIN:
                keep |= blob                                     # a white decal
            elif bm[0] >= med[0] + SIL_L_MIN and _continues_into_neutral(labels, blob, (x0, y0), rid, group_of,
                                                                          neutral, bm):
                keep |= blob                                     # a clear or white part at the edge
        if not keep.any():
            continue
        view[keep] = next_rid
        parents[next_rid] = rid
        next_rid += 1
    assignment = {r.id: r.group_id for r in regions}
    regions = refresh_regions(labels, albedo_lab, regions, parents)
    by_new = {r.id: r for r in regions}
    fresh = len(groups)
    for rid in parents:
        gid = _nearest_group(by_new[rid].albedo_lab, groups)
        if gid is None:
            gid, fresh = fresh, fresh + 1
        assignment[rid] = gid
    if parents:
        islands = np.isin(labels, list(parents))
    return (labels,) + _rebuild(regions, labels, assignment) + (islands,)


# ---------------------------------------------------------------------- step 2b: decal gaps

def fill_decal_gaps(labels: np.ndarray, albedo_lab: np.ndarray, regions: list[Region], groups: list[ColorGroup]
                    ) -> tuple[np.ndarray, list[Region], list[ColorGroup], np.ndarray, int]:
    """Step 2b: small gaps of the paint's own colour inside a decal's region go back to the
    paint (see GAP_*). Region ids do not change; the records are refreshed. Returns
    ``(labels, regions, groups, group_map, moved_px)``."""
    paint = main_paint(groups, DECAL_PAINT_CHROMA)
    if paint is None:
        return (labels,) + _rebuild(regions, labels, {r.id: r.group_id for r in regions}) + (0,)
    family = set(paint_family(groups))
    by_id = {r.id: r for r in regions}
    pairs, cnt = adjacency(labels, len(regions))
    touch: dict[int, dict[int, int]] = {}
    for (a, b), c in zip(pairs.tolist(), cnt.tolist()):
        touch.setdefault(a, {})[b] = c
        touch.setdefault(b, {})[a] = c
    out = labels.copy()
    moved = 0
    for r in regions:
        if r.group_id in family or r.area > GAP_MAX_REGION or by_id[r.id].backdrop:
            continue
        hosts = {o: c for o, c in touch.get(r.id, {}).items() if by_id[o].group_id == paint.id and c >= GAP_MIN_TOUCH}
        if not hosts:
            continue
        host = max(hosts, key=hosts.get)
        h_med = np.array(by_id[host].albedo_lab, np.float32)
        r_med = np.array(r.albedo_lab, np.float32)
        if _de(h_med, r_med) <= GAP_OWN_DE:
            continue
        x0, y0, x1, y1 = r.bbox
        view = labels[y0:y1, x0:x1]
        m = view == r.id
        p = albedo_lab[y0:y1, x0:x1][m]
        de_h = imageio.delta_e(p, np.broadcast_to(h_med, p.shape))
        de_r = imageio.delta_e(p, np.broadcast_to(r_med, p.shape))
        gap = np.zeros(m.shape, bool)
        gap[m] = (de_h < GAP_PAINT_DE) & (de_r > GAP_OWN_DE)
        if not gap.any():
            continue
        n, cc, stats, _ = cv2.connectedComponentsWithStats(gap.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(gap)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) <= GAP_MAX_PX:
                keep |= cc == i
        if keep.any():
            out[y0:y1, x0:x1][keep] = host
            moved += int(keep.sum())
    if not moved:
        return (labels,) + _rebuild(regions, labels, {r.id: r.group_id for r in regions}) + (0,)
    regions = refresh_regions(out, albedo_lab, regions)
    return (out,) + _rebuild(regions, out, {r.id: r.group_id for r in regions}) + (moved,)


def _de(a, b) -> float:
    return float(imageio.delta_e(np.asarray(a, np.float32)[None], np.asarray(b, np.float32)[None])[0])


# ---------------------------------------------------------------------- step 2c: letter counters

#: Letter counters (step 2c): the paint seen through a letter (the holes of an "8", the triangle
#: of a "4") is the paint. Inside a lettering region, or inside a small distinct region (source
#: 'small') at least COUNTER_RING_TEXT of whose 3 px outer ring is lettering, every 8-connected
#: blob of pixels in the main paint's hue (within COUNTER_HUE degrees) with at least COUNTER_C
#: of its chroma, COUNTER_MIN_PX to COUNTER_MAX_PX large and paint-like in its median too, goes
#: to the main paint's region touching the lettering most (the rest of the region, a letter's
#: outline, stays). Only lettering printed on the paint counts (COUNTER_ON_PAINT, and its own
#: colour not the paint's). The Ducati's "748": the counters are the red paint
#: mixed with the white and gold of the letters' edges (lighter, 15-20 deg toward orange, 0.8 of
#: the chroma), so neither the decal-gap rule (within dE 10 of the paint) nor the junk pruning
#: (a distinct small part is an island, and islands are exempt) took them, and a navy repaint of
#: the paint left them red.
COUNTER_HUE = 22.0
COUNTER_C = 0.6
COUNTER_MIN_PX = 4
COUNTER_MAX_PX = 300
COUNTER_RING_TEXT = 0.4
#: ... of lettering printed on the paint: at least this share of its 3 px outer ring is the main paint.
COUNTER_ON_PAINT = 0.5
#: Lettering: the regions stage's OCR-prompted masks.
TEXT_SOURCE = "text"


def fill_letter_counters(labels: np.ndarray, albedo_lab: np.ndarray, regions: list[Region], groups: list[ColorGroup]
                         ) -> tuple[np.ndarray, list[Region], list[ColorGroup], np.ndarray, int]:
    """Step 2c: the letter counters of the lettering on the main paint go back to the paint
    (see COUNTER_*). Region ids do not change (a small distinct region that is a counter is
    emptied into the paint, and the relabel after the snap drops it); the records are
    refreshed. Returns ``(labels, regions, groups, group_map, moved_px)``."""
    paint = main_paint(groups, DECAL_PAINT_CHROMA)
    if paint is None:
        return (labels,) + _rebuild(regions, labels, {r.id: r.group_id for r in regions}) + (0,)
    h_paint, c_paint = _hue(paint.albedo_lab), _chroma(paint.albedo_lab)
    by_id = {r.id: r for r in regions}

    def paint_like(lab) -> bool:
        return _hue_delta(_hue(lab), h_paint) <= COUNTER_HUE and _chroma(lab) >= COUNTER_C * c_paint

    g_of = np.zeros(int(labels.max()) + 1, np.int64)
    for r in regions:
        g_of[r.id] = r.group_id

    def on_paint(r: Region) -> bool:
        # lettering printed on the paint: most of its 3 px outer ring is the main paint, and its
        # own colour (the strokes, its median) is not the paint's (else the strokes pass as counters:
        # the navy letters of a FILA logo against a pair of blue jeans)
        if paint_like(r.albedo_lab):
            return False
        x0, y0, x1, y1 = r.bbox
        H, W = labels.shape
        qy0, qy1, qx0, qx1 = max(0, y0 - 4), min(H, y1 + 4), max(0, x0 - 4), min(W, x1 + 4)
        loc = labels[qy0:qy1, qx0:qx1]
        rm = loc == r.id
        ring = cv2.dilate(rm.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~rm
        return bool(ring.any()) and float((g_of[loc[ring]] == paint.id).mean()) >= COUNTER_ON_PAINT

    text = [r for r in regions if r.source == TEXT_SOURCE and r.group_id != paint.id and on_paint(r)]
    if not text:
        return (labels,) + _rebuild(regions, labels, {r.id: r.group_id for r in regions}) + (0,)
    pairs, cnt = adjacency(labels, int(labels.max()) + 1)
    touch: dict[int, dict[int, int]] = {}
    for (a, b), c in zip(pairs.tolist(), cnt.tolist()):
        touch.setdefault(a, {})[b] = c
        touch.setdefault(b, {})[a] = c

    def host_of(rid: int) -> Optional[int]:
        # the main paint's region touching it most (a counter goes to the paint the user paints)
        cand = {o: c for o, c in touch.get(rid, {}).items() if o in by_id and by_id[o].group_id == paint.id}
        return max(cand, key=lambda o: (cand[o], -o)) if cand else None

    out = labels.copy()
    text_ids = {r.id for r in text}
    k3 = np.ones((7, 7), np.uint8)

    def give_back(r: Region, host: int) -> int:
        """The paint-coloured blobs of region ``r`` (per pixel: the paint's hue and chroma; per
        blob: its median too) go to region ``host``; returns the pixels moved."""
        x0, y0, x1, y1 = r.bbox
        m = labels[y0:y1, x0:x1] == r.id
        p = albedo_lab[y0:y1, x0:x1]
        hue = (np.degrees(np.arctan2(p[..., 2], p[..., 1])) + 360.0) % 360.0
        dh = np.abs((hue - h_paint + 180.0) % 360.0 - 180.0)
        cand = m & (dh <= COUNTER_HUE) & (np.hypot(p[..., 1], p[..., 2]) >= COUNTER_C * c_paint)
        if not cand.any():
            return 0
        n, cc, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(cand)
        for i in range(1, n):
            if COUNTER_MIN_PX <= int(stats[i, cv2.CC_STAT_AREA]) <= COUNTER_MAX_PX:
                blob = cc == i
                if paint_like(np.median(p[blob], axis=0)):
                    keep |= blob
        if keep.any():
            out[y0:y1, x0:x1][keep] = host
        return int(keep.sum())

    moved = 0
    # small distinct regions enclosed by lettering (the holes of an "8" stamped as a small part,
    # which may hold a sliver of the letter's outline: only the paint-coloured pixels go)
    for r in regions:
        if r.source != "small" or r.group_id == paint.id or r.area < COUNTER_MIN_PX:
            continue
        x0, y0, x1, y1 = r.bbox
        H, W = labels.shape
        qy0, qy1, qx0, qx1 = max(0, y0 - 4), min(H, y1 + 4), max(0, x0 - 4), min(W, x1 + 4)
        loc = labels[qy0:qy1, qx0:qx1]
        rm = loc == r.id
        ring = cv2.dilate(rm.astype(np.uint8), k3).astype(bool) & ~rm
        if not ring.any():
            continue
        ids, c = np.unique(loc[ring], return_counts=True)
        share = float(sum(cc for i, cc in zip(ids.tolist(), c.tolist()) if i in text_ids)) / float(c.sum())
        if share < COUNTER_RING_TEXT:
            continue
        hosts = [host_of(int(i)) for i in ids.tolist() if i in text_ids] + [host_of(r.id)]
        hosts = [x for x in hosts if x is not None]
        if hosts:
            moved += give_back(r, max(set(hosts), key=lambda x: (by_id[x].area, -x)))
    # counters inside a lettering region's own mask
    for t in text:
        host = host_of(t.id)
        if host is not None:
            moved += give_back(t, host)
    if not moved:
        return (labels,) + _rebuild(regions, labels, {r.id: r.group_id for r in regions}) + (0,)
    regions = refresh_regions(out, albedo_lab, regions)
    return (out,) + _rebuild(regions, out, {r.id: r.group_id for r in regions}) + (moved,)


# ---------------------------------------------------------------------- step 3: after the snap

def _relabel(labels: np.ndarray, albedo_lab: np.ndarray, regions: list[Region]
             ) -> tuple[np.ndarray, list[Region], list[ColorGroup], np.ndarray, np.ndarray]:
    """:func:`relabel` plus the id map it applied (old id -> new id, -1 when emptied). The old
    ids may have gaps (a region an earlier step emptied, like a letter counter given back to the
    paint); every region keeps its old id until here."""
    labels = np.ascontiguousarray(labels, np.int32)
    n = max(len(regions), int(labels.max()) + 1, max((r.id for r in regions), default=-1) + 1)
    labels, mapping = compact(labels, n)
    labels = labels.astype(np.int32)
    moved: list[Region] = []
    for r in regions:
        new = int(mapping[r.id]) if r.id < len(mapping) else -1
        if new >= 0:
            moved.append(replace(r, id=new))
    moved = refresh_regions(labels, albedo_lab, moved)
    regions, groups, group_map = _rebuild(moved, labels, {r.id: r.group_id for r in moved})
    _check_partition(labels, regions)
    return labels, regions, groups, group_map, np.asarray(mapping, np.int64)


def relabel(labels: np.ndarray, albedo_lab: np.ndarray, regions: list[Region]) -> tuple[np.ndarray, list[Region], list[ColorGroup], np.ndarray]:
    """Records around a label map whose boundaries moved (the boundary snap): ids made
    contiguous again (a region the snap emptied disappears), every region's statistics
    refreshed, each region kept in its group. Returns ``(labels, regions, groups, group_map)``."""
    return _relabel(labels, albedo_lab, regions)[:4]


# ---------------------------------------------------------------------- step 4: other materials

def _material_ids(groups: Sequence[ColorGroup]) -> list[int]:
    """The unlocked paint-family groups that are much less chromatic and much darker than the
    main paint (MAT_*): another material in the paint's hue."""
    main = main_paint(groups)
    if main is None:
        return []
    family = set(paint_family(groups))
    c_main, l_main = _chroma(main.albedo_lab), float(main.albedo_lab[0])
    return [int(g.id) for g in groups if g.id in family and g.id != main.id and not g.locked
            and _chroma(g.albedo_lab) < MAT_C_RATIO * c_main and float(g.albedo_lab[0]) < l_main - MAT_L_MARGIN]


def lock_materials(groups: list[ColorGroup]) -> list[int]:
    """Step 4: lock (in place) every paint-family group that is much less chromatic and much
    darker than the main paint: another material in the paint's hue. Returns the ids."""
    ids = _material_ids(groups)
    for g in groups:
        if g.id in ids:
            g.locked = True
    return ids


# ---------------------------------------------------------------------- step 5: own-coloured objects

_PHOTO_LAB: "OrderedDict[bytes, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_PHOTO_LAB_LOCK = threading.Lock()
_PHOTO_LAB_KEEP = 2


def _photo_hue_chroma(photo_rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The photo's CIELAB hue (degrees) and chroma, float32 HxW, cached by the photo's content
    (the last two photos): every group edit and lock toggle recomputes the protect mask, and the
    conversion was most of its 0.2-0.8 s on the working image."""
    photo = np.ascontiguousarray(photo_rgb_u8)
    key = hashlib.blake2b(photo.data, digest_size=16).digest() + repr(photo.shape).encode()
    with _PHOTO_LAB_LOCK:
        hit = _PHOTO_LAB.get(key)
        if hit is not None:
            _PHOTO_LAB.move_to_end(key)
            return hit
    lab = imageio.rgb_to_lab(imageio.to_float(photo))
    h = ((np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) + 360.0) % 360.0).astype(np.float32)
    c = np.hypot(lab[..., 1], lab[..., 2]).astype(np.float32)
    with _PHOTO_LAB_LOCK:
        _PHOTO_LAB[key] = (h, c)
        while len(_PHOTO_LAB) > _PHOTO_LAB_KEEP:
            _PHOTO_LAB.popitem(last=False)
    return h, c


def protect_mask(photo_rgb_u8: np.ndarray, labels: np.ndarray, group_map: np.ndarray, groups: Sequence[ColorGroup],
                 islands: Optional[np.ndarray] = None) -> np.ndarray:
    """Step 5: bool HxW of the regions outside the paint (its unlocked family) and the
    islands whose photo pixels are mostly the old paint's colour: objects of that colour
    in their own right, which the engine's reflection stage must not recolour."""
    out = np.zeros(labels.shape, bool)
    main = main_paint(groups)
    if main is None:
        return out
    locked = {int(g.id) for g in groups if g.locked}
    paint_ids = [i for i in paint_family(groups) if i not in locked]
    h, chroma = _photo_hue_chroma(photo_rgb_u8)
    dh = np.abs((h - _hue(main.albedo_lab) + 180.0) % 360.0 - 180.0)
    cand = (chroma > PROTECT_CHROMA) & (dh < PROTECT_HUE)
    excl = np.isin(group_map, paint_ids)
    if islands is not None:
        excl |= islands
    n = np.bincount(labels.ravel())
    n_cand = np.bincount(labels[cand & ~excl].ravel(), minlength=n.size)
    n_excl = np.bincount(labels[excl].ravel(), minlength=n.size)
    for rid in np.flatnonzero(n_cand):
        area = int(n[rid] - n_excl[rid])
        if area >= PROTECT_MIN_PX and n_cand[rid] / max(1, area) >= PROTECT_FRAC:
            out |= (labels == rid) & ~excl
    return out


# ---------------------------------------------------------------------- orchestration

#: ``snap(image_u8, labels, group_map, groups, protect=, progress=) -> (labels, method)``,
#: i.e. :func:`recolor.segmentation.matting.snap_labels`.
Snapper = Callable[..., tuple[np.ndarray, str]]


@dataclass
class Refined:
    """The groups stage's result: a complete partition, the engine's per-pixel masks and
    what a regroup needs to reproduce the grouping (``origin``)."""
    labels: np.ndarray            # int32 HxW, 0..N-1
    regions: list[Region]
    groups: list[ColorGroup]
    group_map: np.ndarray         # int32 HxW, 0..G-1
    islands: np.ndarray           # bool HxW, decals carved out of the paint
    protect: np.ndarray           # bool HxW, own-coloured objects rule 8 must not recolour
    report: dict = field(default_factory=dict)
    #: int32 [N]: for every final region the id of the input region (the regions stage's
    #: label map) it descends from; -1 for a carved decal, which has none.
    origin: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))


def _part_report(groups: Sequence[ColorGroup]) -> list[dict]:
    return [{"id": int(g.id), "name": g.name, "kind": g.part, "instances": int(g.part_instances), "area": int(g.area)}
            for g in groups if g.part]


def _prune(photo_rgb_u8, albedo_lin, shading, labels, regions, groups, group_map, islands, fg, part_masks, params,
           user_flags=None):
    from .junk import prune_junk
    return prune_junk(photo_rgb_u8, albedo_lin, shading, labels, regions, groups, group_map, islands, fg=fg,
                      part_masks=part_masks or (), params=params, user_flags=user_flags)


def refine_groups(photo_rgb_u8: np.ndarray, albedo_lin: np.ndarray, labels: np.ndarray,
                  regions: list[Region], groups: list[ColorGroup], group_map: np.ndarray,
                  snap: Snapper, progress: Optional[Callable[[float, str], None]] = None,
                  parts: Optional[Sequence[int]] = None, residual: Optional[np.ndarray] = None,
                  shading: Optional[np.ndarray] = None, fg: Optional[np.ndarray] = None,
                  part_masks: Sequence[np.ndarray] = (), prune=None) -> Refined:
    """Steps 1-5 on a fresh grouping (see the module docstring), then, with ``prune`` (a
    :class:`junk.JunkParams`), the junk pruning (:func:`junk.prune_junk`, which reads the
    ``shading`` layer, the foreground matte ``fg`` and the detected parts' SAM masks
    ``part_masks``). ``snap`` moves the chromatic groups' boundaries (ViTMatte or its
    fallback); decal islands and detected parts are never moved by it, and every colour step
    leaves the detected parts in their part groups. ``parts`` are the input regions the
    regions stage cut out of chromatic regions (default: those of source PART_SOURCE;
    :func:`isolate_parts`). ``residual`` (the intrinsic residual, float32 linear) sharpens the
    shininess cues; without it clipped pixels alone count. ``report`` records what each step
    did (absorbed regions, isolated parts, absorbed highlights, decal px, gap px, snap method,
    locked and background group ids, protected px, the part groups, the pruned regions) for
    the stage message and the log; ``origin`` maps every final region back to the input
    region it descends from (see :func:`regroup_refined`)."""
    lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
    labels = np.ascontiguousarray(labels, np.int32)
    n_input = int(labels.max()) + 1

    def note(frac: float, msg: str) -> None:
        if progress is not None:
            progress(frac, msg)

    note(0.1, "Absorbing washed-out highlights into their paint")
    regions, groups, group_map, moves = absorb_washed(list(regions), list(groups), labels, lab)
    if parts is None:
        parts = [r.id for r in regions if r.source == PART_SOURCE]
    regions, groups, group_map, isolated = isolate_parts(list(regions), list(groups), labels, parts)
    feats = materials.shine_features(labels, albedo_lin, photo_rgb_u8, residual)
    regions = materials.tag_regions(regions, feats)
    regions, groups, group_map, hl_moves = materials.absorb_highlights(list(regions), list(groups), labels, feats)
    note(0.2, "Carving decals out of the paint")
    labels, regions, groups, group_map, islands = split_decals(labels, lab, regions, groups)
    n_decal_regions = len(regions) - n_input
    labels, regions, groups, group_map, gap_px = fill_decal_gaps(labels, lab, regions, groups)
    labels, regions, groups, group_map, counter_px = fill_letter_counters(labels, lab, regions, groups)
    stamped = [r.id for r in regions if r.source in ISLAND_SOURCES]
    if stamped:
        islands = islands | np.isin(labels, stamped)

    def snap_progress(frac: float, msg: str) -> None:
        note(0.25 + 0.6 * frac, msg)

    # the snap never moves a detected part's pixels (nor claims them for a paint)
    part_ids = [r.id for r in regions if r.part_kind]
    keep = islands | np.isin(labels, part_ids) if part_ids else islands
    snapped, method = snap(photo_rgb_u8, labels, group_map, groups, protect=keep, progress=snap_progress)
    labels, regions, groups, group_map, mapping = _relabel(snapped, lab, regions)
    regions, groups, group_map = grouping.enforce_parts(regions, groups, labels)
    origin = np.full(len(regions), -1, np.int32)
    for old in range(min(n_input, len(mapping))):
        if mapping[old] >= 0:
            origin[int(mapping[old])] = old
    note(0.9, "Locking other materials in the paint's colour")
    lock_parts(groups, [int(mapping[i]) for i in isolated if i < len(mapping) and mapping[i] >= 0], regions)
    locked = lock_materials(groups)
    grouping.annotate_groups(groups, regions, group_map)       # the paint family reads the panel view
    protect = protect_mask(photo_rgb_u8, labels, group_map, groups, islands)
    locked = sorted(set(locked) | {int(g.id) for g in groups if g.locked})
    report = {"absorbed": [m["region"] for m in moves], "isolated_parts": len(isolated),
              "highlights": [m["region"] for m in hl_moves], "decal_px": int(islands.sum()),
              "decal_regions": int(n_decal_regions), "gap_px": int(gap_px), "counter_px": int(counter_px),
              "snap": method, "locked": locked,
              "background": [int(g.id) for g in groups if g.is_background], "protect_px": int(protect.sum())}
    if prune is not None:
        note(0.95, "Folding lighting slivers into their neighbours")
        pr = _prune(photo_rgb_u8, albedo_lin, shading, labels, regions, groups, group_map, islands, fg, part_masks,
                    prune)
        regions, groups, group_map, islands, protect = pr.regions, pr.groups, pr.group_map, pr.islands, pr.protect
        report.update({"pruned": [{"region": m["region"], "rule": m["rule"], "into": m["into_name"], "px": m["area"]}
                                  for m in pr.log],
                       "locked": [int(g.id) for g in groups if g.locked],
                       "background": [int(g.id) for g in groups if g.is_background],
                       "protect_px": int(protect.sum()), "decal_px": int(islands.sum())})
    regions, groups, group_map, protect, sheen = absorb_sheen(photo_rgb_u8, lab, labels, regions, groups, group_map,
                                                             islands, protect)
    if sheen:
        report.update({"sheen": [{"group": m["name"], "into": m["into_name"], "px": m["area"]} for m in sheen],
                       "locked": [int(g.id) for g in groups if g.locked],
                       "background": [int(g.id) for g in groups if g.is_background], "protect_px": int(protect.sum())})
    report["parts"] = _part_report(groups)
    return Refined(labels, regions, groups, group_map, islands, protect, report, origin)


def absorb_sheen(photo_rgb_u8: np.ndarray, albedo_lab: np.ndarray, labels: np.ndarray, regions: list[Region],
                 groups: list[ColorGroup], group_map: np.ndarray, islands: Optional[np.ndarray],
                 protect: np.ndarray) -> tuple[list[Region], list[ColorGroup], np.ndarray, np.ndarray, list[dict]]:
    """The refinement's last step: a chromatic group that is the paint under a white sheen (its
    albedo the anchor's, its photo lighter and duller in the anchor's hue: the Torana's roof and
    boot lid facing the showroom's ceiling) joins that paint (:func:`grouping.absorb_lit` with
    :data:`grouping.SHEEN_LIT`); locked, background and part groups never move. The label map and
    the islands do not change; the protect mask and the panel view follow the new groups.
    Returns ``(regions, groups, group_map, protect, moves)``."""
    regions, groups, gm, moves = grouping.absorb_lit(list(regions), list(groups), labels, albedo_lab,
                                                     grouping.photo_lab_of(photo_rgb_u8), grouping.SHEEN_LIT)
    if not moves:
        return regions, groups, group_map, protect, []
    grouping.annotate_groups(groups, regions, gm)
    return regions, groups, gm, protect_mask(photo_rgb_u8, labels, gm, groups, islands), moves


def regroup_refined(photo_rgb_u8: np.ndarray, albedo_lin: np.ndarray, labels_input: np.ndarray,
                    origin: np.ndarray, labels: np.ndarray, regions: list[Region],
                    islands: Optional[np.ndarray], max_groups: Optional[int] = None,
                    delta_e: float = 10.0, parts: Sequence[int] = (), bg: Optional[np.ndarray] = None,
                    residual: Optional[np.ndarray] = None, sources: Optional[Sequence[str]] = None,
                    part_tags: Optional[dict[int, dict]] = None, shading: Optional[np.ndarray] = None,
                    fg: Optional[np.ndarray] = None, part_masks: Sequence[np.ndarray] = (), prune=None,
                    user_flags: Optional[dict] = None
                    ) -> tuple[list[Region], list[ColorGroup], np.ndarray, np.ndarray]:
    """Regroup a refined job the way its analysis grouped it, for any ``max_groups`` /
    ``delta_e``: the input regions ``labels_input`` (the regions stage's label map, before
    decals and the boundary snap) are clustered again (with their backdrop decisions ``bg``,
    their detected-part tags ``part_tags`` (input region id -> ``{"kind", "label", "plural",
    "instance"}``: one group per part kind again), the photo's lit / shadowed merge and the
    absorb rules), exactly as :func:`refine_groups` did, then every current region takes the
    group of the input region it descends from (``origin``). A region without one (a decal
    island, a piece a pixel-level split cut off: origin -1 or beyond the array) joins the
    nearest colour group within NEAREST_GROUP_DE, else a group of its own, as
    :func:`split_decals` assigns decals (a piece of a detected part goes back to its part
    group). ``parts`` (input region ids) are kept out of the paint as the analysis kept them
    (:func:`isolate_parts`), and ``sources`` (the input regions' regions-stage sources, by id)
    keep the lettering and small distinct parts out of the highlight absorb, as the analysis
    did. The material lock and the protect mask then run on the result, and with ``prune``
    the junk pruning, as the analysis ran it (``shading``, ``fg``, ``part_masks``: see
    :func:`refine_groups`), where a region the user voted on (``user_flags``, the per-region lock
    and background choices the caller re-applies afterwards) joins only a host that ends with its
    vote (:func:`junk.vote_agrees`: a sliver the user locked on its own keeps its group, a pruned
    sliver of a group the user locked goes back into it). With the analysis's own options the groups, flags and protect mask
    are the analysis's (up to groups of exactly equal area). The label map and the islands do
    not change. Returns ``(regions, groups, group_map, protect)``."""
    labels_input = np.ascontiguousarray(labels_input, np.int32)
    labels = np.ascontiguousarray(labels, np.int32)
    origin = np.asarray(origin, np.int64).ravel()
    lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
    n_in = int(labels_input.max()) + 1
    info: list[dict] = []
    for i in range(n_in):
        d: dict = {"id": i}
        if bg is not None and i < len(bg):
            d["bg"] = int(bg[i])
        if sources is not None and i < len(sources):
            d["source"] = str(sources[i])
        tag = (part_tags or {}).get(i)
        if tag and tag.get("kind"):
            d.update(part_kind=str(tag["kind"]), part_label=str(tag.get("label", "")),
                     part_plural=str(tag.get("plural", "")), part_instance=int(tag.get("instance", 0)))
        info.append(d)
    in_regions, in_groups, _ = grouping.group_regions(labels_input, albedo_lin, info, max_groups=max_groups,
                                                      delta_e=delta_e, photo_rgb_u8=photo_rgb_u8)
    in_regions, in_groups, _, _ = absorb_washed(list(in_regions), list(in_groups), labels_input, lab)
    in_regions, in_groups, _, isolated = isolate_parts(list(in_regions), list(in_groups), labels_input, parts)
    feats = materials.shine_features(labels_input, albedo_lin, photo_rgb_u8, residual)
    in_regions = materials.tag_regions(in_regions, feats)
    in_regions, in_groups, _, _ = materials.absorb_highlights(list(in_regions), list(in_groups), labels_input, feats)
    group_of = {r.id: r.group_id for r in in_regions}
    assignment: dict[int, int] = {}
    fresh = len(in_groups)
    for r in sorted(regions, key=lambda r: r.id):
        o = int(origin[r.id]) if r.id < len(origin) else -1
        if o in group_of:
            assignment[r.id] = group_of[o]
            continue
        gid = _nearest_group(r.albedo_lab, in_groups)
        if gid is None:
            gid, fresh = fresh, fresh + 1
        assignment[r.id] = gid
    regions, groups, group_map = _rebuild(list(regions), labels, assignment)
    regions, groups, group_map = grouping.enforce_parts(regions, groups, labels)
    moved = set(isolated)
    lock_parts(groups, [r.id for r in regions if r.id < len(origin) and int(origin[r.id]) in moved], regions)
    lock_materials(groups)
    grouping.annotate_groups(groups, regions, group_map)       # the paint family reads the panel view
    protect = protect_mask(photo_rgb_u8, labels, group_map, groups, islands)
    if prune is not None:
        pr = _prune(photo_rgb_u8, albedo_lin, shading, labels, regions, groups, group_map, islands, fg, part_masks,
                    prune, user_flags)
        regions, groups, group_map, protect = pr.regions, pr.groups, pr.group_map, pr.protect
    regions, groups, group_map, protect, _ = absorb_sheen(photo_rgb_u8, lab, labels, regions, groups, group_map,
                                                         islands, protect)
    return regions, groups, group_map, protect


def refine_after_edit(photo_rgb_u8: np.ndarray, albedo_lin: np.ndarray, labels: np.ndarray,
                      regions: list[Region], groups: list[ColorGroup], group_map: np.ndarray,
                      islands: Optional[np.ndarray], regrouped: bool) -> tuple[list[Region], list[ColorGroup], np.ndarray, np.ndarray]:
    """Keep a refined job consistent after a user edit of its grouping. After a regroup of
    the current regions (fresh groups, user flags gone; the fallback when a job has no
    :func:`regroup_refined` inputs) the absorb rule and the material lock run again; after
    every edit the protect mask is recomputed for the current paint. Labels and islands
    are pixel facts and do not change. Returns ``(regions, groups, group_map, protect)``.
    CPU only unless ``regrouped`` (the absorb rule's region medians run on the GPU)."""
    labels = np.ascontiguousarray(labels, np.int32)
    if regrouped:
        lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
        regions, groups, group_map, _ = absorb_washed(list(regions), list(groups), labels, lab)
        lock_materials(groups)
    return regions, groups, group_map, protect_mask(photo_rgb_u8, labels, group_map, groups, islands)
