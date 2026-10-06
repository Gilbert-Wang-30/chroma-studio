"""Material cues from the layers: how shiny a region is, whether it reflects like chrome, and
the highlights of a paint that clustered as white.

A broad specular reflection pushes the albedo of glossy paint toward white, so the highlight
zone of a fairing clusters with the white parts and stays unpainted when the paint is
repainted. Per region, over its own pixels (:func:`shine_features`):

* ``clip``: share of sensor-clipped pixels (max channel >= CLIP_SRGB); ``spec``: share of
  glints (a neutral positive residual above half its 99th percentile); ``hl``: share of
  highlight pixels (clipped, glint, or a positive residual above HL_PX_FRAC of the pixel's
  brightness). ``hl`` is the region's *shininess*.
* ``rem_lab``: the median photo colour of the unclipped remainder (unclipped pixels under the
  region's REM_Q luminance quantile, 2 px inside the rim of a region of at least REM_ERODE_PX
  pixels, so a decal's anti-aliased rim cannot fake the paint's hue): the paint under the
  highlight. ``hueR``: the circular mean resultant length of the photo hue over the
  chromatic pixels (1 = one stable hue, 0 = every hue: a mirror). ``cfrac``: share of
  chromatic pixels. ``lstd``: spread of the photo's log luminance.

Two uses, both after the clustering and before the decals:

1. :func:`absorb_highlights`: a bright, washed region touching a paint whose unclipped
   remainder carries that paint's hue joins the paint (the RX-78's lit shield facet, the
   BMW's nose). Guards: the candidate is not backdrop and not lettering, at most
   HL_MAX_AREA_RATIO of the paint's area, and not a lamp (a blown lamp is mostly clipped:
   ``clip`` above HL_CLIP_MAX).
2. :func:`chrome_regions`: an advisory only. A region that glints or clips like chrome, has a
   near-neutral albedo, no stable hue and a wide luminance spread is tagged ``chrome`` so the
   UI can show a badge; it is never locked or moved automatically (measured recall 6 of 29
   chrome parts, and false hits on dark glossy plastic). The shiny badge itself is based on
   the glint share (``max(clip, spec)``, ``Region.glint``), not on ``hl``: a strong positive
   residual is near-universal on glossy paint (98 of the yellow BMW's 113 regions at 15 %),
   glints at 20 % mark two to five groups per photo.

Everything is CPU numpy plus the GPU region medians of :mod:`labelops`; nothing loads a model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import cv2
import numpy as np

from .. import imageio
from ..types import ColorGroup, Region
from . import grouping
from .labelops import adjacency, region_medians

CLIP_SRGB = 0.98
CLIP_LIN = CLIP_SRGB ** 2.2
CHROMA_PX = 8.0          # a photo pixel above this CIELAB chroma has a hue
REM_Q = 0.6              # the remainder: unclipped pixels below this luminance quantile
REM_ERODE_PX = 400       # regions at least this big take the remainder 2 px inside their rim
HL_PX_FRAC = 0.3         # a pixel is a highlight when its positive residual is this share of its brightness
# absorb_highlights (rule i)
HL_PAINT_CHROMA = 30.0   # a paint group: non-background, unlocked, albedo chroma at least this
HL_CAND_C_RATIO = 0.95   # candidate albedo chroma below this share of the paint's (washed)
HL_CAND_AL_MIN = 55.0    # the albedo was pushed toward white ...
HL_CAND_PL_MIN = 50.0    # ... and the photo is bright there
HL_MIN_TOUCH = 20        # shared boundary px with the paint group
HL_HUE_TOL = 18.0        # remainder hue within this of the paint's hue
HL_REM_C_MIN = 10.0      # remainder normalised chroma at least this (a colour is there)
HL_HUE_R_MIN = 0.6       # the remainder's hue is stable
HL_SHINY_MIN = 0.15      # highlight-pixel share at least this: the region is shiny
HL_MAX_AREA_RATIO = 0.5  # not larger than half the paint group
HL_CLIP_MAX = 0.6        # a candidate mostly clipped is a lamp, not a highlight of the paint
# chrome_regions (rule ii, advisory)
CH_MIN_PX = 500
CH_SHINY_MIN = 0.25      # max(clip, spec) share at least this
CH_HUE_R_MAX = 0.55      # no stable hue ...
CH_CFRAC_MAX = 0.35      # ... or few chromatic pixels
CH_AC_MAX = 16.0         # albedo near-neutral
CH_AL_MIN, CH_AL_MAX = 25.0, 78.0   # not a white albedo (glossy white plastic glints too)
CH_REM_L_MAX = 70.0      # the non-highlight remainder is not white either
CH_LSTD_MIN = 0.55       # reflections: wide luminance spread
#: Region sources that are decals by construction: never absorbed as a highlight.
DECAL_SOURCES = ("text", "small")


def _lum(lin: np.ndarray) -> np.ndarray:
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def _chroma(lab) -> float:
    return float(math.hypot(float(lab[1]), float(lab[2])))


def _hue(lab) -> float:
    return float((math.degrees(math.atan2(float(lab[2]), float(lab[1]))) + 360.0) % 360.0)


def _hue_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass
class ShineFeatures:
    """Per-region cues (index = region id); see the module docstring."""
    n: int
    area: np.ndarray
    alb: np.ndarray        # [n, 3] median albedo CIELAB
    pho: np.ndarray        # [n, 3] median photo CIELAB
    clip: np.ndarray
    spec: np.ndarray
    hl: np.ndarray
    cfrac: np.ndarray
    hue_r: np.ndarray
    rem_lab: np.ndarray    # [n, 3]
    lstd: np.ndarray
    touch: list            # [n] {neighbour id: shared boundary px}

    def rem_chroma(self, i: int) -> float:
        return _chroma(grouping.normalise_lab(self.rem_lab[i]))


def shine_features(labels: np.ndarray, albedo_lin: np.ndarray, photo_rgb_u8: np.ndarray,
                   residual: Optional[np.ndarray] = None, spec_q: Optional[float] = None) -> ShineFeatures:
    """The per-region cues from the label map, the albedo, the photo and (optionally) the
    intrinsic residual; without a residual ``spec`` is 0 and ``hl`` counts clipped pixels only.
    ``spec_q`` is the glint threshold's reference, the 99th percentile of the residual's neutral
    part (:func:`spec_reference`), computed from ``residual`` when not given: a caller working on a
    window of the image passes the whole image's."""
    labels = np.ascontiguousarray(labels, np.int32)
    n = int(labels.max()) + 1
    flat = labels.ravel().astype(np.int64)
    area = np.bincount(flat, minlength=n).astype(np.float64)
    alb = np.ascontiguousarray(albedo_lin, np.float32)
    pho = np.ascontiguousarray(imageio.srgb_to_linear(imageio.to_float(photo_rgb_u8)), np.float32)
    lab_a = imageio.linear_to_lab(alb)
    lab_p = imageio.linear_to_lab(pho)
    alb_med = region_medians(labels, lab_a, n).astype(np.float64)
    pho_med = region_medians(labels, lab_p, n).astype(np.float64)
    lp = _lum(pho)
    clip_px = pho.max(axis=-1) >= CLIP_LIN
    if residual is not None:
        res = np.asarray(residual, np.float32)
        lr = _lum(res)
        gray = np.clip(res, 0, None).min(axis=-1)
        q = float(spec_q) if spec_q is not None else spec_reference(res)
        spec_px = (gray / max(q, 1e-4)) > 0.5
        strong_px = np.clip(lr, 0, None) > HL_PX_FRAC * np.clip(lp, 1e-4, None)
    else:
        spec_px = np.zeros(labels.shape, bool)
        strong_px = np.zeros(labels.shape, bool)
    clip = np.bincount(flat, weights=clip_px.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
    spec = np.bincount(flat, weights=spec_px.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
    hl_px = clip_px | spec_px | strong_px
    hl = np.bincount(flat, weights=hl_px.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
    # rim pixels (2 px inside a region boundary), excluded from the remainder of big regions
    e = np.zeros(labels.shape, bool)
    e[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    e[1:, :] |= labels[1:, :] != labels[:-1, :]
    e[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    e[:-1, :] |= labels[1:, :] != labels[:-1, :]
    rim = cv2.dilate(e.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool).ravel()
    c_px = np.hypot(lab_p[..., 1], lab_p[..., 2])
    chrom = c_px >= CHROMA_PX
    cfrac = np.bincount(flat, weights=chrom.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
    ang = np.arctan2(lab_p[..., 2], lab_p[..., 1])
    cx = np.bincount(flat, weights=(np.cos(ang) * chrom).ravel(), minlength=n)
    cy = np.bincount(flat, weights=(np.sin(ang) * chrom).ravel(), minlength=n)
    nch = np.bincount(flat, weights=chrom.ravel().astype(np.float64), minlength=n)
    hue_r = np.where(nch > 0, np.hypot(cx, cy) / np.maximum(nch, 1), 0.0)
    ll = np.log(np.clip(lp, 1e-4, None)).ravel()
    m1 = np.bincount(flat, weights=ll, minlength=n) / np.maximum(area, 1)
    m2 = np.bincount(flat, weights=ll * ll, minlength=n) / np.maximum(area, 1)
    lstd = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
    rem_lab = pho_med.copy()
    order = np.argsort(flat, kind="stable")
    starts = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=n))])
    lp_flat = lp.ravel()
    clip_flat = clip_px.ravel()
    lab_flat = lab_p.reshape(-1, 3)
    for i in range(n):
        idx = order[starts[i]:starts[i + 1]]
        if len(idx) == 0:
            continue
        if len(idx) > 20000:
            idx = idx[:: len(idx) // 20000 + 1]
        ok = ~clip_flat[idx]
        if len(idx) >= REM_ERODE_PX and (ok & ~rim[idx]).sum() >= 64:
            ok &= ~rim[idx]
        if ok.sum() >= 16:
            sel = idx[ok]
            q = np.quantile(lp_flat[sel], REM_Q)
            low = sel[lp_flat[sel] <= q]
            if len(low) >= 8:
                rem_lab[i] = np.median(lab_flat[low], axis=0)
    pairs, cnt = adjacency(labels, n)
    touch: list[dict] = [dict() for _ in range(n)]
    for (a, b), c in zip(pairs.tolist(), cnt.tolist()):
        touch[a][b] = touch[a].get(b, 0) + c
        touch[b][a] = touch[b].get(a, 0) + c
    return ShineFeatures(n, area, alb_med, pho_med, clip, spec, hl, cfrac, hue_r, rem_lab, lstd, touch)


def spec_reference(residual: np.ndarray) -> float:
    """The glint threshold's reference of :func:`shine_features`: the 99th percentile of the
    residual's neutral part (its smallest channel, clipped at 0), on at most 1M pixels."""
    gray = np.clip(np.asarray(residual, np.float32), 0, None).min(axis=-1)
    return float(np.quantile(gray.ravel()[:: max(1, gray.size // 1_000_000)], 0.99))


def chrome_regions(regions: Sequence[Region], feats: ShineFeatures) -> set[int]:
    """Ids of the regions that reflect their surroundings like chrome or glass (rule ii):
    an advisory for the UI's badge, never a lock."""
    out = set()
    for r in regions:
        i = r.id
        if i >= feats.n or r.area < CH_MIN_PX:
            continue
        if (max(feats.clip[i], feats.spec[i]) >= CH_SHINY_MIN and _chroma(feats.alb[i]) <= CH_AC_MAX
                and CH_AL_MIN <= feats.alb[i][0] <= CH_AL_MAX and feats.rem_lab[i][0] <= CH_REM_L_MAX
                and (feats.hue_r[i] <= CH_HUE_R_MAX or feats.cfrac[i] <= CH_CFRAC_MAX) and feats.lstd[i] >= CH_LSTD_MIN):
            out.add(int(i))
    return out


def tag_regions(regions: Sequence[Region], feats: ShineFeatures) -> list[Region]:
    """Region records with their ``shiny`` share (``hl``), their ``glint`` share (the larger
    of ``clip`` and ``spec``: what the UI's shiny badge is based on) and the ``chrome``
    advisory filled in (the groups built from them carry the area-weighted values and the
    finish badge)."""
    chrome = chrome_regions(regions, feats)
    out = []
    for r in regions:
        shiny = float(feats.hl[r.id]) if r.id < feats.n else 0.0
        glint = float(max(feats.clip[r.id], feats.spec[r.id])) if r.id < feats.n else 0.0
        out.append(replace(r, shiny=round(shiny, 4), glint=round(glint, 4), chrome=r.id in chrome))
    return out


def absorb_highlights(regions: list[Region], groups: list[ColorGroup], labels: np.ndarray, feats: ShineFeatures
                      ) -> tuple[list[Region], list[ColorGroup], np.ndarray, list[dict]]:
    """Rule i: every region that is a highlight of a touching paint (see the module
    docstring) joins that paint's group; ties go to the closest remainder hue. A part group
    (a detected part) is never a paint here and its regions never move. Returns the rebuilt
    ``(regions, groups, group_map)`` (flags and custom names kept) and the moves."""
    g_of = {rid: g.id for g in groups for rid in g.region_ids}
    by_g = {g.id: g for g in groups}
    paints = [g for g in groups if not g.is_background and not g.locked and not g.part
              and _chroma(g.albedo_lab) >= HL_PAINT_CHROMA]
    moves: list[dict] = []
    assignment = {r.id: r.group_id for r in regions}
    for r in regions:
        i = r.id
        if i >= feats.n or r.backdrop or r.source in DECAL_SOURCES or r.part_kind:
            continue
        if by_g[g_of[i]].is_background or by_g[g_of[i]].locked or by_g[g_of[i]].part:
            continue
        if feats.hl[i] < HL_SHINY_MIN or feats.clip[i] > HL_CLIP_MAX:
            continue
        if feats.alb[i][0] < HL_CAND_AL_MIN or feats.pho[i][0] < HL_CAND_PL_MIN:
            continue
        rem_c = feats.rem_chroma(i)
        if rem_c < HL_REM_C_MIN or feats.hue_r[i] < HL_HUE_R_MIN:
            continue
        best = None
        for g in paints:
            if g_of[i] == g.id or r.area > HL_MAX_AREA_RATIO * g.area:
                continue
            shared = sum(feats.touch[i].get(o, 0) for o in g.region_ids)
            if shared < HL_MIN_TOUCH:
                continue
            if _chroma(feats.alb[i]) > HL_CAND_C_RATIO * _chroma(g.albedo_lab):
                continue
            dh = _hue_delta(_hue(feats.rem_lab[i]), _hue(g.albedo_lab))
            if dh > HL_HUE_TOL:
                continue
            score = dh + 0.001 * (1e6 / max(shared, 1))
            if best is None or score < best[0]:
                best = (score, g.id, dh, shared)
        if best is not None:
            _, gid, dh, shared = best
            assignment[i] = gid
            moves.append({"region": int(i), "from_group": int(g_of[i]), "to_group": int(gid), "dh": round(dh, 1),
                          "touch": int(shared), "shiny": round(float(feats.hl[i]), 3), "rem_C": round(rem_c, 1),
                          "area": int(r.area)})
    if not moves:
        return regions, groups, grouping._group_map(regions, labels, len(groups)), moves
    carry = {g.id: {"name": grouping._keep_name(g), "locked": g.locked, "is_background": g.is_background} for g in groups}
    regions, groups, group_map = grouping._finalize(regions, labels, assignment, carry)
    grouping._mark_background(groups, group_map, regions)
    grouping._check_state(regions, groups)
    return regions, groups, group_map, moves
