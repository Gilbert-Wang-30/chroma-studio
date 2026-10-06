"""Junk-group pruning: tiny groups that are only a lighting or colour-cast variant of a
touching group (a shadow with a colour shift, the shadowed rim of a detected part, a seam on a
smooth gradient) join that group, as the last step of the groups stage
(:func:`refine.refine_groups`) and of a regroup (:func:`refine.regroup_refined`), after the
boundary snap and the locks. Real small parts are exempt, and a detected part is never pruned.

The owner's case: the red coupe's flare shadow, a dark-red, colour-shifted strip that the
clustering gave a group of its own and the material lock locked, so a repaint of the body left
it red. Measured on the ten-photo part reference set (16 by-eye junk groups, 17 real small
groups; two fresh analyses): 16 -> 9-10 junk groups left, by-eye severity 25 -> 13-15, the
robot's feet rim folded into the feet, part isolation unchanged; no real small part is pruned
(the one the strict test counts as lost, the BMW's far gold preload adjuster, shares a locked
group with its twin since the gold caliper they were grouped with became a part of its own).
What stays needs other signals: reflections of other objects are baked into the albedo,
cavity shadows stamped as small distinct parts are islands on purpose.

The label map never changes, only which group a region belongs to, so the snap, the decal
islands and the region ids stay valid; merged regions leave the islands and the protect mask
is recomputed for the new grouping.

Four tests, applied to every candidate region (the regions of a non-background group below
``max_frac`` of the object, smallest group first, each merge visible to the next), against
the groups that touch it; every comparison is the region's pixels against the neighbour's
pixels in a RING_PX ring around it, i.e. at the seam, not against the neighbour's median:

0. **Part rim** (rule 2b, the detected parts' SAM masks): a region lying (``rim_inside``)
   within ``rim_px`` of a detected part's mask, whose object ring the part owns at least
   ``rim_share`` of, in the part's hue (within ``rim_hue``) and no more chromatic nor lighter
   than the part (``rim_c_margin``, ``rim_l_margin``), is the part's own shadowed or
   anti-aliased edge: SAM's box-prompted mask stops a few px short of the part's outline and
   the band between is a superpixel of its own. It joins the part. The robot's red feet kept a
   dark-red rim of 900-1300 px as a group of its own, which the material lock locked and the
   shadow test could not decide (it kept 0.46 of the feet's chroma), and a navy repaint of the
   paint, which the rim clustered with, drew a seam around the red feet.
1. **Shadow** (the intrinsic layers). The neighbour owns at least ``host_share`` of the
   object part of the ring; the photo is darker (luminance ratio <= ``rho_dark``); the
   shading layer carries at least ``shade_explains`` of that log-luminance step; and the
   chromaticity moved the way light moves it: the lightness-normalised photo chroma is not
   higher than the neighbour's (+ ``chroma_margin``) with the hue kept (within ``hue_tol``),
   or the photo's chromatic log-shift equals the shading layer's (within ``cast_tol``: a cast
   toward the light's colour). Against a chromatic neighbour (>= ``lost_hue_c``) a candidate
   that went neutral or kept less than ``keep_c`` of its chroma is kept (a black part or a
   deep shadow: undecidable; the floor seen through a bumper frame kept 0.54). Only darker
   candidates are tested: a lighter branch (a rim light) took a rider's glove for a highlight.
2. **Gradient**: the shared boundary carries no photo edge (median |grad log Y| <=
   ``edge_lum`` and |grad log chromaticity| <= ``edge_chr``) and the local albedo cast and
   photo difference are small: a seam on a smooth surface.
3. **Significance**: the group is below ``sig_frac`` of the object and not locked, the
   closest-coloured touching group has the same body colour (lightness-normalised (a, b)
   within ``sig_cast``) and the photo cannot tell them apart (CIEDE2000 <= ``sig_photo_de``).

The part rim and the shadow test move single regions; the colour tests (2, 3) only dissolve a
whole group. Exempt from every test: regions of an ``exempt_sources`` source (lettering 'text',
named parts 'named', the wheel split 'wheel', recovered parts 'part'), groups at least
``island_exempt`` of whose pixels are decal islands, groups mostly off the matte, and every part
group (**rule 1**: a detected grip or door handle is tiny and has its neighbour's colour on
purpose; without the rule the tests dissolved 4 of 35 detected parts). A region the user voted
on (``user_flags``: a lock or background choice, recorded for every region of the group it was
made on) moves only into a host that ends with the flag it voted for (the area-weighted majority
of the host's votes, as the regroup re-applies them, else the host's own flag): a sliver the user
split off and locked keeps its group, while a pruned sliver of a group the user locked, which
carries that group's vote, goes back into it (exempted outright, it came back as a locked junk
group after every lock and regroup). A locked group (the material lock) is a candidate for the
part rim, the shadow and the gradient tests (tests 0-2: evidence that the pixels are one
surface under different light), never for the significance test (3: a colour argument, which
the lock's own colour argument outranks). A group never merges into a background group or a
smaller one; a part group hosts only a candidate that lies inside that part's mask (**rule 2**:
at least ``inside_part`` of the region under the part's SAM mask, dilated by ``part_dilate``
px; such a candidate goes to the part even when a test picked another neighbour, provided the
part is larger) or the part's rim (rule 2b), so part groups hold part pixels only.

``backdrop_crumbs``: tiny background groups join the touching background group of the
closest colour (clutter rows while the background is ignored).

Deterministic, CPU numpy + cv2 (about 0.3 s per 1536-px photo); nothing loads a model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Collection, Mapping, Optional, Sequence

import cv2
import numpy as np

from .. import imageio
from ..types import ColorGroup, Region
from . import grouping

RING_PX = 5          # outer ring (and neighbour band) width, px at work resolution
EDGE_DILATE = 1      # a boundary pixel's edge strength is the max gradient within this many px
EPS = 1e-4
RIM_HUE_C = 8.0      # CIELAB chroma from which a colour has a hue (the part-rim test)


@dataclass(frozen=True)
class JunkParams:
    """The pruning's knobs (the values measured on the part reference set). ``shadow``,
    ``gradient``, ``significance`` and ``part_rim`` switch a test off for an ablation; the
    product runs them all."""
    max_frac: float = 0.004          # candidates: non-background groups below this share of the object (matte > 0.5)
    min_fg: float = 0.6              # ... at least this share of whose pixels lie inside the matte
    nb_min_share: float = 0.15       # a touching group must own this much of the (object) ring
    # 0. part rim (rule 2b)
    part_rim: bool = True
    rim_px: int = 4                  # the part's SAM mask dilated by this many px ...
    rim_inside: float = 0.9          # ... holds at least this share of the candidate
    rim_share: float = 0.5           # the part owns at least this share of the object part of the ring
    rim_hue: float = 30.0            # the candidate's albedo hue within this of the part's ...
    rim_c_margin: float = 5.0        # ... and no more chromatic (+ margin) ...
    rim_l_margin: float = 5.0        # ... nor lighter (+ margin) than the part's
    # 1. shadow
    shadow: bool = True
    host_share: float = 0.3          # the neighbour's share of the object part of the ring
    rho_dark: float = 0.85           # the photo at most this bright against the neighbour
    shade_explains: float = 0.3
    chroma_margin: float = 3.0
    hue_tol: float = 35.0
    neutral_c: float = 6.0           # normalised photo chroma below this has no reliable hue
    lost_hue_c: float = 20.0         # a neighbour at least this chromatic whose candidate went neutral: kept
    keep_c: float = 0.6              # ... and, against such a neighbour, a shadow keeps at least this share of its chroma
    cast_tol: float = 0.08
    # 2. gradient
    gradient: bool = True
    edge_lum: float = 0.12
    edge_chr: float = 0.12
    grad_cast: float = 8.0
    grad_photo_de: float = 6.0
    grad_share: float = 0.5
    # 3. significance
    significance: bool = True
    sig_frac: float = 0.004
    sig_cast: float = 12.0
    sig_photo_de: float = 6.0
    # exemptions
    exempt_sources: tuple = ("text", "named", "wheel", "part")
    island_exempt: float = 0.5
    # detected parts (rule 2)
    inside_part: float = 0.5
    part_dilate: int = 2
    # 4. crumbs: a group this small whose pixels are only crumbs (at least crumb_pieces 8-connected
    # pieces, none above crumb_piece px) is dissolved into the group owning most of its ring, even a
    # small distinct part's island (a 58 px fleck of the Ducati's gold letter edging in 18 pieces, a
    # "0.00 %" row that kept an orange fleck on the "8" under a navy repaint); lettering, detected
    # parts and the user's choices never are
    crumb_group: int = 150
    crumb_piece: int = 40
    crumb_pieces: int = 4
    # optional
    backdrop_crumbs: bool = False


DEFAULT = JunkParams()
#: The value that turns a rule off, for each field added after the pruning first ran: a regroup of
#: a job whose seed recorded parameters without that field reruns the pruning as the analysis did.
LEGACY_OFF = {"crumb_group": 0}


# ---------------------------------------------------------------------- evidence

def _lum(lin: np.ndarray) -> np.ndarray:
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def _grad_mag(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    xs = cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(xs, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(xs, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy)


@dataclass
class Context:
    """Per-photo images the tests read (built once per pruning)."""
    photo_lin: np.ndarray            # float32 HxWx3 linear photo
    photo_lab: np.ndarray            # float32 HxWx3 CIELAB of the photo
    albedo_lab: np.ndarray           # float32 HxWx3 CIELAB of the albedo
    shading: np.ndarray              # float32 HxWx3 linear shading (coloured)
    g_lum: np.ndarray                # |grad log Y| of the photo (sigma 1), max-filtered by EDGE_DILATE
    g_chr: np.ndarray                # |grad| of the photo's log-chromaticity, max-filtered
    object_px: int                   # matte area (fg > 0.5), or the whole image without a matte
    fg: Optional[np.ndarray] = None  # bool HxW, the matte > 0.5 (None: everything is object)


def build_context(photo_u8: np.ndarray, albedo_lin: np.ndarray, shading_lin: Optional[np.ndarray],
                  fg: Optional[np.ndarray] = None) -> Context:
    """The per-photo images of the tests. Without a shading layer the shading is flat (grey 1),
    so the shadow test never passes (it cannot tell a shadow from a darker material)."""
    photo = imageio.srgb_to_linear(imageio.to_float(photo_u8)).astype(np.float32)
    plab = imageio.linear_to_lab(photo).astype(np.float32)
    alab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, np.float32)).astype(np.float32)
    shd = np.ones_like(photo) if shading_lin is None else np.ascontiguousarray(shading_lin, np.float32)
    fl = 1.0 / 255.0                 # keeps sensor noise in near-black pixels out of the log gradients
    ly = np.log(np.clip(_lum(photo), fl, None))
    lrg = np.log(np.clip(photo[..., 0], fl, None)) - np.log(np.clip(photo[..., 1], fl, None))
    lbg = np.log(np.clip(photo[..., 2], fl, None)) - np.log(np.clip(photo[..., 1], fl, None))
    k = np.ones((2 * EDGE_DILATE + 1, 2 * EDGE_DILATE + 1), np.uint8)
    g_lum = cv2.dilate(_grad_mag(ly), k)
    g_chr = cv2.dilate(np.hypot(_grad_mag(lrg), _grad_mag(lbg)), k)
    fgb = (np.asarray(fg) > 0.5) if fg is not None else None
    object_px = int(fgb.sum()) if fgb is not None else int(photo.shape[0] * photo.shape[1])
    return Context(photo, plab, alab, shd, g_lum, g_chr, max(object_px, 1), fgb)


@dataclass
class PairEvidence:
    """A candidate region against one touching group n, on the region's pixels and n's band."""
    nb: int
    share: float                      # n's share of the ring (all ring pixels)
    share_obj: float                  # n's share of the non-background part of the ring
    boundary: int                     # 4-connected boundary px between the two
    rho_photo: float                  # Y_candidate / Y_band, photo (linear)
    rho_shade: float                  # the same on the shading layer
    photo_g: tuple                    # median photo CIELAB of the candidate
    photo_b: tuple                    # ... of the band
    alb_g: tuple                      # median albedo CIELAB of the candidate
    alb_b: tuple                      # ... of the band
    chrom_photo: tuple                # chromatic part of log(photo_g) - log(photo_band)
    chrom_shade: tuple                # chromatic part of log(shading_g) - log(shading_band)
    e_lum: float                      # median |grad log Y| (photo) on the shared boundary
    e_chr: float                      # median |grad log chromaticity| (photo) on the shared boundary

    @property
    def shade_explains(self) -> float:
        """Share of the photo's log-luminance step that the shading layer carries."""
        lp = math.log(max(self.rho_photo, EPS))
        if abs(lp) < 1e-6:
            return 0.0
        return math.log(max(self.rho_shade, EPS)) / lp


@dataclass
class Evidence:
    gid: int
    area: int
    object_share: float
    ring_bg_share: float = 0.0
    fg_share: float = 1.0             # share of the candidate's pixels inside the matte
    sources: tuple = ()
    island_share: float = 0.0
    pairs: list = field(default_factory=list)    # PairEvidence, largest share first


def _median_rows(x: np.ndarray) -> np.ndarray:
    return np.median(x, axis=0) if len(x) else np.zeros(x.shape[1:], np.float32)


def evidence(ctx: Context, group_map: np.ndarray, gid: int, labels: np.ndarray, sources, islands: Optional[np.ndarray],
             bg_ids: set, nb_min_share: float = 0.1, ring_px: int = RING_PX, include_bg: bool = False,
             region: Optional[int] = None) -> Evidence:
    """The evidence for group ``gid`` (or, with ``region``, for that one region of it, whose
    ring then leaves out the group's other regions) against every touching group owning at
    least ``nb_min_share`` of its outer ring (background groups only with ``include_bg``)."""
    H, W = group_map.shape
    sel = (group_map == gid) if region is None else (labels == region)
    ys, xs = np.nonzero(sel)
    area = int(ys.size)
    ev = Evidence(gid=gid, area=area, object_share=area / ctx.object_px)
    if area == 0:
        return ev
    pad = ring_px + 3
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    gm = group_map[y0:y1, x0:x1]
    m = sel[y0:y1, x0:x1]
    kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    ring = cv2.dilate(m.astype(np.uint8), kr).astype(bool) & ~m & (gm != gid)
    rid = np.unique(labels[y0:y1, x0:x1][m])
    ev.sources = tuple(sorted({str(sources[i]) for i in rid}))
    if islands is not None:
        ev.island_share = float(islands[y0:y1, x0:x1][m].mean())
    if ctx.fg is not None:
        ev.fg_share = float(ctx.fg[y0:y1, x0:x1][m].mean())
    nb, cnt = np.unique(gm[ring], return_counts=True)
    if nb.size == 0:
        return ev
    tot = float(cnt.sum())
    bg_cnt = float(sum(c for b, c in zip(nb.tolist(), cnt.tolist()) if b in bg_ids))
    ev.ring_bg_share = bg_cnt / tot
    obj_tot = max(tot - bg_cnt, 1.0)
    P = ctx.photo_lin[y0:y1, x0:x1]
    S = ctx.shading[y0:y1, x0:x1]
    A = ctx.albedo_lab[y0:y1, x0:x1]
    PL = ctx.photo_lab[y0:y1, x0:x1]
    GL = ctx.g_lum[y0:y1, x0:x1]
    GC = ctx.g_chr[y0:y1, x0:x1]
    pg = _median_rows(P[m])
    sg = _median_rows(S[m])
    plg = tuple(float(v) for v in _median_rows(PL[m]))
    alg = tuple(float(v) for v in _median_rows(A[m]))
    shifted = []                                   # 4-neighbour shifted copies, for the shared boundary
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        s = np.full(gm.shape, -1, np.int32)
        ys0, ys1 = max(0, dy), gm.shape[0] + min(0, dy)
        xs0, xs1 = max(0, dx), gm.shape[1] + min(0, dx)
        s[ys0:ys1, xs0:xs1] = gm[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        shifted.append(s)
    for j in np.argsort(-cnt):
        n = int(nb[j])
        share = float(cnt[j] / tot)
        is_bg = n in bg_ids
        if (is_bg and not include_bg) or share < nb_min_share * (1.0 if is_bg else (obj_tot / tot)):
            continue
        band = ring & (gm == n)
        bnd = np.zeros_like(m)
        for s in shifted:
            bnd |= m & (s == n)
        pb = _median_rows(P[band])
        sb = _median_rows(S[band])
        vp = np.log(np.clip(pg, EPS, None)) - np.log(np.clip(pb, EPS, None))
        vs = np.log(np.clip(sg, EPS, None)) - np.log(np.clip(sb, EPS, None))
        ev.pairs.append(PairEvidence(
            nb=n, share=share, share_obj=float(cnt[j] / obj_tot) if not is_bg else 0.0, boundary=int(bnd.sum()),
            rho_photo=float(_lum(pg) / max(_lum(pb), EPS)), rho_shade=float(_lum(sg) / max(_lum(sb), EPS)),
            photo_g=plg, photo_b=tuple(float(v) for v in _median_rows(PL[band])),
            alb_g=alg, alb_b=tuple(float(v) for v in _median_rows(A[band])),
            chrom_photo=tuple(float(v) for v in vp - vp.mean()), chrom_shade=tuple(float(v) for v in vs - vs.mean()),
            e_lum=float(np.median(GL[bnd])) if bnd.any() else 0.0,
            e_chr=float(np.median(GC[bnd])) if bnd.any() else 0.0))
    return ev


def hue_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def norm_ab(lab, L_ref: float = 50.0) -> np.ndarray:
    """Lightness-normalised (a, b) (grouping.normalise_lab): what one material keeps under
    different light."""
    L, a, b = (float(v) for v in lab[:3])
    k = (L_ref + 16.0) / max(L + 16.0, 8.0)
    return np.array([a * k, b * k], np.float64)


def cast(lab_a, lab_b) -> float:
    """Distance of the body colours of two CIELAB colours (lightness-normalised (a, b))."""
    return float(np.linalg.norm(norm_ab(lab_a) - norm_ab(lab_b)))


def de2000(lab_a, lab_b) -> float:
    return float(imageio.delta_e(np.asarray(lab_a, np.float32)[None], np.asarray(lab_b, np.float32)[None])[0])


# ---------------------------------------------------------------------- the three tests

def shadow_test(p: PairEvidence, prm: JunkParams = DEFAULT) -> tuple[bool, str]:
    """Test 1 on one (candidate, neighbour) pair: ``(passes, reason)``. Only a candidate darker
    than its neighbour is a shadow of it."""
    rho = p.rho_photo
    if rho > prm.rho_dark:
        return False, f"photo step too small (rho {rho:.2f})"
    se = p.shade_explains
    if se < prm.shade_explains:
        return False, f"shading carries {se:.2f} of the step (< {prm.shade_explains})"
    cg = norm_ab(p.photo_g)
    cb = norm_ab(p.photo_b)
    Cg, Cb = float(np.hypot(*cg)), float(np.hypot(*cb))
    if Cb >= prm.lost_hue_c and (Cg < prm.neutral_c or Cg < prm.keep_c * Cb):
        return False, f"hue lost (C {Cb:.0f} -> {Cg:.0f}): black part or deep shadow, undecidable"
    same_hue = Cg < prm.neutral_c or Cb < prm.neutral_c or \
        hue_delta(math.degrees(math.atan2(cg[1], cg[0])), math.degrees(math.atan2(cb[1], cb[0]))) <= prm.hue_tol
    duller = Cg <= Cb + prm.chroma_margin
    cast_ok = float(np.linalg.norm(np.asarray(p.chrom_photo) - np.asarray(p.chrom_shade))) <= prm.cast_tol
    if (same_hue and duller) or cast_ok:
        why = "duller, same hue" if (same_hue and duller) else "cast of the light"
        return True, f"shadow: rho {rho:.2f}, shading carries {se:.2f}, {why} (C {Cb:.0f} -> {Cg:.0f})"
    return False, f"chromaticity moved unlike light (C {Cb:.0f} -> {Cg:.0f}, same hue {same_hue})"


def rim_test(alb_g, alb_part, prm: JunkParams = DEFAULT) -> tuple[bool, str]:
    """Test 0's colour part: the candidate's median albedo ``alb_g`` is the part's (``alb_part``,
    its group's albedo) in shadow or mixed with the dark around it: no lighter than the part
    (+ ``rim_l_margin``), no more chromatic (+ ``rim_c_margin``) and, for a coloured part, still
    coloured in the part's hue (within ``rim_hue``; a neutral band beside a red part is another
    material, a black gap or a seal)."""
    Lg, ag, bg_ = (float(v) for v in alb_g[:3])
    Lp, ap, bp = (float(v) for v in alb_part[:3])
    Cg, Cp = math.hypot(ag, bg_), math.hypot(ap, bp)
    if Lg > Lp + prm.rim_l_margin:
        return False, f"lighter than the part (L {Lp:.0f} -> {Lg:.0f})"
    if Cg > Cp + prm.rim_c_margin:
        return False, f"more chromatic than the part (C {Cp:.0f} -> {Cg:.0f})"
    if Cp >= RIM_HUE_C:
        if Cg < RIM_HUE_C:
            return False, f"lost the part's colour (C {Cp:.0f} -> {Cg:.0f})"
        dh = hue_delta(math.degrees(math.atan2(bg_, ag)), math.degrees(math.atan2(bp, ap)))
        if dh > prm.rim_hue:
            return False, f"another hue ({dh:.0f} deg)"
    return True, f"the part's rim (L {Lp:.0f} -> {Lg:.0f}, C {Cp:.0f} -> {Cg:.0f})"


def gradient_test(p: PairEvidence, prm: JunkParams = DEFAULT) -> tuple[bool, str]:
    """Test 2 on one pair."""
    c = cast(p.alb_g, p.alb_b)
    dp = de2000(p.photo_g, p.photo_b)
    if p.boundary < 4:
        return False, "no shared boundary"
    if p.e_lum <= prm.edge_lum and p.e_chr <= prm.edge_chr and c <= prm.grad_cast and dp <= prm.grad_photo_de:
        return True, f"no photo edge on the seam (|dlogY| {p.e_lum:.2f}, |dlogC| {p.e_chr:.2f}), cast {c:.1f}, photo dE {dp:.1f}"
    return False, f"edge on the seam (|dlogY| {p.e_lum:.2f}, |dlogC| {p.e_chr:.2f}) or cast {c:.1f} or photo dE {dp:.1f}"


def significance_test(share: float, pairs: Sequence[PairEvidence], prm: JunkParams = DEFAULT
                      ) -> tuple[Optional[PairEvidence], str]:
    """Test 3: the closest-coloured touching group, when the candidate (a group of ``share``
    of the object) is invisible against it; ``(None, reason)`` otherwise."""
    if share >= prm.sig_frac:
        return None, f"not insignificant ({100 * share:.2f} % of the object)"
    best = None
    for p in pairs:
        d = de2000(p.alb_g, p.alb_b)
        if best is None or d < best[0]:
            best = (d, p)
    if best is None:
        return None, "no touching object group"
    d, p = best
    c = cast(p.alb_g, p.alb_b)
    dp = de2000(p.photo_g, p.photo_b)
    if c <= prm.sig_cast and dp <= prm.sig_photo_de:
        return p, f"insignificant ({100 * share:.3f} %), albedo dE {d:.1f} cast {c:.1f}, photo dE {dp:.1f}"
    return None, f"visible or distinct (cast {c:.1f}, photo dE {dp:.1f})"


def _exempt(ev: Evidence, g: ColorGroup, prm: JunkParams, part_groups: set[int], matte: bool = True) -> Optional[str]:
    """Why a candidate region is never moved, or None. With ``matte`` False the matte's
    exemption (a group mostly off the object) is left out: the part-rim rule's evidence is the
    part's own mask, and the matte, a few px tighter than SAM's, often calls the rim backdrop."""
    hit = [s for s in ev.sources if s in prm.exempt_sources]
    if hit:
        return f"exempt source {','.join(hit)}"
    if ev.island_share >= prm.island_exempt:
        return f"decal island ({ev.island_share:.2f})"
    if matte and ev.fg_share < prm.min_fg:
        return f"off the object (matte share {ev.fg_share:.2f})"
    if g.id in part_groups:
        return "detected part"
    return None


def _inside_share(labels: np.ndarray, rid: int, bbox, masks: list[tuple[int, np.ndarray]], dil: int) -> dict[int, float]:
    """part group id -> share of region ``rid``'s pixels under that part's (dilated) masks."""
    x0, y0, x1, y1 = bbox
    H, W = labels.shape
    x0, y0, x1, y1 = max(0, x0 - dil), max(0, y0 - dil), min(W, x1 + dil), min(H, y1 + dil)
    m = labels[y0:y1, x0:x1] == rid
    n = int(m.sum())
    out: dict[int, float] = {}
    if n == 0:
        return out
    k = np.ones((2 * dil + 1, 2 * dil + 1), np.uint8) if dil > 0 else None
    for gid, pm in masks:
        sub = pm[y0:y1, x0:x1]
        if not sub.any():
            continue
        if k is not None:
            sub = cv2.dilate(sub.astype(np.uint8), k).astype(bool)
        s = float((sub & m).sum()) / n
        if s > out.get(gid, 0.0):
            out[gid] = s
    return out


# ---------------------------------------------------------------------- the pruning

#: Region sources that are lettering: never crumbs (rule 4), whatever their pieces.
LETTERING = ("text", "named")


def _crumbs(mask: np.ndarray, prm: JunkParams) -> bool:
    """True when ``mask`` (bool HxW) is only crumbs: at least ``crumb_pieces`` 8-connected
    pieces, none larger than ``crumb_piece`` px."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return False
    sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    return n - 1 >= prm.crumb_pieces and int(st[1:, cv2.CC_STAT_AREA].max()) <= prm.crumb_piece


@dataclass
class Pruned:
    """The pruned grouping (label map unchanged) and what the pruning did. ``log`` records every
    move; ``kept``, ``to_part`` and ``guarded`` are filled only with ``diagnostics`` (the
    experiments read them, the product does not)."""
    regions: list
    groups: list
    group_map: np.ndarray
    islands: np.ndarray
    protect: np.ndarray
    log: list = field(default_factory=list)        # every move: group, region, rule, into, why, evidence
    kept: list = field(default_factory=list)       # every candidate examined and kept, with the reason
    to_part: list = field(default_factory=list)    # moves rule 2 sent into a part group
    guarded: list = field(default_factory=list)    # part groups rule 1 exempted


#: The flags a user sets on a group (``pipeline.USER_FLAG_KINDS``), as ``prune_junk`` reads them.
USER_FLAG_KINDS = ("locked", "is_background")


def user_votes(user_flags: Optional[Mapping]) -> dict[str, dict[int, bool]]:
    """``{"locked": {region id: bool}, "is_background": {...}}`` from the user's flags as the
    pipeline stores them (region ids as strings or ints; anything unreadable is skipped)."""
    out: dict[str, dict[int, bool]] = {k: {} for k in USER_FLAG_KINDS}
    for k in USER_FLAG_KINDS:
        votes = (user_flags or {}).get(k) if isinstance(user_flags, Mapping) else None
        if not isinstance(votes, Mapping):
            continue
        for rid, v in votes.items():
            try:
                out[k][int(rid)] = bool(v)
            except (TypeError, ValueError):
                continue
    return out


def vote_agrees(votes: dict[str, dict[int, bool]], rid: int, members: Collection[int], area: Mapping[int, int],
                host: ColorGroup) -> bool:
    """True when region ``rid``, joining the regions ``members`` of group ``host``, keeps every
    flag it carries a user vote for: the host then holds, for each such flag, the area-weighted
    majority of its (and ``rid``'s) votes when more than half of its area voted (a tie or less
    keeps the host's own flag), exactly as the regroup re-applies the user's choices
    (``pipeline.apply_user_flags``), and that value is ``rid``'s vote."""
    ids = [int(x) for x in members if int(x) != rid] + [rid]
    total = sum(max(int(area.get(x, 1)), 1) for x in ids)
    for k in USER_FLAG_KINDS:
        v = votes.get(k, {}).get(rid)
        if v is None:
            continue
        vk = votes[k]
        yes = sum(max(int(area.get(x, 1)), 1) for x in ids if vk.get(x) is True)
        no = sum(max(int(area.get(x, 1)), 1) for x in ids if vk.get(x) is False)
        flag = (yes > no) if (2 * (yes + no) > total and yes != no) else bool(getattr(host, k))
        if flag != v:
            return False
    return True


def part_group_masks(labels: np.ndarray, regions: Sequence[Region], masks: Sequence[np.ndarray]
                     ) -> list[tuple[int, np.ndarray]]:
    """Each part instance's SAM mask paired with the id of the group holding the part region
    that lies most under it (masks under no part region are dropped)."""
    part_ids = {r.id for r in regions if r.part_kind}
    g_of = {r.id: r.group_id for r in regions}
    out = []
    for m in masks:
        m = np.asarray(m, bool)
        if m.shape != labels.shape or not m.any():
            continue
        ids, cnt = np.unique(labels[m], return_counts=True)
        best = [(c, i) for i, c in zip(ids.tolist(), cnt.tolist()) if i in part_ids]
        if best:
            out.append((int(g_of[max(best)[1]]), m))
    return out


def prune_junk(photo_u8: np.ndarray, albedo_lin: np.ndarray, shading_lin: Optional[np.ndarray], labels: np.ndarray,
               regions: Sequence[Region], groups: Sequence[ColorGroup], group_map: np.ndarray,
               islands: Optional[np.ndarray], fg: Optional[np.ndarray] = None,
               part_masks: Sequence[np.ndarray] = (), params: JunkParams = DEFAULT,
               ctx: Optional[Context] = None, user_flags: Optional[Mapping] = None,
               diagnostics: bool = False) -> Pruned:
    """Merge the junk groups of a refined grouping into their neighbours (module docstring).
    ``fg`` is the foreground matte (float or bool HxW; > 0.5 is the object), ``part_masks``
    the detected parts' SAM masks (bool HxW, for rules 2 and 2b), ``user_flags`` the user's lock
    and background votes per region (``{"locked": {region id: bool}, "is_background": {...}}``,
    as ``pipeline.USER_FLAGS_FILE`` stores them): a voted region only joins a host that ends
    with its vote (:func:`vote_agrees`). Returns the rebuilt records: ids
    0..G-1 by area, flags and custom names kept (a merged group's lock goes with it),
    background re-marked from the regions' backdrop decisions (a part group and a lone colour
    group are never background), the panel view annotated (:func:`grouping.annotate_groups`),
    protect mask recomputed, merged regions out of the islands; the label map is unchanged.
    With ``diagnostics`` the record also lists what was kept and why."""
    from . import refine
    prm = params
    labels = np.ascontiguousarray(labels, np.int32)
    ctx = ctx or build_context(photo_u8, albedo_lin, shading_lin, fg)
    regions = list(regions)
    groups = list(groups)
    sources = [""] * (int(labels.max()) + 1)
    for r in regions:
        sources[r.id] = r.source
    gm = np.ascontiguousarray(group_map, np.int32).copy()
    by_id = {g.id: g for g in groups}
    area = {g.id: int(g.area) for g in groups}
    bg_ids = {g.id for g in groups if g.is_background}
    part_set = {r.id for r in regions if r.part_kind}
    kind_gids = {g.id for g in groups if part_set & set(g.region_ids)}
    masks = part_group_masks(labels, regions, part_masks) if part_masks else []
    assign = {r.id: int(r.group_id) for r in regions}
    r_area = {r.id: int(r.area) for r in regions}
    r_bbox = {r.id: r.bbox for r in regions}
    votes = user_votes(user_flags)
    voted = {rid for vk in votes.values() for rid in vk}
    log: list[dict] = []
    kept: list[dict] = []
    to_part: list[dict] = []
    guarded: list[dict] = []
    isl = np.zeros(labels.shape, bool) if islands is None else np.asarray(islands, bool)

    def keep(g: ColorGroup, rid: int, reason: str) -> None:
        if diagnostics:
            kept.append({"group": int(g.id), "region": int(rid), "name": g.name, "area": r_area[rid], "reason": reason})

    def pick(passing):
        # the passing neighbour with the closest body colour at the seam, then the largest share
        return min(passing, key=lambda t: (cast(t[0].alb_g, t[0].alb_b), -t[0].share_obj)) if passing else None

    def move(rid: int, gid: int, host: int) -> None:
        assign[rid] = host
        gm[labels == rid] = host
        area[host] = area.get(host, 0) + r_area[rid]
        area[gid] -= r_area[rid]

    def keeps_vote(rid: int, host: int) -> bool:
        # a region the user voted on joins only a host that ends with its vote
        if rid not in voted:
            return True
        return vote_agrees(votes, rid, [x for x, a in assign.items() if a == host], r_area, by_id[host])

    cands = sorted((g for g in groups if not g.is_background and g.area < prm.max_frac * ctx.object_px),
                   key=lambda g: (g.area, g.id))
    for g in cands:
        if g.id in kind_gids and diagnostics:
            guarded.append({"group": int(g.id), "name": g.name, "area": int(g.area)})
        members = sorted((rid for rid, a in assign.items() if a == g.id), key=lambda i: (-r_area[i], i))
        if g.id not in kind_gids and area[g.id] <= prm.crumb_group and members and \
                not any(sources[rid] in LETTERING for rid in members) and _crumbs(gm == g.id, prm):
            # rule 4: a group of crumbs goes to the group owning most of its ring
            gev = evidence(ctx, gm, g.id, labels, sources, isl, bg_ids, nb_min_share=prm.nb_min_share)
            hosts = [p for p in gev.pairs if p.nb != g.id and area.get(p.nb, 0) > area[g.id] and p.nb not in kind_gids]
            if hosts:
                host = max(hosts, key=lambda p: (p.share_obj, p.share))
                if all(keeps_vote(rid, host.nb) for rid in members):
                    for rid in members:
                        move(rid, g.id, host.nb)
                        log.append({"group": int(g.id), "region": int(rid), "name": g.name, "group_area": int(g.area),
                                    "area": int(r_area[rid]), "object_share": round(gev.object_share, 5),
                                    "locked": bool(g.locked), "sources": list(gev.sources), "rule": "crumbs",
                                    "into": int(host.nb), "into_name": by_id[host.nb].name, "into_part": False,
                                    "why": "a group of crumbs", "evidence": {"share_obj": round(host.share_obj, 3)}})
                    continue
        decided: list[tuple] = []
        undecided = 0
        for rid in members:
            ev = evidence(ctx, gm, g.id, labels, sources, isl, bg_ids, nb_min_share=prm.nb_min_share, region=rid)
            why = _exempt(ev, g, prm, kind_gids, matte=False)
            if why:
                undecided += 1
                keep(g, rid, why)
                continue
            decision = None
            reasons: list[str] = []
            if prm.part_rim and masks:
                # rule 2b: the shadowed rim of a detected part, just outside its SAM mask
                near = _inside_share(labels, rid, r_bbox[rid], masks, prm.rim_px)
                for k, share in sorted(near.items(), key=lambda t: (-t[1], t[0])):
                    if share < prm.rim_inside or k == g.id or area.get(k, 0) <= area[g.id]:
                        continue
                    pk = next((p for p in ev.pairs if p.nb == k), None)
                    if pk is None or pk.share_obj < prm.rim_share:
                        continue
                    ok, w = rim_test(pk.alb_g, by_id[k].albedo_lab, prm)
                    reasons.append(f"g{k} part rim: {w}")
                    if ok:
                        decision = ("part_rim", pk, w)
                        break
            if decision is None:
                off = _exempt(ev, g, prm, kind_gids)       # the matte's exemption, for tests 1-3
                if off:
                    undecided += 1
                    keep(g, rid, "; ".join(reasons + [off]))
                    continue
            inside = _inside_share(labels, rid, r_bbox[rid], masks, prm.part_dilate) if masks else {}
            inside_kinds = {k for k, s in inside.items() if s >= prm.inside_part}
            # a part group hosts only the candidates inside its part (rule 2)
            pairs = [p for p in ev.pairs if p.nb != g.id and area.get(p.nb, 0) > area[g.id]
                     and (p.nb not in kind_gids or p.nb in inside_kinds)]
            island = ev.island_share >= prm.island_exempt
            if decision is None and prm.shadow:
                passing = []
                for p in pairs:
                    if p.share_obj < prm.host_share:
                        continue
                    ok, w = shadow_test(p, prm)
                    reasons.append(f"g{p.nb} shadow: {w}")
                    if ok:
                        passing.append((p, w))
                best = pick(passing)
                if best:
                    decision = ("shadow", best[0], best[1])
            if decision is None and prm.gradient and not island:
                passing = []
                for p in pairs:
                    if p.share_obj < prm.grad_share:
                        continue
                    ok, w = gradient_test(p, prm)
                    reasons.append(f"g{p.nb} gradient: {w}")
                    if ok:
                        passing.append((p, w))
                best = pick(passing)
                if best:
                    decision = ("gradient", best[0], best[1])
            if decision is None and prm.significance and not g.locked and not island:
                p, w = significance_test(area[g.id] / ctx.object_px, pairs, prm)
                reasons.append(f"significance: {w}")
                if p is not None:
                    decision = ("significance", p, w)
            if decision is None:
                undecided += 1
                keep(g, rid, "; ".join(reasons) or "no larger object neighbour")
                continue
            host_pair = decision[1]
            retarget = None
            if decision[0] != "part_rim" and inside_kinds and host_pair.nb not in inside_kinds:
                # rule 2: inside a part -> the part's group, when it touches the region and is larger
                touching = [p for p in ev.pairs if p.nb in inside_kinds and area.get(p.nb, 0) > area[g.id]]
                if touching:
                    retarget = max(touching, key=lambda p: (inside[p.nb], p.share))
            if not keeps_vote(rid, retarget.nb if retarget is not None else host_pair.nb):
                undecided += 1
                keep(g, rid, f"{decision[0]}: the user's lock or background choice is not the host's")
                continue
            decided.append((rid, decision[0], host_pair, decision[2], ev, retarget))
        # the physical tests move single regions; the colour tests only dissolve a whole group
        for rid, rule, p, w, ev, retarget in decided:
            if rule not in ("shadow", "part_rim") and undecided:
                keep(g, rid, f"{rule} would move it but the rest of the group stays")
                continue
            host = retarget.nb if retarget is not None else p.nb
            move(rid, g.id, host)
            rec = {"group": int(g.id), "region": int(rid), "name": g.name, "group_area": int(g.area),
                   "area": int(r_area[rid]), "object_share": round(ev.object_share, 5), "locked": bool(g.locked),
                   "sources": list(ev.sources), "rule": rule, "into": int(host), "into_name": by_id[host].name,
                   "into_part": bool(host in kind_gids), "why": w, "fg_share": round(ev.fg_share, 3),
                   "evidence": {"rho_photo": round(p.rho_photo, 3), "rho_shade": round(p.rho_shade, 3),
                                "shade_explains": round(p.shade_explains, 3), "share_obj": round(p.share_obj, 3),
                                "e_lum": round(p.e_lum, 3), "e_chr": round(p.e_chr, 3)}}
            if retarget is not None:
                rec["test_host"] = int(p.nb)
                if diagnostics:
                    to_part.append({"region": int(rid), "from_group": int(g.id), "test_host": int(p.nb),
                                    "into_part": int(host), "into_name": by_id[host].name, "rule": rule})
            log.append(rec)
    if prm.backdrop_crumbs:
        for g in sorted((g for g in groups if g.is_background and g.area < prm.max_frac * ctx.object_px),
                        key=lambda g: (g.area, g.id)):
            ev = evidence(ctx, gm, g.id, labels, sources, isl, bg_ids, nb_min_share=prm.nb_min_share, include_bg=True)
            pairs = [p for p in ev.pairs if p.nb in bg_ids and area.get(p.nb, 0) > area[g.id]]
            if not pairs:
                continue
            p = min(pairs, key=lambda p: de2000(p.alb_g, p.alb_b))
            for rid in [rid for rid, a in assign.items() if a == g.id]:
                if not keeps_vote(rid, p.nb):
                    continue
                move(rid, g.id, p.nb)
                log.append({"group": int(g.id), "region": int(rid), "name": g.name, "group_area": int(g.area),
                            "area": int(r_area[rid]), "object_share": round(ev.object_share, 5), "locked": bool(g.locked),
                            "sources": list(ev.sources), "rule": "backdrop_crumbs", "into": int(p.nb),
                            "into_name": by_id[p.nb].name, "into_part": False, "why": "tiny background group",
                            "evidence": {}})
    if not log:
        grouping.annotate_groups(groups, regions, np.ascontiguousarray(group_map, np.int32))
        protect = refine.protect_mask(photo_u8, labels, group_map, groups, islands)
        return Pruned(regions, groups, np.ascontiguousarray(group_map, np.int32), isl.copy(), protect, log, kept,
                      to_part, guarded)
    carry = {g.id: {"name": grouping._keep_name(g), "locked": g.locked, "is_background": g.is_background} for g in groups}
    new_regions, new_groups, new_gm = grouping._finalize(regions, labels, assign, carry)
    grouping._mark_background(new_groups, new_gm, new_regions)
    grouping._check_state(new_regions, new_groups)
    grouping.annotate_groups(new_groups, new_regions, new_gm)
    moved = [m["region"] for m in log]
    new_isl = isl & ~np.isin(labels, moved)
    protect = refine.protect_mask(photo_u8, labels, new_gm, new_groups, new_isl)
    return Pruned(new_regions, new_groups, new_gm, new_isl, protect, log, kept, to_part, guarded)
