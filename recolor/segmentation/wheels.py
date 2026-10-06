"""Split a SAM wheel mask into its tyre and its rim.

SAM 2.1 returns tyre + rim (+ spokes) as one mask for every box or point prompt on a wheel
seen from the side (the BMW's wheels: score 0.97 for the box, for points on the tyre and for
the box with negative points on the tyre alike), and the two are 5-10 dE apart in albedo,
so no colour rule separates them. The wheel's outer contour is an ellipse (a circle in
perspective); the rim's lip is another ellipse inside it, not concentric in the normalised
radius (the tyre looks thicker on one side), so a radius threshold does not work either.

:func:`split_wheel` fits the outer ellipse with RANSAC on the filled mask's contour (an
occluder like a muffler cuts into it), casts N_RAYS rays from its centre and takes on each
ray the outermost strong edge of the photo's lightness (plus half the albedo's) inside the
tyre band (normalised radius RHO_LO..RHO_HI); an ellipse fitted to those points with RANSAC
is the rim's lip: rim = the mask inside it, tyre = the rest. This is how the reference
partitions built their rims (ellipses through hand-picked edge points), done automatically.
A wheel whose contour is not an ellipse or that gives no rim ellipse is left alone (None).

Two cases the first version left alone: a swingarm, a chain guard and a fender cutting deep
notches into the outline (the Ducati's rear wheel: 31 % of its contour on the ellipse, below
the 35 % the fit asked for, although the ellipse was supported all the way round but for the
swingarm) now pass on the ellipse's angular support (MIN_OUTER_SUPPORT of 5-degree sectors
holding an inlier, with MIN_OUTER_SHARE of the contour); a black rim in a black tyre (the
BMW's front wheel), whose lip is a weaker edge than the silver disc inside it on most rays, is
found by the lip fallback when the first fit fails: every edge peak of each ray in the outer
band (LIP_RHO) is a candidate, and the ellipse that the most rays support is the lip.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage

N_RAYS = 144
RHO_LO, RHO_HI = 0.55, 0.95
#: Share of the mask's outer contour that must lie on the fitted outer ellipse ...
MIN_OUTER_INLIERS = 0.35
#: ... or at least MIN_OUTER_SHARE of it with an inlier in MIN_OUTER_SUPPORT of the ellipse's
#: 5-degree sectors (an outline notched deep by a swingarm is still an ellipse all the way round).
MIN_OUTER_SHARE = 0.25
MIN_OUTER_SUPPORT = 0.6
OUTER_SECTORS = 72
#: The lip fallback: every edge peak of at least LIP_REL of the ray's strongest edge in the band
#: LIP_RHO is a candidate; the rim ellipse (axes LIP_AXES of the outer ones, centred within
#: LIP_CENTRE of its radius) supported by the most rays, at least MIN_INLIER_FRAC of all rays and
#: LIP_SUPPORT of the rays holding a candidate.
LIP_RHO = (0.7, 0.95)
LIP_REL = 0.25
LIP_AXES = (0.55, 0.97)
LIP_CENTRE = 0.12
LIP_SUPPORT = 0.6
LIP_ITERS = 600
#: Rim-lip points within this share of the outer radius count as inliers of the rim ellipse ...
INLIER_TOL = 0.025
#: ... and at least this share of the rays must be inliers.
MIN_INLIER_FRAC = 0.33
RANSAC_ITERS = 300
#: Each part must hold at least this share of the wheel mask.
MIN_PART_SHARE = 0.12
MIN_WHEEL_PX = 2000


def _ellipse_mask(shape: tuple[int, int], e) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    (cx, cy), (a, b), ang = e
    cv2.ellipse(m, ((float(cx), float(cy)), (float(a), float(b)), float(ang)), 1, -1)
    return m.astype(bool)


def _rho(xs: np.ndarray, ys: np.ndarray, e) -> np.ndarray:
    """Normalised radius of the points (1 on the ellipse)."""
    (cx, cy), (a, b), ang = e
    th = math.radians(ang)
    dx, dy = xs - cx, ys - cy
    u = dx * math.cos(th) + dy * math.sin(th)
    v = -dx * math.sin(th) + dy * math.cos(th)
    return np.sqrt((u / max(a / 2, 1e-6)) ** 2 + (v / max(b / 2, 1e-6)) ** 2)


def _angular_support(points: np.ndarray, e) -> float:
    """Share of the ellipse's OUTER_SECTORS angular sectors holding at least one of ``points``."""
    if len(points) == 0:
        return 0.0
    (cx, cy), (a, b), ang = e
    th = math.radians(ang)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    u = (dx * math.cos(th) + dy * math.sin(th)) / max(a / 2, 1e-6)
    v = (-dx * math.sin(th) + dy * math.cos(th)) / max(b / 2, 1e-6)
    sector = (np.degrees(np.arctan2(v, u)) % 360.0 // (360.0 / OUTER_SECTORS)).astype(int)
    return float(len(np.unique(sector))) / OUTER_SECTORS


def _ray(k: int, outer, ts: np.ndarray, wc: np.ndarray, L: np.ndarray, aL: np.ndarray):
    """Ray ``k`` of N_RAYS from the outer ellipse's centre at the normalised radii ``ts``: its
    points and its edge strength (0 outside the mask), or None when it runs mostly outside the
    mask or has no edge."""
    (cx, cy), (A, B), ang = outer
    th = math.radians(ang)
    phi = 2 * math.pi * k / N_RAYS
    uu, vv = (A / 2) * math.cos(phi), (B / 2) * math.sin(phi)
    dx = uu * math.cos(th) - vv * math.sin(th)
    dy = uu * math.sin(th) + vv * math.cos(th)
    px = cx + ts * dx
    py = cy + ts * dy
    ix = np.clip(np.round(px).astype(int), 0, wc.shape[1] - 1)
    iy = np.clip(np.round(py).astype(int), 0, wc.shape[0] - 1)
    inside = wc[iy, ix]
    if inside.mean() < 0.6:
        return None
    g = np.abs(np.gradient(L[iy, ix])) + 0.5 * np.abs(np.gradient(aL[iy, ix]))
    g[~inside] = 0.0
    if float(g.max()) < 2.0:
        return None
    return px, py, g


def _lip_fallback(outer, wc: np.ndarray, L: np.ndarray, aL: np.ndarray, rng, debug: Optional[dict] = None):
    """The rim's lip when the outermost-strong-edge fit found none (a black rim in a black tyre,
    whose lip is weaker than the disc inside it on most rays): every edge peak of each ray in the
    band LIP_RHO of at least LIP_REL of the ray's strongest edge is a candidate, and RANSAC keeps
    the ellipse (axes LIP_AXES of the outer, centred within LIP_CENTRE of its radius) that the most
    rays support. Returns the ellipse, or None when too few rays support any."""
    (cx, cy), (A, B), _ang = outer
    ts = np.linspace(LIP_RHO[1], LIP_RHO[0], 61)
    pts, ray = [], []
    for k in range(N_RAYS):
        got = _ray(k, outer, ts, wc, L, aL)
        if got is None:
            continue
        px, py, g = got
        thr = LIP_REL * float(g.max())
        for j in range(1, len(ts) - 1):
            if g[j] >= thr and g[j] >= g[j - 1] and g[j] >= g[j + 1]:
                pts.append((px[j], py[j]))
                ray.append(k)
    rays = np.unique(np.asarray(ray, np.int64))
    if debug is not None:
        debug["lip_rays"] = int(len(rays))
    if len(rays) < max(6, MIN_INLIER_FRAC * N_RAYS):
        return None
    P = np.asarray(pts, np.float32)
    ray = np.asarray(ray, np.int64)
    by_ray = {int(r): np.flatnonzero(ray == r) for r in rays.tolist()}
    R = 0.5 * (A + B) / 2.0
    lo, hi = LIP_AXES
    best, best_n, best_in = None, -1, None
    for _ in range(LIP_ITERS):
        pick = rng.choice(rays, 6, replace=False)
        idx = [int(rng.choice(by_ray[int(r)])) for r in pick]
        try:
            e = cv2.fitEllipse(P[idx].reshape(-1, 1, 2))
        except cv2.error:
            continue
        (ex, ey), (ea, eb), _ = e
        if not (lo * A < ea < hi * A and lo * B < eb < hi * B) and not (lo * B < ea < hi * B and lo * A < eb < hi * A):
            continue
        if math.hypot(ex - cx, ey - cy) > LIP_CENTRE * R:
            continue
        r = _rho(P[:, 0], P[:, 1], e)
        inl = np.abs(r - 1.0) * (0.25 * (ea + eb)) < INLIER_TOL * R
        n = len(np.unique(ray[inl]))
        if n > best_n:
            best, best_n, best_in = e, n, inl
    if debug is not None:
        debug["lip_support"] = int(max(best_n, 0))
    if best is None or best_n < MIN_INLIER_FRAC * N_RAYS or best_n < LIP_SUPPORT * len(rays):
        return None
    try:
        return cv2.fitEllipse(P[best_in].reshape(-1, 1, 2))
    except cv2.error:
        return best


def split_wheel(wheel: np.ndarray, image_rgb_u8: np.ndarray, albedo_lab: np.ndarray, seed: int = 0,
                debug: Optional[dict] = None) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """``(tyre, rim)`` full-size bool masks of the wheel mask ``wheel`` (bool HxW), or None
    when it is not an elliptical wheel or no rim ellipse is found. Deterministic (``seed``).
    ``debug`` (a dict) receives the fit statistics. The outer ellipse passes on the share of the
    contour on it (MIN_OUTER_INLIERS) or on its angular support (MIN_OUTER_SUPPORT, with
    MIN_OUTER_SHARE of the contour: an outline notched deep by a swingarm); the lip is the
    outermost strong edge on each ray, or, when that finds no lip, the fallback that lets every
    edge peak of the outer band compete (:func:`_lip_fallback`)."""
    h, w = wheel.shape
    ys, xs = np.nonzero(wheel)
    if len(ys) < MIN_WHEEL_PX:
        return None
    pad = 4
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    wc = wheel[y0:y1, x0:x1]
    filled = ndimage.binary_fill_holes(wc)
    cnts, _ = cv2.findContours(filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c.reshape(-1, 2) for c in cnts if len(c) >= 5]
    if not cnts:
        return None
    contour = np.concatenate(cnts).astype(np.float32)
    if len(contour) < 40:
        return None
    rng = np.random.default_rng(seed)
    size = 0.5 * float(max(wc.shape))
    best_o, best_oin = None, None
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(len(contour), 6, replace=False)
        try:
            e = cv2.fitEllipse(contour[idx].reshape(-1, 1, 2))
        except cv2.error:
            continue
        (_ex, _ey), (ea, eb), _a = e
        if not (0.5 * size < 0.5 * max(ea, eb) < 1.3 * size) or min(ea, eb) < 0.3 * max(ea, eb):
            continue
        r = _rho(contour[:, 0], contour[:, 1], e)
        inl = np.abs(r - 1.0) * (0.25 * (ea + eb)) < 0.015 * size + 1.5
        if best_oin is None or inl.sum() > best_oin.sum():
            best_o, best_oin = e, inl
    if best_o is None:
        return None
    try:
        outer = cv2.fitEllipse(contour[best_oin].reshape(-1, 1, 2))
    except cv2.error:
        outer = best_o
    fill_iou = float(best_oin.mean())
    if debug is not None:
        debug["fill_iou"] = fill_iou
    if fill_iou < MIN_OUTER_INLIERS:
        support = _angular_support(contour[best_oin], outer) if fill_iou >= MIN_OUTER_SHARE else 0.0
        if debug is not None:
            debug["support"] = round(support, 3)
        if support < MIN_OUTER_SUPPORT:
            return None
    (cx, cy), (A, B), ang = outer
    L = cv2.cvtColor(np.ascontiguousarray(image_rgb_u8[y0:y1, x0:x1]), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * (100.0 / 255.0)
    L = cv2.GaussianBlur(L, (0, 0), 1.2)
    aL = cv2.GaussianBlur(np.ascontiguousarray(albedo_lab[y0:y1, x0:x1, 0], np.float32), (0, 0), 1.2)
    ts = np.linspace(RHO_HI, RHO_LO, 81)
    pts = []
    for k in range(N_RAYS):
        got = _ray(k, outer, ts, wc, L, aL)
        if got is None:
            continue
        px, py, g = got
        thr = 0.4 * float(g.max())             # the outermost local maximum above 40 % of the ray's strongest edge
        for j in range(1, len(ts) - 1):
            if g[j] >= thr and g[j] >= g[j - 1] and g[j] >= g[j + 1]:
                pts.append((px[j], py[j]))
                break
    if debug is not None:
        debug["edge_points"] = len(pts)
    rim_e = None
    R = 0.5 * (A + B) / 2.0
    if len(pts) >= 0.35 * N_RAYS:
        P = np.asarray(pts, np.float32)
        best, best_in = None, None
        for _ in range(RANSAC_ITERS):
            idx = rng.choice(len(P), 6, replace=False)
            try:
                e = cv2.fitEllipse(P[idx].reshape(-1, 1, 2))
            except cv2.error:
                continue
            (_ex, _ey), (ea, eb), _eang = e
            if not (0.4 * A < ea < 0.97 * A and 0.4 * B < eb < 0.97 * B) and \
                    not (0.4 * B < ea < 0.97 * B and 0.4 * A < eb < 0.97 * A):
                continue
            r = _rho(P[:, 0], P[:, 1], e)
            inl = np.abs(r - 1.0) * (0.25 * (ea + eb)) < INLIER_TOL * R
            if best_in is None or inl.sum() > best_in.sum():
                best, best_in = e, inl
        if debug is not None:
            debug["inliers"] = int(best_in.sum()) if best_in is not None else 0
        if best is not None and best_in.sum() >= MIN_INLIER_FRAC * N_RAYS:
            try:
                rim_e = cv2.fitEllipse(P[best_in].reshape(-1, 1, 2))
            except cv2.error:
                rim_e = best
    if rim_e is None:
        rim_e = _lip_fallback(outer, wc, L, aL, rng, debug)
        if rim_e is None:
            return None
        if debug is not None:
            debug["lip"] = "fallback"
    inner = _ellipse_mask(wc.shape, rim_e)
    rim_c = wc & inner
    tyre_c = wc & ~inner
    if rim_c.sum() < MIN_PART_SHARE * wc.sum() or tyre_c.sum() < MIN_PART_SHARE * wc.sum():
        return None
    tyre = np.zeros((h, w), bool)
    rim = np.zeros((h, w), bool)
    tyre[y0:y1, x0:x1] = tyre_c
    rim[y0:y1, x0:x1] = rim_c
    return tyre, rim
