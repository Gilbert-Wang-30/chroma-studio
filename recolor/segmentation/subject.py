"""The matte keeps one subject: another object of the photo's class that the foreground matte
took in goes to the backdrop.

BiRefNet's matte (:mod:`foreground`) picks the salient object, and an object touching its
silhouette from behind can come with it: in the Torana showroom the red coupe parked behind the
Torana had its door panel inside the matte (matte 0.95), so the backdrop decisions called it
object, the clustering put it into the Torana's orange "Rust" paint, and the one-click repaint of
the Torana painted half of the coupe navy. The partition cannot tell the two apart (the coupe's
paint is the Torana's colour and its panel touches the Torana's body along the roofline), but
SAM can: prompted with the box of the matte's main component it answers with the subject's own
silhouette (the Torana without the coupe), and prompted with a point on a piece of the matte left
outside that silhouette it answers with that piece's own object.

:func:`other_objects` (a step of the regions stage after the detected parts, for the classes
whose photo shows one of them: a car, a motorcycle, a bicycle; a pair of sneakers is two
subjects of one product, and SAM's box answer on a pair covered one shoe) returns the matte
pixels of every such other object: a component of the matte outside the subject's silhouette
(grown by GROW_PX) of at least MIN_SHARE of the subject and with a core (a sliver along the
silhouette, where the matte and SAM disagree by a few px, has none), whose own SAM object covers
it (COVER) and does not reach into the subject (SUBJECT_MAX), and that holds no detected part.
Nothing is returned unless SAM's subject covers SUBJECT_COVER of the matte (a subject mask that
misses a wheel is no silhouette to judge by). Measured on the part reference photos: only the
Torana's coupe (15.6k px, 4 % of the Torana) and the red rope in front of the Corvette (10.5k px)
go; the BMW's licence plate (0.7 % of the bike), the Corvette's bumper corner behind a sign post
(its SAM object reaches into the Corvette) and every crumb stay.

CPU numpy plus the SAM prompts the caller passes in (``prompter``: :meth:`SamMasker.prompt_boxes`,
one image embedding for the box and the points); nothing loads a model.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import cv2
import numpy as np

#: The object classes (:func:`smallparts.object_class`) whose photo shows one subject.
SINGLE_CLASSES = ("car", "motorcycle", "bicycle")
#: SAM's answer to the matte's box is the subject when it scores SAM_MIN, lies SUBJECT_ON on the
#: matte and covers SUBJECT_COVER of it (the most covering such answer is taken).
SAM_MIN = 0.9
SUBJECT_ON = 0.9
SUBJECT_COVER = 0.85
#: The subject's silhouette grown by this many px (the matte and SAM disagree by a few px).
GROW_PX = 5
#: A leftover matte component: at least MIN_SHARE of the subject's area and MIN_PX, holding a
#: disk of CORE_PX radius.
MIN_SHARE = 0.02
MIN_PX = 1500
CORE_PX = 3
#: Its own object (SAM on a point at its deepest pixel, the best answer of at least POINT_MIN):
#: covering COVER of it and at most SUBJECT_MAX of the object inside the subject.
POINT_MIN = 0.85
COVER = 0.8
SUBJECT_MAX = 0.05
#: A component holding more than PART_MAX of a detected part (or part region) is kept.
PART_MAX = 0.2

#: ``prompter(image_u8, jobs) -> candidates`` as :meth:`SamMasker.prompt_boxes` returns them.
Prompter = Callable[[np.ndarray, list[dict]], list[list[dict]]]


def _full(c: dict, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), bool)
    cm = np.asarray(c["mask"], bool)
    y0, x0 = int(c["y0"]), int(c["x0"])
    m[y0:y0 + cm.shape[0], x0:x0 + cm.shape[1]] = cm[: h - y0, : w - x0]
    return m


#: A region cut by the other objects' pixels (:func:`cut_off`) keeps a side below this many px.
CUT_MIN_PX = 50
#: A region at least this much inside the other objects' pixels goes with them whole
#: (:func:`whole_regions`): the rest of it is the seam GROW_PX left along the subject.
WHOLE_SHARE = 0.5


def whole_regions(labels: np.ndarray, mask: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """``mask`` (the other objects' pixels) grown to every region lying at least WHOLE_SHARE
    inside it, less ``keep``: the coupe's door region lay 80 % in the coupe's piece of the matte,
    and its rest, the 5 px seam kept along the Torana's silhouette and a patch of its roof, stayed
    in the Torana's paint (2.1k px painted navy with the Torana)."""
    labels = np.ascontiguousarray(labels, np.int32)
    n = int(labels.max()) + 1
    m = np.asarray(mask, bool)
    inside = np.bincount(labels[m].ravel(), minlength=n)
    total = np.maximum(np.bincount(labels.ravel(), minlength=n), 1)
    ids = np.flatnonzero(inside >= WHOLE_SHARE * total)
    out = m | np.isin(labels, ids) if ids.size else m.copy()
    if keep is not None:
        out &= ~np.asarray(keep, bool)
    return out


def cut_off(labels: np.ndarray, info: list[dict], mask: np.ndarray, albedo_lin: Optional[np.ndarray] = None
            ) -> tuple[np.ndarray, list[dict], int]:
    """Every region with at least CUT_MIN_PX on each side of ``mask`` (the other objects' pixels)
    is cut along it, whatever the shares: the pixels in ``mask`` become a new region (appended
    id, the parent's record, source 'split'), the rest keeps the id (the matte cut's 20 % rule
    would leave the coupe's panel in the region it shares with the Torana). Returns ``(labels,
    info, n_cut)``, ``area`` and ``albedo_lab`` refreshed when ``albedo_lin`` is given."""
    from .labelops import region_areas, region_medians
    labels = np.ascontiguousarray(labels, np.int32)
    n = int(labels.max()) + 1
    m = np.asarray(mask, bool)
    inside = np.bincount(labels[m].ravel(), minlength=n)
    total = np.bincount(labels.ravel(), minlength=n)
    ids = np.flatnonzero((inside >= CUT_MIN_PX) & (total - inside >= CUT_MIN_PX))
    if ids.size == 0:
        return labels.copy(), [dict(d) for d in info], 0
    out = labels.copy()
    new_info = [dict(d) for d in info]
    for r in ids.tolist():
        rid = len(new_info)
        out[m & (labels == r)] = rid
        child = dict(info[r], id=rid, source="split")
        child.pop("exempt", None)
        new_info.append(child)
    if albedo_lin is not None:
        from .. import imageio
        lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
        areas = region_areas(out, len(new_info))
        meds = region_medians(out, lab, len(new_info))
        for d in new_info:
            d["area"] = int(areas[d["id"]])
            d["albedo_lab"] = tuple(float(v) for v in meds[d["id"]])
    return out, new_info, int(ids.size)


def other_objects(image_rgb_u8: np.ndarray, fg: np.ndarray, prompter: Prompter,
                  keep: Sequence[np.ndarray] = (), report: Optional[dict] = None) -> np.ndarray:
    """The matte pixels (bool HxW) of the other objects of a one-subject photo (module
    docstring); all False when there are none or the subject's silhouette is not clear.
    ``keep`` are masks the result never holds (the detected parts and their regions: a part is
    the object by definition). ``report`` (a dict) gets the subject's coverage and one record per
    examined component."""
    h, w = fg.shape
    obj = np.asarray(fg, np.float32) >= 0.5
    none = np.zeros((h, w), bool)
    n, cc, st, _ = cv2.connectedComponentsWithStats(obj.astype(np.uint8), connectivity=8)
    if n < 2:
        return none
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = (int(v) for v in st[k, :4])
    box = [float(x), float(y), float(x + bw), float(y + bh)]
    mx, my = 0.08 * bw + 8, 0.08 * bh + 8
    crop = (int(max(0, x - mx)), int(max(0, y - my)), int(min(w, x + bw + mx)), int(min(h, y + bh + my)))
    res = prompter(image_rgb_u8, [{"crop": crop, "box": box}])
    n_obj = max(int(obj.sum()), 1)
    subject, cover = None, 0.0
    for c in (res[0] if res else []):
        if float(c["score"]) < SAM_MIN:
            continue
        m = _full(c, (h, w))
        on = int((m & obj).sum())
        if on < SUBJECT_ON * max(int(m.sum()), 1):
            continue
        if on / n_obj > cover:
            subject, cover = m, on / n_obj
    if report is not None:
        report["subject_cover"] = round(cover, 3)
    if subject is None or cover < SUBJECT_COVER:
        return none
    grown = cv2.dilate(subject.astype(np.uint8), np.ones((2 * GROW_PX + 1, 2 * GROW_PX + 1), np.uint8)).astype(bool)
    left = obj & ~grown
    n2, cc2, st2, _ = cv2.connectedComponentsWithStats(left.astype(np.uint8), connectivity=8)
    need = max(MIN_PX, MIN_SHARE * float(subject.sum()))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CORE_PX + 1, 2 * CORE_PX + 1))
    keep_m = np.zeros((h, w), bool)
    for m in keep:
        m = np.asarray(m, bool)
        if m.shape == (h, w):
            keep_m |= m
    comps, jobs = [], []
    for i in range(1, n2):
        if st2[i, cv2.CC_STAT_AREA] < need:
            continue
        ci = cc2 == i
        if not cv2.erode(ci.astype(np.uint8), kern).any():
            continue
        if keep_m.any() and float((ci & keep_m).sum()) > PART_MAX * float(ci.sum()):
            if report is not None:
                report.setdefault("components", []).append({"px": int(ci.sum()), "kept": "holds a detected part"})
            continue
        dt = cv2.distanceTransform(ci.astype(np.uint8), cv2.DIST_L2, 3)
        yy, xx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        comps.append(ci)
        jobs.append({"crop": crop, "points": [[int(xx), int(yy)]], "labels": [1]})
    out = none.copy()
    if not jobs:
        return out
    for ci, cands in zip(comps, prompter(image_rgb_u8, jobs)):
        best = max((c for c in cands or [] if float(c["score"]) >= POINT_MIN), key=lambda c: float(c["score"]), default=None)
        rec: dict = {"px": int(ci.sum())}
        if best is None:
            rec["kept"] = "no confident object"
        else:
            m = _full(best, (h, w))
            a = max(int(m.sum()), 1)
            cov = float((m & ci).sum()) / max(int(ci.sum()), 1)
            inside = float((m & subject).sum()) / a
            rec.update(cover=round(cov, 3), in_subject=round(inside, 3))
            if cov >= COVER and inside <= SUBJECT_MAX:
                out |= ci & ~keep_m
                rec["other"] = True
            else:
                rec["kept"] = "the subject's own piece" if inside > SUBJECT_MAX else "not one object"
        if report is not None:
            report.setdefault("components", []).append(rec)
    return out
