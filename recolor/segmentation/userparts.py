"""User parts: a part the user pointed at with SAM 2 (Select part, Find part) made a group of its own.

The automatic pass misses parts (the Ducati's far-side front caliper, seen through its wheel, sat
in the "Brake disc" and "Wheel rims" groups). A user part is carved into the label map with
:func:`carve` and tagged like a detected part (``Region.part_kind`` ``user_<n>``, its label the
name the user gave it or "Part <n>"), so every rule that protects a detected part protects it:
the colour clustering never takes it, it is one group whatever its colour and however small,
never flagged background by the automatic rules (only by the user's own choice), never the paint,
never locked as another material, never pruned as junk, and a regroup gives it back (the regroup
seed carries it as an input region of its own, and the registry in ``user_flags.json`` is the
safety net: :func:`apply_registry`).

The carve (:func:`carve`) never renumbers a region: a region lying wholly inside the part keeps
its id and joins it, the pieces cut off the other regions become one new region (the next free
id), and every other region keeps its id and its pixels. No speck is created: a piece of the
mask under ``min_px`` merges back into the region it lies on (unless that region is inside the
mask whole: it joins whole, its own specks too), a piece a cut region would keep under ``min_px``
next to the part joins the part, and the part's real holes (the gaps of a spring's coils, the
backdrop seen through a caliper) stay with the regions they belong to.

A part instance (a detected or drawn part's regions of one instance) the new part covers almost
whole (:data:`ABSORB_SHARE`) is taken in whole (:func:`absorb_parts`), and a group taken in whole
(a part group or a colour group) that makes up most of the new part is replaced by it (the
pipeline carries its paint and, unnamed, its name over); every group taken in whole is recorded so
Remove gives it back as it was, and a user part taken in whole waits in the registry inside the
new one (``inside``) and comes back when that one is removed. A user-part edit renames no group it
did not make (:func:`keep_names`; a part group's automatic kind name follows its instance count),
and changes no other group's lock or background flag.

After a merge, move or split (:func:`normalize`) a user part's regions follow the group they
are in: merged into another part they become an instance of it (a far caliper merged into the
detected "Brake caliper" makes "Brake calipers", which Split separates again), merged into a
colour group larger than the part they lose the tag (the part is gone, also after a regroup), and
a colour region merged or moved into a user part joins the part. (The studio's Remove part, which
gives the pixels back to the parts they came from, is `pipeline._remove_user_part`.)
Everything here is numpy on the CPU.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from ..types import ColorGroup, Region

#: Region source of the new region a carve makes (the pieces it cut off other regions).
USER_SOURCE = "user"
#: Part kinds of user parts: ``user_1``, ``user_2``, ... (a detected part's kind is a vocabulary key).
USER_KIND_PREFIX = "user_"
#: Pieces below this many pixels are never made: a speck of the mask merges back into its
#: region, and a speck a cut region would keep next to the part joins the part.
MIN_PIECE_PX = 12
#: A part instance (every region of one detected or drawn part's instance, together) a new part
#: covers at least this share of (by pixels) is taken in whole, so no sliver of it stays behind:
#: Find's "spring" over the detected "Shock spring" left a 0.02 % "Shock spring" group, selecting
#: the far caliper again cut the earlier "Far caliper" part down to a 112 px sliver that kept its
#: name and its paint, and a part covering 99.5 % of a two-region user part left its 225 px second
#: region behind as the old part (:func:`absorb_parts`).
ABSORB_SHARE = 0.9
#: A group taken in whole (a part group or a colour group) that makes up at least this share of the
#: new part is *replaced* by it: the new part keeps its paint (and its name when the user gave none).
REPLACE_SHARE = 0.5
_KIND_RE = re.compile(r"^user_(\d+)$")


def is_user_kind(kind: Optional[str]) -> bool:
    """True for the part kind of a user part (``user_<n>``)."""
    return bool(kind) and bool(_KIND_RE.match(str(kind)))


def user_number(kind: Optional[str]) -> Optional[int]:
    """``n`` of a ``user_<n>`` kind, None for anything else."""
    m = _KIND_RE.match(str(kind or ""))
    return int(m.group(1)) if m else None


def next_user_kind(kinds: Iterable[str]) -> tuple[str, int]:
    """The kind and number of a new user part: one more than the largest ``user_<n>`` in ``kinds``
    (the regions' tags and the registry together). The number of a removed part can come back: its
    tags are gone from the regions, the registry and the regroup seed (:func:`sync_seed_tags`)."""
    n = max([user_number(k) or 0 for k in kinds] + [0]) + 1
    return f"{USER_KIND_PREFIX}{n}", n


def default_label(number: int) -> str:
    """The automatic name of user part ``number``: "Part 3"."""
    return f"Part {number}"


# ---------------------------------------------------------------------- the carve

def drop_islands(mask: np.ndarray, min_px: int = MIN_PIECE_PX) -> np.ndarray:
    """``mask`` without its 8-connected pieces smaller than ``min_px`` (a copy; the components are
    labelled on the mask's bounding box only)."""
    m = np.asarray(mask, bool)
    if min_px <= 1 or not m.any():
        return m.copy()
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    k, cc, st, _ = cv2.connectedComponentsWithStats(m[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    small = np.flatnonzero(st[:, cv2.CC_STAT_AREA] < min_px)
    small = small[small > 0]
    out = m.copy()
    if small.size:
        out[y0:y1, x0:x1] &= ~np.isin(cc, small)
    return out


@dataclass
class Carve:
    """A carve's result. ``labels`` is the new complete partition (int32, no -1); ``part_ids`` the
    part's regions (``covered`` ones kept their ids, ``new_id`` holds the pieces cut off the other
    regions, None when there were none); ``donors`` how many pixels each cut region gave;
    ``mask`` the part's final pixels (``np.isin(labels, part_ids)``)."""
    labels: np.ndarray
    part_ids: list[int]
    new_id: Optional[int]
    covered: list[int]
    donors: dict[int, int]
    mask: np.ndarray

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def carve(labels: np.ndarray, mask: np.ndarray, min_px: int = MIN_PIECE_PX) -> Optional[Carve]:
    """Carve ``mask`` (bool HxW) into the label map ``labels`` (int32 HxW, 0..N-1) as one part.

    Guarantees: the result is a complete partition with no -1; every region id that existed
    keeps its id (none is emptied, none renumbered: a region wholly inside the part keeps its id
    and belongs to the part, the pieces cut off the other regions get the one new id N); pieces
    of the mask under ``min_px`` are not carved (they merge back into their region) unless they
    belong to a region the mask covers whole (it joins the part whole), a piece a cut region
    would keep under ``min_px`` that touches the part joins the part, and larger holes of the mask
    stay with the regions they are in. Returns None when nothing is left to carve."""
    labels = np.ascontiguousarray(labels, np.int32)
    mask = np.asarray(mask, bool)
    if mask.shape != labels.shape:
        raise ValueError(f"mask {mask.shape} does not match the label map {labels.shape}")
    if labels.size == 0 or labels.min() < 0:
        raise ValueError("labels must be a complete 0..N-1 partition")
    n = int(labels.max()) + 1
    area = np.bincount(labels.ravel(), minlength=n)
    m = drop_islands(mask, min_px)
    # a region the mask covers whole stays whole in the part, its own specks included: dropped as
    # islands of the mask, the 1 px pieces of a group the part was told to take stayed behind as it
    inside0 = np.bincount(labels[mask], minlength=n)
    whole = (inside0 > 0) & (inside0 == area)
    if whole.any():
        m |= mask & whole[labels]
    if not m.any():
        return None
    H, W = labels.shape
    ys, xs = np.nonzero(m)
    pad = max(2, int(min_px) + 1)                  # a speck next to the part lies inside the window
    y0, y1 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + 1 + pad)
    lab = labels[y0:y1, x0:x1]
    mc = m[y0:y1, x0:x1].copy()
    grown = cv2.dilate(mc.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    hh, ww = lab.shape
    for r in np.unique(lab[mc]).tolist():
        rem = (lab == r) & ~mc
        if not rem.any():
            continue
        k, cc, st, _ = cv2.connectedComponentsWithStats(rem.astype(np.uint8), connectivity=8)
        for i in range(1, k):
            if st[i, cv2.CC_STAT_AREA] >= min_px:
                continue
            bx, by, bw, bh = (int(v) for v in st[i, :4])
            # a piece at the window's edge may go on outside it (unless that edge is the image's)
            if (bx == 0 and x0 > 0) or (by == 0 and y0 > 0) or (bx + bw == ww and x1 < W) or (by + bh == hh and y1 < H):
                continue
            piece = cc == i
            if (piece & grown).any():
                mc |= piece
    inside = np.bincount(lab[mc], minlength=n)
    covered = [int(r) for r in np.flatnonzero(inside) if inside[r] == area[r]]
    donors = {int(r): int(inside[r]) for r in np.flatnonzero(inside) if inside[r] != area[r]}
    cut = mc & ~np.isin(lab, covered) if covered else mc.copy()
    out = labels.copy()
    view = out[y0:y1, x0:x1]
    new_id: Optional[int] = None
    if cut.any():
        if covered and int(cut.sum()) < min_px:
            view[cut] = max(covered, key=lambda r: (int(area[r]), -r))    # a few px: the covered region grows
        else:
            new_id = n
            view[cut] = new_id
    final = np.zeros((H, W), bool)
    final[y0:y1, x0:x1] = mc
    part_ids = sorted(covered) + ([new_id] if new_id is not None else [])
    return Carve(out, part_ids, new_id, sorted(covered), donors, final)


def absorb_parts(mask: np.ndarray, labels: np.ndarray, regions: Sequence[Region],
                 share: float = ABSORB_SHARE) -> tuple[np.ndarray, list[int]]:
    """``(mask, taken)``: ``mask`` grown by every part instance it covers at least ``share`` of but
    not all (an instance is every region of one ``(part_kind, part_instance)``, measured together:
    a user part of a 47 152 px region and a 225 px one covered 99.5 % is taken whole, its small
    region included), and the ids of the regions it grew over. Guarantees: the result holds
    ``mask``; an instance covered less than ``share``, a part's other instances and every colour
    region are left to the carve (which cuts what it overlaps), so a far instance of the same kind
    never joins the new part."""
    m = np.asarray(mask, bool)
    lab = np.asarray(labels)
    if m.shape != lab.shape or not m.any():
        return m.copy(), []
    n = max(int(lab.max()) + 1, max((r.id for r in regions), default=-1) + 1)
    inside = np.bincount(lab[m].ravel(), minlength=n)
    total = np.bincount(lab.ravel(), minlength=n)
    instances: dict[tuple[str, int], list[int]] = {}
    for r in regions:
        if r.part_kind and 0 <= int(r.id) < n:
            instances.setdefault((str(r.part_kind), int(r.part_instance)), []).append(int(r.id))
    taken: list[int] = []
    for ids in instances.values():
        got = int(sum(int(inside[i]) for i in ids))
        tot = int(sum(int(total[i]) for i in ids))
        if 0 < got < tot and got >= share * tot:
            taken.extend(i for i in ids if inside[i] < total[i])
    taken.sort()
    return (m | np.isin(lab, taken)) if taken else m.copy(), taken


def keep_names(old_groups: Sequence[ColorGroup], groups: Sequence[ColorGroup], new_of: dict[int, int],
               recompute: Iterable[int] = (), restore: Optional[dict[int, str]] = None) -> None:
    """In place, after a user-part edit rebuilt the groups (``new_of``: old group id -> new id of
    every group that is still there): each of those groups keeps the name it had, so the edit
    renames nothing it did not make (the rebuild renumbers repeated colour names by id: cutting a
    part out of "Silver 2" renamed the untouched "Silver 3" to "Silver 2", and the donor "Gold"
    became "Yellow"). Only the part groups in ``recompute`` (old ids) whose name was the kind's
    automatic one take the name that follows their instances now ("Brake calipers" -> "Brake
    caliper"). A group given back by Remove (``restore``: new id -> the name it had) gets that
    name again. A group's name that now repeats another's gets the next free suffix."""
    from . import grouping
    recompute = {int(i) for i in recompute}
    restore = {int(k): str(v) for k, v in (restore or {}).items() if v}
    old_at = {new_of[g.id]: g for g in old_groups if g.id in new_of}
    fixed: set[int] = set()
    for g in groups:
        old = old_at.get(g.id)
        if old is None or (old.part or "") != (g.part or ""):
            continue
        if g.part and old.id in recompute and grouping._keep_name(old) is None:
            continue
        g.name = old.name
        fixed.add(g.id)
    taken = {g.name for g in groups if g.id in fixed}
    for g in groups:
        if g.id in fixed:
            continue
        if g.id in restore:
            base = restore[g.id]
        else:
            m = grouping._NAME_SUFFIX.match(g.name)
            base = m.group(1) if m and m.group(1) == grouping._auto_group_name(g) else g.name
        name, k = base, 1
        while name in taken:
            k += 1
            name = f"{base} {k}"
        g.name = name
        taken.add(name)


def region_stats(labels: np.ndarray, albedo_lin: np.ndarray, ids: Sequence[int]) -> dict[int, dict]:
    """``{id: {"area", "bbox" (x0, y0, x1, y1 exclusive), "lab" (the lower median of each CIELAB
    channel of the linear albedo, to 0.01 like :func:`labelops.region_medians`), "touches_border"}}``
    for ``ids`` (numpy, on each region's bounding box, converting only its pixels); an empty id is
    left out."""
    from .. import imageio
    labels = np.ascontiguousarray(labels, np.int32)
    H, W = labels.shape
    out: dict[int, dict] = {}
    from scipy import ndimage
    want = sorted({int(i) for i in ids})
    if not want:
        return out
    objs = ndimage.find_objects(labels + 1, max_label=max(want) + 1)
    for i in want:
        sl = objs[i] if i < len(objs) else None
        if sl is None:
            continue
        m = labels[sl] == i
        px = np.asarray(imageio.linear_to_lab(np.ascontiguousarray(albedo_lin[sl][m], np.float32)[None])[0], np.float64)
        n = len(px)
        if n == 0:
            continue
        k = (n - 1) // 2
        med = tuple(float(np.round(np.partition(px[:, c], k)[k], 2)) for c in range(3))
        ys, xs = sl
        touches = ys.start == 0 or xs.start == 0 or ys.stop == H or xs.stop == W
        if touches:
            touches = bool(m[0].any() and ys.start == 0) or bool(m[-1].any() and ys.stop == H) \
                or bool(m[:, 0].any() and xs.start == 0) or bool(m[:, -1].any() and xs.stop == W)
        out[i] = {"area": int(n), "bbox": (int(xs.start), int(ys.start), int(xs.stop), int(ys.stop)),
                  "lab": med, "touches_border": bool(touches)}
    return out


# ---------------------------------------------------------------------- tags

def tag(r: Region, kind: str, label: str, plural: Optional[str] = None, instance: int = 0) -> Region:
    """``r`` as a region of part ``kind`` (a part is the object: its backdrop flag goes)."""
    return replace(r, part_kind=str(kind), part_label=str(label), part_plural=str(plural or label),
                   part_instance=int(instance), backdrop=False)


def untag(r: Region) -> Region:
    """``r`` without a part tag (a colour region again)."""
    return replace(r, part_kind="", part_label="", part_plural="", part_instance=-1)


def registry_of(regions: Sequence[Region]) -> dict[str, dict]:
    """``{kind: {"label", "regions": [ids]}}`` of every user part among ``regions``."""
    out: dict[str, dict] = {}
    for r in sorted(regions, key=lambda r: r.id):
        if is_user_kind(r.part_kind):
            rec = out.setdefault(r.part_kind, {"label": r.part_label or r.part_kind, "regions": []})
            rec["regions"].append(int(r.id))
    return out


def _rebuild(regions: list[Region], groups: Sequence[ColorGroup], labels: np.ndarray,
             assignment: dict[int, int], fresh_names: Optional[dict[int, str]] = None
             ) -> tuple[list[Region], list[ColorGroup], np.ndarray]:
    """Groups from ``assignment`` keeping every existing group's flags (the user's own background
    choice on a part group included: flagged by hand it stayed flagged), custom name and engine
    reference (an automatic name is recomputed, so a part that gained an instance reads "Brake
    calipers"); a fresh group (``fresh_names``) is unlocked and never background."""
    from . import grouping
    carry = {g.id: {"name": grouping._keep_name(g), "locked": g.locked, "is_background": g.is_background,
                    "ref_lab": g.ref_lab} for g in groups}
    for gid, name in (fresh_names or {}).items():
        carry[gid] = {"name": name, "locked": False, "is_background": False, "ref_lab": None}
    regions, groups, group_map = grouping._finalize(regions, labels, assignment, carry, sort_by_area=False)
    grouping._check_state(regions, groups)
    return regions, groups, group_map


def normalize(regions: Sequence[Region], groups: Sequence[ColorGroup], labels: np.ndarray
              ) -> tuple[list[Region], list[ColorGroup], np.ndarray, bool]:
    """After a merge, split or move: every user part's regions follow the group they are in.

    A region of user part ``k`` in a group that is not a part group of ``k`` (the group's part is
    decided by area, as the Part badge is) either joins that group's part (a group of another
    part kind: it becomes a new instance of that kind, so a far caliper merged into "Brake
    caliper" makes "Brake calipers") or loses its tag (a colour group: the part was merged into
    its neighbour and is gone, also after a regroup). A colour region in a user part's group
    joins the part. Detected parts are left as they are. Returns ``(regions, groups, group_map,
    changed)``; the group ids, flags and custom names are kept."""
    regions = list(regions)
    by_gid = {g.id: g for g in groups}
    inst: dict[str, int] = {}
    for r in regions:
        if r.part_kind:
            inst[r.part_kind] = max(inst.get(r.part_kind, -1), int(r.part_instance))
    retag: dict[tuple[str, str], int] = {}
    first_inst: dict[str, int] = {}
    for r in regions:
        if r.part_kind:
            first_inst[r.part_kind] = min(first_inst.get(r.part_kind, 1 << 30), max(0, int(r.part_instance)))
    out: list[Region] = []
    changed = False
    for r in regions:
        g = by_gid.get(r.group_id)
        if g is None:
            out.append(r)
            continue
        if is_user_kind(r.part_kind):
            if g.part == r.part_kind:
                out.append(r)
            elif g.part:
                key = (r.part_kind, g.part)
                if key not in retag:
                    inst[g.part] = inst.get(g.part, -1) + 1
                    retag[key] = inst[g.part]
                out.append(tag(r, g.part, g.part_label or g.part, g.part_plural or g.part_label, retag[key]))
                changed = True
            else:
                out.append(untag(r))
                changed = True
        elif not r.part_kind and is_user_kind(g.part):
            out.append(tag(r, g.part, g.part_label, g.part_plural, first_inst.get(g.part, 0)))
            changed = True
        else:
            out.append(r)
    if not changed:
        from . import grouping
        return regions, list(groups), grouping._group_map(regions, labels, len(groups)), False
    regions, groups, group_map = _rebuild(out, groups, labels, {r.id: r.group_id for r in out})
    return regions, groups, group_map, True


def apply_registry(regions: Sequence[Region], groups: Sequence[ColorGroup], labels: np.ndarray,
                   registry: dict[str, dict]) -> tuple[list[Region], list[ColorGroup], np.ndarray, bool]:
    """The safety net after a regroup: every user part of ``registry`` (``{kind: {"label",
    "regions"}}``, as ``user_flags.json`` keeps it) is one group of its own. A listed region that
    lost its tag gets it back, and a part whose regions are not alone in a part group of its
    kind is moved into a fresh group. A no-op (``changed`` False) in the normal case, where the
    seed and the region tags already gave the part its group."""
    regions = list(regions)
    by_id = {r.id: r for r in regions}
    changed = False
    for kind, rec in (registry or {}).items():
        if not is_user_kind(kind):
            continue
        label = str(rec.get("label") or kind)
        for rid in rec.get("regions") or []:
            r = by_id.get(int(rid))
            if r is not None and r.part_kind != kind and (not r.part_kind or is_user_kind(r.part_kind)):
                by_id[r.id] = tag(r, kind, label, label, 0)
                changed = True
    regions = [by_id[r.id] for r in regions]
    if changed:
        regions, groups, group_map = _rebuild(regions, groups, labels, {r.id: r.group_id for r in regions})
    else:
        groups = list(groups)
    assignment = {r.id: r.group_id for r in regions}
    fresh = max([g.id for g in groups] + [-1]) + 1
    names: dict[int, str] = {}
    moved = False
    by_gid = {g.id: g for g in groups}
    for kind in sorted({r.part_kind for r in regions if is_user_kind(r.part_kind)}):
        mine = [r for r in regions if r.part_kind == kind]
        gids = {r.group_id for r in mine}
        if len(gids) == 1:
            g = by_gid[next(iter(gids))]
            if g.part == kind and all(by_id.get(rid) is None or by_id[rid].part_kind == kind for rid in g.region_ids):
                continue
        for r in mine:
            assignment[r.id] = fresh
        names[fresh] = mine[0].part_label or kind
        fresh += 1
        moved = True
    if moved:
        regions, groups, group_map = _rebuild(regions, groups, labels, assignment, fresh_names=names)
        return regions, groups, group_map, True
    from . import grouping
    return regions, groups, grouping._group_map(regions, labels, len(groups)), changed


def sync_seed_tags(tags: dict[int, dict], origin: np.ndarray, regions: Sequence[Region], n_in: int
                   ) -> tuple[dict[int, dict], bool]:
    """The regroup seed's input-region part tags after an edit re-tagged final regions
    (:func:`normalize`): an input region tagged with a user kind, or all of whose descendants
    (the final regions whose ``origin`` it is) carry one user kind, takes its descendants' tag
    (the area-weighted majority) or loses its tag when they carry none. Every other tag stays.
    Returns ``(tags, changed)``."""
    origin = np.asarray(origin, np.int64).ravel()
    desc: dict[int, list[Region]] = {}
    for r in regions:
        o = int(origin[r.id]) if r.id < len(origin) else -1
        if 0 <= o < n_in:
            desc.setdefault(o, []).append(r)
    out = {int(k): dict(v) for k, v in (tags or {}).items()}
    changed = False
    for i, rs in desc.items():
        cur = out.get(i)
        cur_user = bool(cur) and is_user_kind(cur.get("kind"))
        weights: Counter[str] = Counter()
        for r in rs:
            weights[r.part_kind] += max(int(r.area), 1)
        best, w = weights.most_common(1)[0]
        whole_user = is_user_kind(best) and w == sum(weights.values())
        if not (cur_user or whole_user):
            continue
        if best:
            src = next(r for r in rs if r.part_kind == best)
            rec = {"kind": best, "label": src.part_label or best, "plural": src.part_plural or src.part_label or best,
                   "instance": max(0, int(src.part_instance))}
            if cur != rec:
                out[i] = rec
                changed = True
        elif cur is not None:
            out.pop(i, None)
            changed = True
    # a user part that is gone (removed, merged away) leaves no tag behind on an input region
    alive = {r.part_kind for r in regions if is_user_kind(r.part_kind)}
    for i in [i for i, t in out.items() if is_user_kind(t.get("kind")) and t.get("kind") not in alive]:
        out.pop(i)
        changed = True
    return out, changed
