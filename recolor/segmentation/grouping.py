"""Cluster regions into color groups by median albedo, and edit those groups.

Group ids are always 0..G-1 and `group_map` always holds exactly those ids, so the
engine can one-hot it directly. Region ids are stable across every operation here;
only `split_group` may append new region ids (when it has to split a single region at
the pixel level, in which case `labels` is updated in place).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from .. import colornames, imageio
from ..types import ColorGroup, Region
from .labelops import bboxes, border_counts, region_areas, region_medians

BACKGROUND_BORDER_FRACTION = 0.35
SPLIT_MIN_DELTA_E = 3.5         # split_group is a no-op when its k-means centroids are all closer than this
_GRID_PRECLUSTER_ABOVE = 1500   # regions; above this, near-identical medians are pooled first
_GRID_STEP = 1.5                # Lab units of that pooling grid

GroupState = tuple[list[Region], list[ColorGroup], np.ndarray]


# ---------------------------------------------------------------------- clustering

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-column weighted median of `values` [n, 3] with weights [n]."""
    out = np.zeros(values.shape[1], np.float32)
    w = weights.astype(np.float64)
    for c in range(values.shape[1]):
        order = np.argsort(values[:, c], kind="stable")
        cw = np.cumsum(w[order])
        idx = int(np.searchsorted(cw, 0.5 * cw[-1]))
        out[c] = values[order[min(idx, len(order) - 1)], c]
    return out


def cluster_colors(lab: np.ndarray, areas: np.ndarray, delta_e: float,
                   max_groups: Optional[int] = None) -> np.ndarray:
    """Area-weighted agglomerative clustering of Lab colors.

    Repeatedly merges the two clusters whose area-weighted centroids are closest in
    CIEDE2000 while that distance is below `delta_e`; then keeps merging the closest
    pair until at most `max_groups` remain (if set). Returns an int64 cluster index per
    row, 0..K-1, numbered by first appearance. Deterministic.
    """
    n = len(lab)
    if n == 0:
        return np.zeros(0, np.int64)
    lab = np.asarray(lab, np.float64)
    areas = np.asarray(areas, np.float64).clip(min=1.0)
    member = np.arange(n)
    # Pool near-identical colors on a coarse grid first when there are very many regions.
    if n > _GRID_PRECLUSTER_ABOVE:
        cell = np.round(lab / _GRID_STEP).astype(np.int64)
        _, member = np.unique(cell, axis=0, return_inverse=True)
        member = member.ravel()
    k = int(member.max()) + 1
    cent = np.zeros((k, 3))
    wsum = np.zeros(k)
    np.add.at(wsum, member, areas)
    np.add.at(cent, member, lab * areas[:, None])
    cent /= wsum[:, None]
    active = np.ones(k, bool)
    if k > 1:
        ii, jj = np.triu_indices(k, 1)
        D = np.full((k, k), np.inf, np.float32)
        D[ii, jj] = imageio.delta_e(cent[ii], cent[jj])
        D[jj, ii] = D[ii, jj]
    else:
        D = np.full((1, 1), np.inf, np.float32)
    n_active = k
    while n_active > 1:
        flat = int(np.argmin(D))
        i, j = divmod(flat, k)
        d = float(D[i, j])
        within = d < delta_e
        over_cap = max_groups is not None and n_active > max_groups
        if not within and not over_cap:
            break
        if wsum[i] < wsum[j]:
            i, j = j, i
        cent[i] = (cent[i] * wsum[i] + cent[j] * wsum[j]) / (wsum[i] + wsum[j])
        wsum[i] += wsum[j]
        active[j] = False
        member[member == j] = i
        D[j, :] = np.inf
        D[:, j] = np.inf
        idx = np.flatnonzero(active)
        idx = idx[idx != i]
        if len(idx):
            row = imageio.delta_e(np.repeat(cent[i][None], len(idx), 0), cent[idx])
            D[i, idx] = row
            D[idx, i] = row
        D[i, i] = np.inf
        n_active -= 1
    _, out = np.unique(member, return_inverse=True)
    return out.ravel().astype(np.int64)


# ---------------------------------------------------------------------- records

def _make_regions(labels: np.ndarray, region_info: list[dict], meds: np.ndarray) -> list[Region]:
    n = len(meds)
    areas = region_areas(labels, n)
    bb = bboxes(labels, n)
    border = border_counts(labels, n) > 0
    by_id: dict[int, dict] = {}
    for k, d in enumerate(region_info or []):
        by_id[int(d.get("id", k))] = d
    regions: list[Region] = []
    for i in range(n):
        if areas[i] == 0:
            continue
        d = by_id.get(i, {})
        lab = tuple(float(v) for v in meds[i])
        regions.append(Region(
            id=i, area=int(areas[i]), bbox=tuple(int(v) for v in bb[i]),
            albedo_lab=lab, albedo_hex=imageio.lab_to_hex(lab), group_id=0,
            touches_border=bool(border[i]), source=str(d.get("source", "sam")),
            confidence=float(d.get("confidence", 0.0)),
        ))
    return regions


def _auto_name(lab) -> str:
    return colornames.nearest_name(lab)


def _group_from(gid: int, members: list[Region], npx: int, name: str | None = None,
                locked: bool = False, is_background: bool = False) -> ColorGroup:
    lab_arr = np.array([r.albedo_lab for r in members], np.float32)
    w = np.array([r.area for r in members], np.float64)
    lab = tuple(float(v) for v in _weighted_median(lab_arr, w))
    area = int(w.sum())
    return ColorGroup(
        id=gid, name=name or _auto_name(lab), albedo_lab=lab, albedo_hex=imageio.lab_to_hex(lab),
        area=area, area_frac=area / float(max(1, npx)),
        region_ids=sorted(r.id for r in members),
        hue_family=colornames.hue_family(lab), locked=locked, is_background=is_background,
    )


def _finalize(regions: list[Region], labels: np.ndarray, assignment: dict[int, int],
              carry: dict[int, dict[str, Any]] | None = None, sort_by_area: bool = True) -> GroupState:
    """Build groups from a region -> provisional group assignment.

    Provisional ids are renumbered to 0..G-1 (by area descending when `sort_by_area`,
    else by ascending provisional id). `carry` maps provisional ids to attributes to keep
    (name/locked/is_background). Region objects are copied with their new group_id.
    """
    npx = int(labels.size)
    members: dict[int, list[Region]] = {}
    for r in regions:
        members.setdefault(assignment[r.id], []).append(r)
    prov = list(members)
    if sort_by_area:
        prov.sort(key=lambda g: (-sum(r.area for r in members[g]), g))
    else:
        prov.sort()
    carry = carry or {}
    out_regions: list[Region] = []
    groups: list[ColorGroup] = []
    for new_id, g in enumerate(prov):
        c = carry.get(g, {})
        groups.append(_group_from(new_id, members[g], npx, name=c.get("name"),
                                  locked=bool(c.get("locked", False)),
                                  is_background=bool(c.get("is_background", False))))
        for r in members[g]:
            out_regions.append(replace(r, group_id=new_id))
    out_regions.sort(key=lambda r: r.id)
    group_map = _group_map(out_regions, labels, len(groups))
    return out_regions, groups, group_map


def _group_map(regions: list[Region], labels: np.ndarray, n_groups: int) -> np.ndarray:
    lut = np.zeros(int(labels.max()) + 1, np.int32)
    for r in regions:
        lut[r.id] = r.group_id
    gm = lut[labels].astype(np.int32)
    if n_groups and (gm.max() >= n_groups or gm.min() < 0):
        raise AssertionError("group map holds ids outside 0..G-1")
    return gm


def _mark_background(groups: list[ColorGroup], group_map: np.ndarray) -> None:
    if not groups:
        return
    counts = border_counts(group_map, len(groups))
    total = float(counts.sum())
    if total <= 0:
        return
    g = int(np.argmax(counts))
    if counts[g] / total > BACKGROUND_BORDER_FRACTION:
        groups[g].is_background = True


def _check_state(regions: list[Region], groups: list[ColorGroup]) -> None:
    ids = [g.id for g in groups]
    if ids != list(range(len(groups))):
        raise AssertionError("group ids must be 0..G-1")
    seen = set()
    for g in groups:
        for rid in g.region_ids:
            if rid in seen:
                raise AssertionError(f"region {rid} listed in two groups")
            seen.add(rid)
    if seen != {r.id for r in regions}:
        raise AssertionError("groups do not partition the regions")


def _split_is_meaningful(cent: np.ndarray, sample: np.ndarray | None = None,
                         min_delta_e: float = SPLIT_MIN_DELTA_E) -> bool:
    """Decide whether a k-means split into the Lab centroids `cent` separates real
    colours rather than sensor noise.

    A k-means fit always returns k centroids, even on one flat colour, and on a pure
    noise blob they sit ~2-3 sigma apart whatever k is, so two checks are needed: the
    two most distant centroids must differ by at least `min_delta_e` (CIEDE2000), and,
    when `sample` (the pixels the fit was made on, [n, 3] Lab) has enough rows, the
    sample projected onto the axis between those two centroids must have a density
    valley (`hierarchy.is_bimodal`) - a Gaussian blob projected on any axis peaks in the
    middle and is rejected; two tones give two peaks and pass.
    """
    from .hierarchy import is_bimodal
    cent = np.asarray(cent, np.float32)
    if len(cent) < 2:
        return False
    ii, jj = np.triu_indices(len(cent), 1)
    de = imageio.delta_e(cent[ii], cent[jj])
    de = np.where(np.isfinite(de), de, -1.0)
    far = int(np.argmax(de))
    if de[far] < min_delta_e:
        return False
    if sample is not None and len(sample) >= 32:
        return is_bimodal(np.asarray(sample, np.float32), cent[ii[far]], cent[jj[far]])
    return True


def _cluster_regions(regions: list[Region], delta_e: float, max_groups: Optional[int]) -> dict[int, int]:
    lab = np.array([r.albedo_lab for r in regions], np.float32)
    areas = np.array([r.area for r in regions], np.float64)
    cl = cluster_colors(lab, areas, delta_e, max_groups)
    return {r.id: int(c) for r, c in zip(regions, cl)}


# ---------------------------------------------------------------------- public API

def group_regions(labels: np.ndarray, albedo_lin: np.ndarray, region_info: list[dict],
                  max_groups: Optional[int] = None, delta_e: float = 10.0) -> GroupState:
    """Cluster regions by median albedo into color groups.

    Returns `(regions, groups, group_map)`: one `Region` per label id (ids equal to the
    label values), `ColorGroup`s sorted by area descending with ids 0..G-1, and an int32
    HxW map of group ids. Every region belongs to exactly one group; group albedo is the
    area-weighted median of its regions' median albedo; `is_background` marks the group
    owning more than 35 % of the image border (at most one). `delta_e` is the CIEDE2000
    linkage threshold, `max_groups` an optional hard cap.
    """
    n = int(labels.max()) + 1
    if n <= 0 or (labels < 0).any():
        raise ValueError("labels must be a complete 0..N-1 partition")
    lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
    meds = region_medians(labels, lab, n)
    regions = _make_regions(labels, region_info, meds)
    assignment = _cluster_regions(regions, delta_e, max_groups)
    regions, groups, group_map = _finalize(regions, labels, assignment)
    _mark_background(groups, group_map)
    _check_state(regions, groups)
    return regions, groups, group_map


def regroup(regions: list[Region], labels: np.ndarray, albedo_lin: np.ndarray,
            max_groups: Optional[int] = None, delta_e: float = 10.0) -> GroupState:
    """Re-cluster existing regions with new parameters without touching the label map.

    Uses the regions' stored median albedo (no pixel work), so it is cheap. Region ids
    are unchanged; user flags on the previous groups are discarded since the groups are
    redefined. Returns the same tuple as `group_regions`.
    """
    assignment = _cluster_regions(regions, delta_e, max_groups)
    regions, groups, group_map = _finalize(regions, labels, assignment)
    _mark_background(groups, group_map)
    _check_state(regions, groups)
    return regions, groups, group_map


def _carry_all(groups: list[ColorGroup]) -> dict[int, dict[str, Any]]:
    return {g.id: {"name": g.name, "locked": g.locked, "is_background": g.is_background} for g in groups}


def _keep_name(g: ColorGroup) -> str | None:
    """The user's custom name if they set one, else None so the name is recomputed."""
    return g.name if g.name != _auto_name(g.albedo_lab) else None


def merge_groups(groups: list[ColorGroup], regions: list[Region], group_map: np.ndarray,
                 labels: np.ndarray, ids: list[int]) -> GroupState:
    """Merge the groups listed in `ids` into one (the lowest id survives).

    Other groups keep their relative order and are renumbered to stay contiguous; region
    ids do not change. The merged group is locked / background if any member was, and
    keeps a custom name if the surviving group had one. Fewer than two valid ids is a
    no-op that still returns fresh copies.
    """
    ids = sorted({int(i) for i in ids if 0 <= int(i) < len(groups)})
    target = ids[0] if ids else None
    assignment = {}
    for r in regions:
        assignment[r.id] = target if (target is not None and r.group_id in ids) else r.group_id
    carry = _carry_all(groups)
    if target is not None:
        merged = [groups[i] for i in ids]
        carry[target] = {
            "name": _keep_name(groups[target]),
            "locked": any(g.locked for g in merged),
            "is_background": any(g.is_background for g in merged),
        }
    regions, groups, group_map = _finalize(regions, labels, assignment, carry, sort_by_area=False)
    _check_state(regions, groups)
    return regions, groups, group_map


def split_group(groups: list[ColorGroup], regions: list[Region], group_map: np.ndarray,
                labels: np.ndarray, albedo_lin: np.ndarray, gid: int, k: int = 2) -> GroupState:
    """Split group `gid` into up to `k` groups by albedo.

    When the group has enough regions to form `k` distinct clusters, whole regions are
    reassigned (k-means on their median albedo, area-weighted). Otherwise its pixels are
    clustered and each region is cut along albedo; the new pieces get fresh region ids
    appended after the existing ones and `labels` is updated **in place** (still a
    contiguous 0..N-1 partition). New groups get the next free ids; existing group ids
    and flags are unchanged. If no split is possible - including when the group is
    genuinely one colour (all k-means centroids within `SPLIT_MIN_DELTA_E` CIEDE2000 of
    each other, i.e. the split would only follow sensor noise) - the state is returned
    unchanged (as copies) and no region id is added.
    """
    from sklearn.cluster import KMeans

    k = max(2, int(k))
    gid = int(gid)
    if not (0 <= gid < len(groups)):
        raise ValueError(f"no group {gid}")
    members = [r for r in regions if r.group_id == gid]
    carry = _carry_all(groups)
    assignment = {r.id: r.group_id for r in regions}
    next_gid = len(groups)

    # Region-level split first.
    if len(members) >= k:
        lab = np.array([r.albedo_lab for r in members], np.float64)
        w = np.array([r.area for r in members], np.float64)
        km = KMeans(n_clusters=k, n_init=8, random_state=0).fit(lab, sample_weight=w)
        cl = km.labels_
        if len(np.unique(cl)) >= 2 and not _split_is_meaningful(km.cluster_centers_[np.unique(cl)]):
            # Every region is (near-)identical in colour: a split would be meaningless.
            return _finalize(regions, labels, assignment, carry, sort_by_area=False)
        if len(np.unique(cl)) >= 2:
            order = np.argsort(-np.bincount(cl, weights=w, minlength=k))
            rank = {int(c): i for i, c in enumerate(order)}
            for r, c in zip(members, cl):
                rk = rank[int(c)]
                assignment[r.id] = gid if rk == 0 else next_gid + rk - 1
            regions, groups, group_map = _finalize(regions, labels, assignment, carry, sort_by_area=False)
            _check_state(regions, groups)
            return regions, groups, group_map

    # Pixel-level split of the member regions.
    lab_img = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
    sel = group_map == gid
    pix = lab_img[sel]
    if len(pix) < 2 * k:
        return _finalize(regions, labels, assignment, carry, sort_by_area=False)
    sample = pix[:: max(1, len(pix) // 20000)]
    km = KMeans(n_clusters=k, n_init=8, random_state=0).fit(sample)
    cent = km.cluster_centers_.astype(np.float32)
    if not _split_is_meaningful(cent, sample):
        # Unimodal group (flat colour plus noise): splitting would cut along noise and
        # present two identical swatches. Leave the state - and `labels` - untouched.
        return _finalize(regions, labels, assignment, carry, sort_by_area=False)
    # The cluster holding most of the group's pixels stays in `gid`; the others become
    # new groups (provisional ids after the existing ones, compacted by _finalize).
    d_all = ((sample[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1)
    order = np.argsort(-np.bincount(d_all, minlength=k), kind="stable")
    group_for = {int(c): (gid if i == 0 else next_gid + i - 1) for i, c in enumerate(order)}
    next_rid = int(labels.max()) + 1
    new_ids: list[tuple[int, Region]] = []   # (new region id, parent region)
    for r in members:
        x0, y0, x1, y1 = r.bbox
        view = labels[y0:y1, x0:x1]
        crop = view == r.id
        p = lab_img[y0:y1, x0:x1][crop]
        cl = ((p[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1)
        counts = np.bincount(cl, minlength=k)
        keep = int(np.argmax(counts))
        min_part = max(16, int(0.01 * r.area))
        cl = np.where(counts[cl] < min_part, keep, cl)   # crumbs stay with the kept part
        assignment[r.id] = group_for[keep]
        for c in range(k):
            if c == keep or not np.any(cl == c):
                continue
            part = np.zeros(crop.shape, bool)
            part[crop] = cl == c
            view[part] = next_rid
            assignment[next_rid] = group_for[c]
            new_ids.append((next_rid, r))
            next_rid += 1
    if not new_ids:
        return _finalize(regions, labels, assignment, carry, sort_by_area=False)
    # Refresh every region's stats from the label map (cut regions shrank).
    n = next_rid
    meds = region_medians(labels, lab_img, n)
    areas = region_areas(labels, n)
    bb = bboxes(labels, n)
    border = border_counts(labels, n) > 0
    refreshed: list[Region] = []
    for r in regions + [replace(parent, id=rid, source="split") for rid, parent in new_ids]:
        if areas[r.id] == 0:
            continue
        lab_t = tuple(float(v) for v in meds[r.id])
        refreshed.append(replace(r, area=int(areas[r.id]), bbox=tuple(int(v) for v in bb[r.id]),
                                 albedo_lab=lab_t, albedo_hex=imageio.lab_to_hex(lab_t),
                                 touches_border=bool(border[r.id])))
    regions, groups, group_map = _finalize(refreshed, labels, assignment, carry, sort_by_area=False)
    _check_state(regions, groups)
    return regions, groups, group_map


def move_regions(groups: list[ColorGroup], regions: list[Region], group_map: np.ndarray,
                 labels: np.ndarray, region_ids: list[int], gid: int) -> GroupState:
    """Move the given regions into group `gid`.

    Region ids are unchanged. A group left empty is removed and the remaining ids are
    renumbered contiguously (relative order kept). Flags and custom names are preserved;
    auto-generated names of the affected groups are refreshed from their new albedo.
    """
    gid = int(gid)
    if not (0 <= gid < len(groups)):
        raise ValueError(f"no group {gid}")
    want = {int(r) for r in region_ids}
    assignment = {r.id: (gid if r.id in want else r.group_id) for r in regions}
    touched = {gid} | {r.group_id for r in regions if r.id in want}
    carry = _carry_all(groups)
    for g in touched:
        carry[g]["name"] = _keep_name(groups[g])
    regions, groups, group_map = _finalize(regions, labels, assignment, carry, sort_by_area=False)
    _check_state(regions, groups)
    return regions, groups, group_map
