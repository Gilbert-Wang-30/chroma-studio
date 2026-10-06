"""Cluster regions into color groups by median albedo, and edit those groups.

Group ids are always 0..G-1 and `group_map` always holds exactly those ids, so the
engine can one-hot it directly. Region ids are stable across every operation here;
only `split_group` may append new region ids (when it has to split a single region at
the pixel level, in which case `labels` is updated in place).

Three things beyond the clustering itself:

- **Background.** With a foreground matte (:func:`backdrop_decisions`, run by the regions
  stage from :mod:`foreground`), every region is backdrop or object before the clustering,
  the two sets are clustered separately (no group mixes them) and every backdrop group is
  flagged ``is_background``; a group is background afterwards when most of its area is
  backdrop regions (:func:`_mark_background`), so the flags survive every rebuild. The
  backdrop reaches on through chains of regions the matte calls backdrop (the other cars of a
  showroom, a piece at a time) unless a region is enclosed by the object. Without
  a matte the one group owning more than BACKGROUND_BORDER_FRACTION of the border is
  flagged, as before.
- **One paint under different light** (:func:`absorb_lit`). The albedo keeps part of the
  shading, so the same paint clusters into a lit and a shadowed group (the RX-78's chest
  sides, the BMW's panel under the mirror). After the clustering every chromatic group
  joins the closest larger anchor whose lightness-normalised albedo and photo colour it
  shares, unless the shipped material rule says it is another material (duller and much
  darker). The intrinsic shading layer does not carry that difference (measured: the
  decomposition put the whole gap into the albedo), the photo's lightness-normalised
  chromaticity does; neutral pairs are never touched.
- **Names.** Two groups never share a name: the second "Copper" becomes "Copper 2".
- **Detected parts.** A region the regions stage stamped as a detected part
  (``Region.part_kind``, :func:`smallparts.stamp_parts`) never enters the colour
  clustering: every part kind is one group of its own ("Shock spring", "Wheel rims": the
  plural when it holds several instances), whatever its colour and however small, never
  background, never merged with an untagged region or another kind, never moved by the lit
  merge (:func:`absorb_lit`), and outside the ``max_groups`` cap. A group is a part group
  (``ColorGroup.part``) when most of its area is one kind's part regions, so the flag and the
  name follow every rebuild, merge, split and move. :func:`split_instances` splits a part
  group into one group per instance, named by where each sits ("Mirror (left)"; on a vehicle
  whose front the parts give away, "Wheel rim (front)" / "(rear)": :func:`vehicle_front`).
- **Panel view.** :func:`annotate_groups` marks the tiny object groups that are their
  neighbour under other light ``minor`` (below MINOR_FRAC of the object area, neither a part nor
  lettering, within MINOR_DE or, both coloured, of the same body colour as the group owning most
  of the ring around it, their ``parent``); the Groups panel collapses them under a Minor
  divider. A tiny group of a colour of its own is a real small part and stays among the colours
  (a white badge on black trim too). A view rule only: nothing is merged or hidden from
  painting. The rule is versioned (PANEL_RULE), so a job annotated by an older rule is annotated
  again when it is served.
- **One paint under a sheen** (:func:`absorb_lit` with :data:`SHEEN_LIT`): the refinement's last
  step joins a group whose albedo is a larger paint's and whose photo is that paint's under a
  white sheen (lighter and duller in the same hue: a roof facing a bright ceiling).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np

from .. import colornames, imageio
from ..types import SHINY_GLINT_SHARE, ColorGroup, Region
from .labelops import adjacency, bboxes, border_counts, region_areas, region_medians

BACKGROUND_BORDER_FRACTION = 0.35
SPLIT_MIN_DELTA_E = 3.5         # split_group is a no-op when its k-means centroids are all closer than this
#: Region sources a piece cut off by a pixel-level split keeps (lettering, named parts, the wheel
#: split, recovered parts, small distinct parts, prompted parts, detected parts, the pieces a user
#: part cut, :mod:`userparts`): they carry the junk pruning's exemptions, the decal-island
#: treatment and the Minor view's lettering rule, which a cut letter lost as source 'split', and a
#: removed user part's pieces go back to the part they came from. Other pieces are 'split'.
SPLIT_KEEPS_SOURCE = ("text", "named", "wheel", "part", "small", "prompt", "kind", "user")
_GRID_PRECLUSTER_ABOVE = 1500   # regions; above this, near-identical medians are pooled first
_GRID_STEP = 1.5                # Lab units of that pooling grid
# The matte rule (backdrop_decisions): the smallest set of plain-clustering groups owning
# BORDER_COVER of the image border is the border set; a region of matte share <= MATTE_BACK in
# it or next to it is backdrop, one of share >= MATTE_OBJECT is object, every other region of
# the border set is backdrop, every other region is object. A region flagged only through the
# border set and not connected to the border through flagged regions leaves the background and
# joins the nearest unflagged group within REJOIN_DELTA_E (a see-through hole the matte calls
# backdrop keeps its flag). A group is background when at least BACKDROP_GROUP_SHARE of its
# area is backdrop regions.
BORDER_COVER = 0.9
MATTE_BACK = 0.15
MATTE_OBJECT = 0.6
REJOIN_DELTA_E = 10.0
BACKDROP_GROUP_SHARE = 0.5
#: A backdrop region reaches on through a region the matte calls backdrop unless at least this
#: share of that region's boundary is object (matte share >= MATTE_OBJECT): it is enclosed by it.
ENCLOSED_SHARE = 0.5
#: Region ``backdrop`` kinds in the regions stage's info: 0 object, 1 border set, 2 matte.
BG_OBJECT, BG_BORDER, BG_MATTE = 0, 1, 2

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
        kind = str(d.get("part_kind") or "")
        regions.append(Region(
            id=i, area=int(areas[i]), bbox=tuple(int(v) for v in bb[i]),
            albedo_lab=lab, albedo_hex=imageio.lab_to_hex(lab), group_id=0,
            touches_border=bool(border[i]), source=str(d.get("source", "sam")),
            confidence=float(d.get("confidence", 0.0)),
            # a detected part is the object by definition
            backdrop=int(d.get("bg", BG_OBJECT) or 0) > BG_OBJECT and not kind,
            part_kind=kind, part_label=str(d.get("part_label") or "") if kind else "",
            part_plural=str(d.get("part_plural") or "") if kind else "",
            part_instance=int(d.get("part_instance", 0) or 0) if kind else -1,
        ))
    return regions


def _part_of(members: list[Region]) -> tuple[str, str, str, int]:
    """``(kind, label, plural, instances)`` of a group made of ``members``: the part kind
    holding more than half of its area, with its display names and the number of distinct
    instances of it among the members; ``('', '', '', 0)`` for a colour group."""
    area: dict[str, int] = {}
    total = 0
    for r in members:
        total += r.area
        if r.part_kind:
            area[r.part_kind] = area.get(r.part_kind, 0) + r.area
    if not area:
        return "", "", "", 0
    kind = max(area, key=lambda k: (area[k], k))
    if 2 * area[kind] <= total:
        return "", "", "", 0
    of_kind = [r for r in members if r.part_kind == kind]
    first = min(of_kind, key=lambda r: r.id)
    label = first.part_label or kind.replace("_", " ").capitalize()
    plural = first.part_plural or label
    return kind, label, plural, len({r.part_instance for r in of_kind})


def part_name(label: str, plural: str, instances: int) -> str:
    """The automatic name of a part group: the kind's label, its plural for several instances."""
    return plural if instances > 1 else label


_POSITION = re.compile(r"^(.*) \((left|right|upper|lower|front|rear|\d+)\)$")


def _is_auto_part_name(g: ColorGroup, name: str, positional: bool = True) -> bool:
    """True when ``name`` is one this module gives a part group: its kind's label or plural
    (with or without a duplicate suffix) or, with ``positional``, an instance name
    (:func:`instance_names`: "Mirror (left)", "Exhaust (2)")."""
    if not g.part:
        return False
    m = _NAME_SUFFIX.match(name)
    for cand in (name, m.group(1) if m else None):
        if cand is None:
            continue
        if cand in (g.part_label, g.part_plural):
            return True
        p = _POSITION.match(cand) if positional else None
        if p and p.group(1) == g.part_label:
            return True
    return False


def _auto_group_name(g: ColorGroup) -> str:
    """The name a group gets when it has no custom one: the part name of a part group, the
    nearest colour name otherwise."""
    if g.part:
        return part_name(g.part_label, g.part_plural, g.part_instances)
    return _auto_name(g.albedo_lab)


def _auto_name(lab) -> str:
    return colornames.nearest_name(lab)


#: Advisory finish badges (never automatic behaviour): a group at least SHINY_SHARE of whose
#: area is glint pixels (sensor-clipped or a specular spike of the residual, ``Region.glint``)
#: is 'shiny'; one at least CHROME_SHARE of whose area is chrome-tagged regions is 'chrome'.
#: The broader highlight share (``Region.shiny``, any strong positive residual) is kept for
#: the tooltip only: on glossy paint it is near-universal (98 of the yellow BMW's 113 regions
#: at 15 %), a glint share of 20 % marks two to five groups per photo. The share lives in
#: :mod:`types` because a stored badge is checked against it again on load.
SHINY_SHARE = SHINY_GLINT_SHARE
CHROME_SHARE = 0.5


def _finish(members: list[Region]) -> tuple[float, float, str]:
    """``(shiny, glint, badge)`` of a group: the area-weighted highlight and glint shares of
    its regions and the badge they earn."""
    w = np.array([r.area for r in members], np.float64)
    tot = float(max(w.sum(), 1.0))
    shiny = float((np.array([float(r.shiny) for r in members]) * w).sum() / tot)
    glint = float((np.array([float(r.glint) for r in members]) * w).sum() / tot)
    chrome = float(sum(r.area for r in members if r.chrome) / tot)
    return round(shiny, 4), round(glint, 4), ("chrome" if chrome >= CHROME_SHARE else "shiny" if glint >= SHINY_SHARE else "")


def _group_from(gid: int, members: list[Region], npx: int, name: str | None = None,
                locked: bool = False, is_background: bool = False, ref_lab=None) -> ColorGroup:
    lab_arr = np.array([r.albedo_lab for r in members], np.float32)
    w = np.array([r.area for r in members], np.float64)
    lab = tuple(float(v) for v in _weighted_median(lab_arr, w))
    area = int(w.sum())
    shiny, glint, finish = _finish(members)
    kind, label, plural, n_inst = _part_of(members)
    auto = part_name(label, plural, n_inst) if kind else _auto_name(lab)
    return ColorGroup(
        id=gid, name=name or auto, albedo_lab=lab, albedo_hex=imageio.lab_to_hex(lab),
        area=area, area_frac=area / float(max(1, npx)),
        region_ids=sorted(r.id for r in members),
        hue_family=colornames.hue_family(lab), locked=locked, is_background=is_background,
        shiny=shiny, glint=glint, finish=finish,
        part=kind, part_label=label, part_plural=plural, part_instances=n_inst,
        ref_lab=None if ref_lab is None else tuple(float(v) for v in ref_lab),
    )


_NAME_SUFFIX = re.compile(r"^(.*?) (\d+)$")


def _dedupe_names(groups: list[ColorGroup]) -> None:
    """In place: a name that repeats gets a numeric suffix in id order ("Copper", "Copper 2")."""
    seen: dict[str, int] = {}
    for g in groups:
        base = g.name
        m = _NAME_SUFFIX.match(base)
        auto = _auto_group_name(g)
        if m and base != auto and m.group(1) == auto:
            base = m.group(1)                    # "Copper 2" from an earlier rebuild counts as "Copper"
        n = seen.get(base, 0) + 1
        seen[base] = n
        g.name = base if n == 1 else f"{base} {n}"


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
                                  is_background=bool(c.get("is_background", False)), ref_lab=c.get("ref_lab")))
        for r in members[g]:
            out_regions.append(replace(r, group_id=new_id))
    out_regions.sort(key=lambda r: r.id)
    _dedupe_names(groups)
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


def _mark_background(groups: list[ColorGroup], group_map: np.ndarray,
                     regions: Optional[list[Region]] = None) -> None:
    """Background flags, in place. With ``regions`` carrying a ``backdrop`` decision (the
    matte rule) every group at least BACKDROP_GROUP_SHARE of whose area is backdrop regions
    is flagged and every other group is not, so any number of groups can be background and
    the flags survive a rebuild. Otherwise the border rule: the one group owning more than
    BACKGROUND_BORDER_FRACTION of the image border. A part group is never background (a
    detected part is the object), and neither is a lone colour group (the only group that is
    not a part group): there is nothing to tell it from, and flagged it would leave nothing
    paintable while the background is ignored (a regroup at ``max_groups`` 1). Every rebuild
    goes through here, so the rule holds after each of them."""
    if not groups:
        return
    if regions is not None and any(r.backdrop for r in regions):
        back = {r.id: (r.area if r.backdrop else 0) for r in regions}
        area = {r.id: r.area for r in regions}
        for g in groups:
            tot = sum(area.get(rid, 0) for rid in g.region_ids)
            bg = sum(back.get(rid, 0) for rid in g.region_ids)
            g.is_background = bool(tot > 0 and bg >= BACKDROP_GROUP_SHARE * tot)
    elif len(groups) > 1:
        counts = border_counts(group_map, len(groups))
        total = float(counts.sum())
        if total > 0:
            g = int(np.argmax(counts))
            if counts[g] / total > BACKGROUND_BORDER_FRACTION:
                groups[g].is_background = True
    colour = [g for g in groups if not g.part]
    for g in groups:
        if g.part or (len(colour) == 1 and g is colour[0]):
            g.is_background = False


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


# ---------------------------------------------------------------------- the matte rule

def border_cover(groups: list[ColorGroup], group_map: np.ndarray, cover: float = BORDER_COVER) -> list[int]:
    """The smallest set of groups owning at least ``cover`` of the image-border pixels
    (largest border owners first): the border set of the matte rule. Returns their ids."""
    counts = border_counts(group_map, len(groups)).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return []
    order = np.argsort(-counts, kind="stable")
    out, acc = [], 0.0
    for g in order:
        if counts[g] <= 0:
            break
        out.append(int(g))
        acc += counts[g]
        if acc >= cover * total:
            break
    return out


def _adjacency_sets(labels: np.ndarray, n: int) -> dict[int, set[int]]:
    pairs, _ = adjacency(labels, n)
    adj: dict[int, set[int]] = {}
    for a, b in pairs.tolist():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def _border_connected(regions: list[Region], flagged: set[int], adj: dict[int, set[int]]) -> set[int]:
    """The flagged regions connected to the image border through flagged regions."""
    touch = {r.id for r in regions if r.touches_border and r.id in flagged}
    seen = set(touch)
    stack = list(touch)
    while stack:
        a = stack.pop()
        for b in adj.get(a, ()):
            if b in flagged and b not in seen:
                seen.add(b)
                stack.append(b)
    return seen


def backdrop_decisions(labels: np.ndarray, albedo_lin: np.ndarray, region_info: list[dict], fg: np.ndarray,
                       delta_e: float = 10.0, max_groups: Optional[int] = None) -> np.ndarray:
    """Per region (int8 [N], indexed by region id): BG_MATTE for a region the foreground
    matte ``fg`` (float32 HxW, 1 = object) calls backdrop (matte share <= MATTE_BACK) that is in
    the border set of a plain clustering, next to it, or connected to such a region through
    regions the matte calls backdrop too (the other cars of a showroom, a piece at a time),
    BG_BORDER for another region of the border set that the matte does not call object (share <
    MATTE_OBJECT), BG_OBJECT for the rest (a see-through hole inside the object, which no chain
    of backdrop regions reaches, too). The regions stage stores the decision as
    ``info[i]["bg"]`` for :func:`group_regions`."""
    labels = np.ascontiguousarray(labels, np.int32)
    n = int(labels.max()) + 1
    if fg.shape != labels.shape:
        raise ValueError("the matte must have the label map's shape")
    lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
    regions = _make_regions(labels, region_info, region_medians(labels, lab, n))
    regions = [replace(r, backdrop=False) for r in regions]
    assignment = _cluster_regions(regions, delta_e, max_groups)
    regions0, groups0, gm0 = _finalize(regions, labels, assignment)
    flag0 = set(border_cover(groups0, gm0))
    S = {r.id for r in regions0 if r.group_id in flag0}
    flat = labels.ravel()
    f = np.bincount(flat, weights=np.asarray(fg, np.float64).ravel(), minlength=n) / np.maximum(np.bincount(flat, minlength=n), 1)
    adj = _adjacency_sets(labels, n)
    touch_S = set(S) | {b for a in S for b in adj.get(a, ())}
    # the backdrop the matte calls backdrop reaches on from there through regions it calls backdrop
    # too, none of them enclosed by the object: pieces of the other cars in a showroom (the yellow
    # car's sill, a headlight of the black car behind) sat beyond the first ring of the border set
    # and were decided object, tiny colour rows of the subject's panel. A region at least half of
    # whose boundary is the object (a piece of the subject the matte cut a little tight) is not
    # reached so, and neither is a see-through hole inside the object
    low = {int(r.id) for r in regions if f[r.id] <= MATTE_BACK}
    solid = f >= MATTE_OBJECT
    pairs, cnt = adjacency(labels, n)
    perim = np.zeros(n, np.float64)
    on_obj = np.zeros(n, np.float64)
    for (a, b), c in zip(pairs.tolist(), cnt.tolist()):
        perim[a] += c
        perim[b] += c
        if solid[b]:
            on_obj[a] += c
        if solid[a]:
            on_obj[b] += c
    back = low & touch_S
    stack = list(back)
    while stack:
        a = stack.pop()
        for b in adj.get(a, ()):
            if b in low and b not in back and on_obj[b] < ENCLOSED_SHARE * max(perim[b], 1.0):
                back.add(b)
                stack.append(b)
    kinds = np.full(n, BG_OBJECT, np.int8)
    for r in regions:
        if r.id in back:
            kinds[r.id] = BG_MATTE
        elif r.id in S and f[r.id] < MATTE_OBJECT:
            kinds[r.id] = BG_BORDER
    return kinds


def _cluster_split(regions: list[Region], kinds: dict[int, int], labels: np.ndarray, delta_e: float,
                   max_groups: Optional[int], connectivity: bool = True) -> tuple[list[Region], dict[int, int]]:
    """Backdrop and object regions clustered separately, then the connectivity check: a
    region flagged only through the border set that is not connected to the image border
    through flagged regions leaves the background and joins the nearest unflagged group
    within REJOIN_DELTA_E, else a new group of its own kind. ``max_groups`` caps the
    clustering as a whole: the object side is clustered first with ``max_groups - 1`` (the
    painted thing keeps its colours), the backdrop side with whatever the object left (at
    least one group), and the regions the connectivity check drops join existing groups once
    the cap is reached; with ``max_groups`` 1 everything is one group. (The refinement may
    still add a group per isolated part or decal island, see :mod:`refine`.) Returns the
    regions with their final ``backdrop`` flag and the provisional assignment."""
    b_rows = [r for r in regions if kinds.get(r.id, BG_OBJECT) > BG_OBJECT]
    o_rows = [r for r in regions if kinds.get(r.id, BG_OBJECT) == BG_OBJECT]
    if max_groups is not None and max_groups < 2 and b_rows and o_rows:
        prov = _cluster_regions(regions, delta_e, 1)
        return [replace(r, backdrop=(r.id in {r.id for r in b_rows})) for r in regions], prov
    cap_o = None if max_groups is None else max(1, max_groups - (1 if b_rows else 0))
    co = _cluster_regions(o_rows, delta_e, cap_o) if o_rows else {}
    n_o = (max(co.values()) + 1) if co else 0
    cap_b = None if max_groups is None else max(1, max_groups - n_o)
    cb = _cluster_regions(b_rows, delta_e, cap_b) if b_rows else {}
    off = (max(cb.values()) + 1) if cb else 0
    prov = {**cb, **{rid: off + c for rid, c in co.items()}}
    flagged_prov = set(cb.values())
    flagged = set(cb)
    kept = flagged
    if connectivity and flagged:
        n = int(labels.max()) + 1
        adj = _adjacency_sets(labels, n)
        kept = _border_connected(regions, flagged, adj) | {rid for rid in flagged if kinds.get(rid) == BG_MATTE}
    dropped = sorted(flagged - kept, key=lambda rid: -next(r.area for r in regions if r.id == rid))
    if dropped:
        members: dict[int, list[Region]] = {}
        for r in regions:
            if r.id not in dropped:
                members.setdefault(prov[r.id], []).append(r)
        cand = [g for g in members if g not in flagged_prov]
        cent = {g: _weighted_median(np.array([r.albedo_lab for r in members[g]], np.float32),
                                    np.array([r.area for r in members[g]], np.float64)) for g in cand}
        by_id = {r.id: r for r in regions}
        left: list[Region] = []
        for rid in dropped:
            best = None
            if cand:
                labs = np.array([cent[g] for g in cand], np.float64)
                de = imageio.delta_e(np.repeat(np.asarray(by_id[rid].albedo_lab, np.float64)[None], len(cand), 0), labs)
                j = int(np.argmin(de))
                if de[j] < REJOIN_DELTA_E:
                    best = cand[j]
            if best is None:
                left.append(by_id[rid])
            else:
                prov[rid] = best
        if left:
            budget = None if max_groups is None else max_groups - len(members)
            if budget is not None and budget < 1:
                # the cap is spent: each joins the nearest group that is not backdrop (or,
                # failing that, the nearest group at all)
                pool = cand or list(members)
                if not cent:
                    cent = {g: _weighted_median(np.array([r.albedo_lab for r in members[g]], np.float32),
                                                np.array([r.area for r in members[g]], np.float64)) for g in pool}
                labs = np.array([cent[g] for g in pool], np.float64)
                for r in left:
                    de = imageio.delta_e(np.repeat(np.asarray(r.albedo_lab, np.float64)[None], len(pool), 0), labs)
                    prov[r.id] = pool[int(np.argmin(de))]
            else:
                base = max(prov.values()) + 1
                for rid, c in _cluster_regions(left, delta_e, budget).items():
                    prov[rid] = base + c
    regions = [replace(r, backdrop=(r.id in kept)) for r in regions]
    return regions, prov


# ---------------------------------------------------------------------- one paint under different light

@dataclass(frozen=True)
class LitParams:
    """Knobs of :func:`absorb_lit` (the values the round-3 experiment chose on six analysed
    photos and checked on fourteen samples)."""
    anchor_chroma: float = 30.0    # an anchor is a non-background, unlocked group above this CIELAB chroma ...
    anchor_min_px: int = 2000      # ... of at least this many pixels
    cand_chroma: float = 12.0      # a candidate group is chromatic: lightness-normalised chroma at least this ...
    cand_raw_chroma: float = 18.0  # ... and raw chroma at least this (a mixed edge band of the paint's hue is not)
    alb_tol: float = 8.0           # CIEDE2000 (kC = 2) between the lightness-normalised albedo means
    pho_tol: float = 7.0           # CIEDE2000 (kC = 2) between the lightness-normalised photo means (0 disables)
    kC: float = 2.0
    mat_c_ratio: float = 0.6       # veto: candidate chroma below this share of the anchor's ...
    mat_l_margin: float = 10.0     # ... and more than this many L darker (the material lock's rule)
    # With ``sheen``, a photo that fails pho_tol still passes when it is the anchor's photo under a
    # white sheen (a surface facing a bright ceiling): lighter by sheen_l, the hue of the
    # lightness-normalised photo colour within sheen_hue, and duller. The Torana's roof and boot lid
    # (albedo within 5 of the body, photo 16 L lighter and half as chromatic, the same hue) stayed
    # a group of their own at photo dE 8.1, and a navy repaint of the body left them orange. The
    # clustering runs without it; the refinement merges the sheen pairs last (``sheen_only``,
    # :data:`SHEEN_LIT`), after the edge snap and the junk pruning: merged before them, the paint's
    # new outline moved the snap at the Torana's flare, and the flare shadow, 31 px smaller, no
    # longer passed the shadow test.
    sheen: bool = False
    sheen_only: bool = False
    sheen_l: float = 6.0
    sheen_hue: float = 12.0


DEFAULT_LIT = LitParams()
#: The refinement's last merge: one paint under a white sheen (see LitParams.sheen).
SHEEN_LIT = LitParams(sheen=True, sheen_only=True)


def normalise_lab(lab: np.ndarray, L_ref: float = 50.0) -> np.ndarray:
    """Each Lab colour rescaled to lightness ``L_ref``: a grey shading leak scales L + 16, a
    and b by the same factor above CIELAB's toe, so a / (L + 16) and b / (L + 16) are what
    the same material keeps under different light (the dichromatic body colour)."""
    lab = np.asarray(lab, np.float64)
    k = (L_ref + 16.0) / np.clip(lab[..., 0] + 16.0, 8.0, None)
    out = np.empty_like(lab)
    out[..., 0] = L_ref
    out[..., 1] = lab[..., 1] * k
    out[..., 2] = lab[..., 2] * k
    return out


def _de_kc(a: np.ndarray, b: np.ndarray, kC: float) -> float:
    from skimage import color as skcolor
    return float(skcolor.deltaE_ciede2000(np.asarray(a, np.float64)[None], np.asarray(b, np.float64)[None], kC=kC)[0])


def _chroma(lab) -> float:
    return float(np.hypot(float(lab[1]), float(lab[2])))


def _sheen_of(pho_g, pho_a, params: LitParams) -> bool:
    """True when photo colour ``pho_g`` is ``pho_a`` under a white sheen: lighter by at least
    ``sheen_l``, its lightness-normalised hue within ``sheen_hue`` and no more chromatic."""
    ng, na = normalise_lab(pho_g), normalise_lab(pho_a)
    cg, ca = _chroma(ng), _chroma(na)
    if float(pho_g[0]) < float(pho_a[0]) + params.sheen_l or cg > ca or cg < 1.0 or ca < 1.0:
        return False
    hg = np.degrees(np.arctan2(ng[2], ng[1]))
    ha = np.degrees(np.arctan2(na[2], na[1]))
    return abs((hg - ha + 180.0) % 360.0 - 180.0) <= params.sheen_hue


def absorb_lit(regions: list[Region], groups: list[ColorGroup], labels: np.ndarray, albedo_lab: np.ndarray,
               photo_lab: np.ndarray, params: LitParams = DEFAULT_LIT
               ) -> tuple[list[Region], list[ColorGroup], np.ndarray, list[dict]]:
    """Merge every chromatic group that is the same paint as a larger anchor under different
    light into that anchor (see the module docstring): the candidate's and the anchor's
    lightness-normalised albedo means are within ``alb_tol`` (CIEDE2000, chroma weight
    ``kC``), their lightness-normalised photo colours within ``pho_tol``, and the material
    veto (duller than ``mat_c_ratio`` of the anchor's chroma and more than ``mat_l_margin``
    darker: a gold caliper, a mirror stalk tinted by the paint's bounce) does not fire. Every
    test is against the anchor's own colour, never a drifting centroid; a chain of moves
    resolves to its final anchor. Locked, background, part and neutral groups never move or
    absorb. Returns the rebuilt ``(regions, groups, group_map)`` (flags and custom names kept,
    ids by area) and a log of the moves."""
    n = int(labels.max()) + 1
    pmeds = region_medians(np.ascontiguousarray(labels, np.int32), photo_lab, n)
    alab = np.array([r.albedo_lab for r in regions], np.float64)
    plab = np.array([pmeds[r.id] for r in regions], np.float64)
    area = np.array([r.area for r in regions], np.float64)
    row = {r.id: k for k, r in enumerate(regions)}
    feats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for g in groups:
        idx = np.array([row[rid] for rid in g.region_ids if rid in row], np.int64)
        w = area[idx]
        ok = np.isfinite(plab[idx]).all(axis=1)
        feats[g.id] = (
            (alab[idx] * w[:, None]).sum(0) / max(w.sum(), 1.0),
            (plab[idx][ok] * w[ok, None]).sum(0) / max(w[ok].sum(), 1.0) if ok.any() else np.array(g.albedo_lab, np.float64),
        )
    anchors = [g for g in groups if not g.is_background and not g.locked and not g.part
               and _chroma(g.albedo_lab) >= params.anchor_chroma and g.area >= params.anchor_min_px]
    anchors.sort(key=lambda g: -g.area)
    largest = anchors[0].area if anchors else 0
    log: list[dict] = []
    for g in groups:
        if g.is_background or g.locked or g.part or (g in anchors and g.area >= largest):
            continue
        lab_g, pho_g = feats[g.id]
        if _chroma(normalise_lab(lab_g)) < params.cand_chroma or _chroma(lab_g) < params.cand_raw_chroma:
            continue
        best = None
        for a in anchors:
            if a.id == g.id or a.area <= g.area:
                continue
            lab_a, pho_a = feats[a.id]
            d_alb = _de_kc(normalise_lab(lab_g), normalise_lab(lab_a), params.kC)
            if d_alb > params.alb_tol:
                continue
            if _chroma(lab_g) < params.mat_c_ratio * _chroma(lab_a) and lab_g[0] < lab_a[0] - params.mat_l_margin:
                continue                                  # another material in the paint's hue
            d_pho = _de_kc(normalise_lab(pho_g), normalise_lab(pho_a), params.kC)
            photo_ok = not params.pho_tol or d_pho <= params.pho_tol
            sheen_ok = params.sheen and not photo_ok and _sheen_of(pho_g, pho_a, params)
            if not (sheen_ok if params.sheen_only else (photo_ok or sheen_ok)):
                continue
            if best is None or d_alb < best[0]:
                best = (d_alb, a, d_pho)
        if best is not None:
            d_alb, a, d_pho = best
            log.append({"group": int(g.id), "name": g.name, "area": int(g.area), "into": int(a.id),
                        "into_name": a.name, "d_alb": round(d_alb, 2), "d_pho": round(d_pho, 2)})
    if not log:
        return regions, groups, _group_map(regions, labels, len(groups)), log
    into = {m["group"]: m["into"] for m in log}

    def final(gid: int) -> int:
        seen = set()
        while gid in into and gid not in seen:
            seen.add(gid)
            gid = into[gid]
        return gid

    assignment = {r.id: r.group_id for r in regions}
    by_id = {g.id: g for g in groups}
    for m in log:
        target = final(m["group"])
        m["final"] = int(target)
        for rid in by_id[m["group"]].region_ids:
            assignment[rid] = target
    carry = {g.id: {"name": _keep_name(g), "locked": g.locked, "is_background": g.is_background} for g in groups}
    regions, groups, group_map = _finalize(regions, labels, assignment, carry)
    _mark_background(groups, group_map, regions)
    _check_state(regions, groups)
    return regions, groups, group_map, log


def photo_lab_of(image_rgb_u8: np.ndarray) -> np.ndarray:
    """CIELAB of the photograph itself (float32 HxWx3), the input of :func:`absorb_lit`."""
    return imageio.rgb_to_lab(imageio.to_float(image_rgb_u8)).astype(np.float32)


# ---------------------------------------------------------------------- public API

def group_regions(labels: np.ndarray, albedo_lin: np.ndarray, region_info: list[dict],
                  max_groups: Optional[int] = None, delta_e: float = 10.0,
                  photo_rgb_u8: Optional[np.ndarray] = None) -> GroupState:
    """Cluster regions by median albedo into color groups.

    Returns `(regions, groups, group_map)`: one `Region` per label id (ids equal to the
    label values), `ColorGroup`s sorted by area descending with ids 0..G-1, and an int32
    HxW map of group ids. Every region belongs to exactly one group; group albedo is the
    area-weighted median of its regions' median albedo. `delta_e` is the CIEDE2000 linkage
    threshold, `max_groups` an optional hard cap on the clustering's group count (with a
    backdrop side the object side is clustered first and keeps up to `max_groups - 1`
    groups, the backdrop takes the rest, at least one); the refinement that follows in the
    pipeline may add a locked group per isolated part and a group per decal island.

    Background: when ``region_info`` carries the matte rule's decisions (``"bg"``, see
    :func:`backdrop_decisions`) the backdrop and the object regions are clustered
    separately, every backdrop group is flagged ``is_background`` and no object region
    lands in one; otherwise `is_background` marks the group owning more than 35 % of the
    image border (at most one). With the photo (``photo_rgb_u8``) the pieces of one paint
    under different light are merged afterwards (:func:`absorb_lit`).
    """
    n = int(labels.max()) + 1
    if n <= 0 or (labels < 0).any():
        raise ValueError("labels must be a complete 0..N-1 partition")
    lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
    meds = region_medians(labels, lab, n)
    regions = _make_regions(labels, region_info, meds)
    kinds = {int(d.get("id", k)): int(d.get("bg", BG_OBJECT) or 0) for k, d in enumerate(region_info or [])}
    for r in regions:
        if r.part_kind:
            kinds[r.id] = BG_OBJECT                  # a detected part is the object
    split = any(kinds.get(r.id, BG_OBJECT) > BG_OBJECT for r in regions if not r.part_kind)
    regions, groups, group_map = _cluster_with_parts(regions, kinds if split else None, labels, delta_e, max_groups)
    if photo_rgb_u8 is not None:
        regions, groups, group_map, _ = absorb_lit(regions, groups, labels, lab, photo_lab_of(photo_rgb_u8))
    _check_state(regions, groups)
    return regions, groups, group_map


def _cluster_with_parts(regions: list[Region], kinds: Optional[dict[int, int]], labels: np.ndarray, delta_e: float,
                        max_groups: Optional[int], connectivity: bool = True) -> GroupState:
    """The colour clustering of every region that is not a detected part (backdrop and object
    apart when ``kinds`` gives the backdrop decisions, :func:`_cluster_split`), plus one group
    per part kind holding that kind's part regions (outside the ``max_groups`` cap), built and
    background-marked like :func:`group_regions` builds them; part groups are never background."""
    parts = {r.id: r.part_kind for r in regions if r.part_kind}
    rest = [r for r in regions if not r.part_kind]
    if kinds is not None:
        rest, prov = _cluster_split(rest, kinds, labels, delta_e, max_groups, connectivity=connectivity)
    else:
        prov = _cluster_regions(rest, delta_e, max_groups) if rest else {}
    base = (max(prov.values()) + 1) if prov else 0
    kid = {k: base + i for i, k in enumerate(sorted(set(parts.values())))}
    assignment = dict(prov)
    for rid, k in parts.items():
        assignment[rid] = kid[k]
    by_id = {r.id: r for r in rest}
    regions = [by_id.get(r.id, r) for r in regions]
    regions, groups, group_map = _finalize(regions, labels, assignment)
    # (a part group and a lone colour group, max_groups 1, are never background: _mark_background)
    if kinds is not None:
        _mark_background(groups, group_map, regions)
    else:
        _mark_background(groups, group_map)
    return regions, groups, group_map


def regroup(regions: list[Region], labels: np.ndarray, albedo_lin: np.ndarray,
            max_groups: Optional[int] = None, delta_e: float = 10.0,
            photo_rgb_u8: Optional[np.ndarray] = None) -> GroupState:
    """Re-cluster existing regions with new parameters without touching the label map.

    Uses the regions' stored median albedo (no pixel work), so it is cheap. Region ids
    are unchanged; user flags on the previous groups are discarded since the groups are
    redefined. Regions the analysis called backdrop are clustered apart from the object's
    and their groups flagged; detected parts keep one group per kind; with the photo the lit
    and shadowed pieces of one paint are merged (:func:`absorb_lit`). Returns the same tuple
    as `group_regions`.
    """
    kinds = {r.id: (BG_MATTE if r.backdrop else BG_OBJECT) for r in regions} \
        if any(r.backdrop and not r.part_kind for r in regions) else None
    regions, groups, group_map = _cluster_with_parts(list(regions), kinds, labels, delta_e, max_groups,
                                                     connectivity=False)
    if photo_rgb_u8 is not None:
        lab = imageio.linear_to_lab(np.ascontiguousarray(albedo_lin, dtype=np.float32))
        regions, groups, group_map, _ = absorb_lit(regions, groups, labels, lab, photo_lab_of(photo_rgb_u8))
    _check_state(regions, groups)
    return regions, groups, group_map


def _carry_all(groups: list[ColorGroup]) -> dict[int, dict[str, Any]]:
    """Every group's flags, name and engine reference, for an edit that keeps them (the groups
    whose regions the edit changes drop the reference: see ``ColorGroup.ref_lab``)."""
    return {g.id: {"name": g.name, "locked": g.locked, "is_background": g.is_background, "ref_lab": g.ref_lab}
            for g in groups}


def _keep_name(g: ColorGroup) -> str | None:
    """The user's custom name if they set one, else None so the name is recomputed. An
    automatic name with a duplicate suffix ("Copper 2") is not a custom name, and neither is a
    part group's kind name ("Wheel rims"); an instance name ("Mirror (left)") is kept, the
    way a split named it, until a merge joins the instances again (:func:`merge_groups`)."""
    auto = _auto_name(g.albedo_lab)
    if g.name == auto or _is_auto_part_name(g, g.name, positional=False):
        return None
    m = _NAME_SUFFIX.match(g.name)
    if m and m.group(1) == auto:
        return None
    return g.name


def merge_groups(groups: list[ColorGroup], regions: list[Region], group_map: np.ndarray,
                 labels: np.ndarray, ids: list[int], into: Optional[int] = None) -> GroupState:
    """Merge the groups listed in `ids` into one (the lowest id survives, or ``into`` when it is
    one of them: the group the others were merged into, whose name the result keeps).

    Other groups keep their relative order and are renumbered to stay contiguous; region
    ids do not change. The merged group is locked / background if any member was, and
    keeps a custom name if the surviving group had one; instances of one part kind merged
    back together take the kind's name again ("Mirror (left)" + "Mirror (right)" =
    "Mirrors"). Fewer than two valid ids is a no-op that still returns fresh copies.
    """
    ids = sorted({int(i) for i in ids if 0 <= int(i) < len(groups)})
    target = ids[0] if ids else None
    if into is not None and int(into) in ids:
        target = int(into)
    assignment = {}
    for r in regions:
        assignment[r.id] = target if (target is not None and r.group_id in ids) else r.group_id
    carry = _carry_all(groups)
    if target is not None:
        merged = [groups[i] for i in ids]
        instances = len(merged) > 1 and len({g.part for g in merged}) == 1 and merged[0].part \
            and all(_is_auto_part_name(g, g.name) for g in merged)
        carry[target] = {
            "name": None if instances else _keep_name(groups[target]),
            "locked": any(g.locked for g in merged),
            "is_background": any(g.is_background for g in merged),
            "ref_lab": None,
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
    contiguous 0..N-1 partition). New groups get the next free ids and inherit the parent's
    ``locked`` and ``is_background`` flags (a background group's halves are both
    background); existing group ids and flags are unchanged. If no split is possible - including when the group is
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
    # the halves keep the parent's lock and background flag (the user's intent, and a
    # background group's regions stay backdrop): only the name is recomputed
    for i in range(k - 1):
        carry[next_gid + i] = {"name": None, "locked": groups[gid].locked, "is_background": groups[gid].is_background}

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
            _split_carry(carry, groups[gid])
            regions, groups, group_map = _finalize(regions, labels, assignment, carry, sort_by_area=False)
            _name_split_pieces(groups, regions, {gid} | set(range(next_gid, len(groups))))
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
    # a piece cut off lettering, a named or recovered part or a detected part keeps its parent's
    # source (and with it the exemptions and the island treatment those sources carry)
    for r in regions + [replace(parent, id=rid, source=parent.source if parent.source in SPLIT_KEEPS_SOURCE else "split")
                        for rid, parent in new_ids]:
        if areas[r.id] == 0:
            continue
        lab_t = tuple(float(v) for v in meds[r.id])
        refreshed.append(replace(r, area=int(areas[r.id]), bbox=tuple(int(v) for v in bb[r.id]),
                                 albedo_lab=lab_t, albedo_hex=imageio.lab_to_hex(lab_t),
                                 touches_border=bool(border[r.id])))
    _split_carry(carry, groups[gid])
    regions, groups, group_map = _finalize(refreshed, labels, assignment, carry, sort_by_area=False)
    _name_split_pieces(groups, regions, {gid} | set(range(next_gid, len(groups))))
    _check_state(regions, groups)
    return regions, groups, group_map


def _split_carry(carry: dict[int, dict[str, Any]], g: ColorGroup) -> None:
    """In place: the group a split cut keeps a custom name, not an automatic one (a part group's
    plural goes when it keeps one instance: "Tyres" -> "Tyre"), and drops its engine reference."""
    carry[g.id] = dict(carry.get(g.id, {}), name=_keep_name(g), ref_lab=None)


def _name_split_pieces(groups: list[ColorGroup], regions: list[Region], ids: set[int]) -> None:
    """In place: when a colour split of a part group left two or more pieces that are each one
    instance of the same kind with automatic names, they are named by where they sit, as
    :func:`split_instances` names them ("Tyre (left)", "Tyre (right)")."""
    pieces = [g for g in groups if g.id in ids and g.part and g.part_instances == 1 and _is_auto_part_name(g, g.name)]
    if len(pieces) < 2 or len({g.part for g in pieces}) != 1:
        return
    cents = []
    for g in pieces:
        inst = group_instances(g, regions)
        if not inst:
            return
        cents.append(inst[0]["centroid"])
    for g, nm in zip(pieces, _instance_names_for(pieces[0], regions, cents)):
        g.name = nm
    _dedupe_names(groups)


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
        carry[g]["ref_lab"] = None
    regions, groups, group_map = _finalize(regions, labels, assignment, carry, sort_by_area=False)
    _check_state(regions, groups)
    return regions, groups, group_map


# ---------------------------------------------------------------------- detected parts

def part_regions(regions: list[Region]) -> dict[int, str]:
    """region id -> part kind of every detected-part region."""
    return {r.id: r.part_kind for r in regions if r.part_kind}


def enforce_parts(regions: list[Region], groups: list[ColorGroup], labels: np.ndarray) -> GroupState:
    """Every part region in its kind's group (the group holding most of that kind's part
    area) and no part region of another kind there. Returns the state unchanged (rebuilt
    group map) when that already holds, which is the normal case: every colour step leaves the
    part regions alone; a piece a pixel-level split cut off a part region, or a region carried
    over without an origin, is put back. Flags and custom names are kept."""
    kinds = part_regions(regions)
    if not kinds:
        return regions, groups, _group_map(regions, labels, len(groups))
    by_kind: dict[str, dict[int, int]] = {}
    for r in regions:
        if r.part_kind:
            by_kind.setdefault(r.part_kind, {})
            by_kind[r.part_kind][r.group_id] = by_kind[r.part_kind].get(r.group_id, 0) + r.area
    target: dict[str, int] = {}
    fresh = max([g.id for g in groups] + [-1]) + 1
    taken: set[int] = set()
    part_of = {g.id: g.part for g in groups}
    for kind in sorted(by_kind, key=lambda k: -sum(by_kind[k].values())):
        # the kind's own part group first, then the group holding most of the kind's area
        for gid, _ in sorted(by_kind[kind].items(), key=lambda t: (part_of.get(t[0]) != kind, -t[1], t[0])):
            if gid not in taken:
                target[kind] = gid
                break
        else:
            target[kind] = fresh
            fresh += 1
        taken.add(target[kind])
    assignment = {r.id: (target[r.part_kind] if r.part_kind else r.group_id) for r in regions}
    if all(assignment[r.id] == r.group_id for r in regions):
        return regions, groups, _group_map(regions, labels, len(groups))
    carry = {g.id: {"name": _keep_name(g), "locked": g.locked, "is_background": g.is_background} for g in groups}
    regions, groups, group_map = _finalize(regions, labels, assignment, carry)
    _mark_background(groups, group_map, regions)          # a part group is never background
    _check_state(regions, groups)
    return regions, groups, group_map


#: Part kinds that come in pairs along a vehicle's length (front and rear wheel, caliper, disc,
#: door, footpeg, seat, a bike's header and muffler): their instances side by side, at least
#: AXLE_MIN_SPREAD of the wheels' span apart, are named front / rear on a vehicle (a car's twin
#: exhaust tips, side by side at the rear, keep left / right).
AXLE_KINDS = ("rim", "tyre", "brake_caliper", "brake_disc", "door_handle", "footpeg", "seat", "exhaust")
AXLE_MIN_SPREAD = 0.2
#: The kinds whose position gives away which end of a vehicle is its front (weights): the front
#: cues sit ahead of the wheels' midpoint, the rear cues behind it.
FRONT_CUES = {"grille": 2.0, "fog_lamp": 2.0, "fork": 2.0, "grip": 1.0, "lever": 1.0, "handlebar": 1.0,
              "mirror": 0.5}
REAR_CUES = {"sprocket": 2.0, "spoiler": 2.0, "exhaust": 1.0, "seat": 0.5, "saddle": 0.5}
#: The cues must agree by at least this (their weighted offsets from the wheels' midpoint, in half
#: the wheelbase) before a side is called the front.
FRONT_MIN_SCORE = 0.3


def vehicle_front(regions: list[Region]) -> int:
    """Which side of the photo a vehicle's front is on, from its detected parts: -1 left, +1
    right, 0 unknown (fewer than two wheels, or cues that do not agree). The wheels (rim / tyre
    part regions, two instances at least: with one, the Corvette's mirror behind its front wheel
    read as a front cue ahead of it) give the midpoint and the wheelbase; every front cue
    (FRONT_CUES: a grille, the fork, the grips) ahead of the midpoint and every rear cue
    (REAR_CUES: the sprocket, the exhaust, the seat) behind it votes, weighted by its offset in
    half wheelbases."""
    wheels = [r for r in regions if r.part_kind in ("rim", "tyre")]
    if max(len({r.part_instance for r in wheels if r.part_kind == k}) for k in ("rim", "tyre")) < 2:
        return 0
    cx = [((r.bbox[0] + r.bbox[2]) / 2.0, r.area) for r in wheels]
    mid = sum(x * a for x, a in cx) / max(sum(a for _, a in cx), 1)
    lo = min(r.bbox[0] for r in wheels)
    hi = max(r.bbox[2] for r in wheels)
    half = max((hi - lo) / 2.0, 1.0)
    score = 0.0
    for r in regions:
        w = FRONT_CUES.get(r.part_kind, 0.0) - REAR_CUES.get(r.part_kind, 0.0)
        if w:
            score += w * float(np.clip(((r.bbox[0] + r.bbox[2]) / 2.0 - mid) / half, -1.5, 1.5))
    if abs(score) < FRONT_MIN_SCORE:
        return 0
    return 1 if score > 0 else -1


def instance_names(label: str, centroids: list[tuple[float, float]], front: int = 0, axle: bool = False) -> list[str]:
    """Names for the instances of one part kind by where they sit in the photo: two side by
    side are "<label> (left)" / "(right)", two stacked "(upper)" / "(lower)", more are
    numbered left to right ("<label> (1)", "(2)", ...). With ``axle`` (a kind of AXLE_KINDS on a
    vehicle, whose ``front`` side is known: -1 left, +1 right; :func:`vehicle_front`) two side by
    side are "(front)" / "(rear)" and more are numbered from the front; an axle kind on a vehicle
    whose front is unknown is numbered left to right (in a side view left and right are the
    front and the rear, and read as the car's own sides)."""
    n = len(centroids)
    if n <= 1:
        return [label] * n
    xs = [c[0] for c in centroids]
    ys = [c[1] for c in centroids]
    horizontal = n != 2 or abs(xs[0] - xs[1]) >= abs(ys[0] - ys[1])
    if axle and horizontal:
        if front and n == 2:
            ahead = (0 if xs[0] < xs[1] else 1) if front < 0 else (0 if xs[0] > xs[1] else 1)
            out = ["", ""]
            out[ahead] = f"{label} (front)"
            out[1 - ahead] = f"{label} (rear)"
            return out
        order = sorted(range(n), key=lambda i: (-xs[i] if front > 0 else xs[i], ys[i]))
        out = [""] * n
        for rank, i in enumerate(order):
            out[i] = f"{label} ({rank + 1})"
        return out
    if n == 2:
        first = (0 if xs[0] < xs[1] else 1) if horizontal else (0 if ys[0] < ys[1] else 1)
        words = ("left", "right") if horizontal else ("upper", "lower")
        out = ["", ""]
        out[first] = f"{label} ({words[0]})"
        out[1 - first] = f"{label} ({words[1]})"
        return out
    out = [""] * n
    for rank, i in enumerate(sorted(range(n), key=lambda i: (xs[i], ys[i]))):
        out[i] = f"{label} ({rank + 1})"
    return out


def _instance_names_for(g: ColorGroup, regions: list[Region], centroids: list[tuple[float, float]]) -> list[str]:
    """:func:`instance_names` for part group ``g``'s instances, front / rear for an axle kind on
    a vehicle (:func:`vehicle_front` of ``regions``) whose instances lie apart along it."""
    wheels = [r for r in regions if r.part_kind in ("rim", "tyre")]
    axle = bool(wheels) and g.part in AXLE_KINDS
    if axle and len(centroids) >= 2:
        span = max(r.bbox[2] for r in wheels) - min(r.bbox[0] for r in wheels)
        xs = [c[0] for c in centroids]
        axle = max(xs) - min(xs) >= AXLE_MIN_SPREAD * max(span, 1)
    return instance_names(g.part_label or g.name, centroids, front=vehicle_front(regions) if axle else 0, axle=axle)


def group_instances(g: ColorGroup, regions: list[Region]) -> list[dict]:
    """The part instances of a part group, in instance order: ``[{"instance", "regions",
    "area", "centroid": (x, y)}]`` (area-weighted centres of the regions' boxes). The group's
    regions that are not part regions of its kind (a shadow the junk pruning put into the
    part) go with the instance they touch the most by box overlap, else the first."""
    by_id = {r.id: r for r in regions}
    inst: dict[int, dict] = {}
    loose: list[Region] = []
    for rid in g.region_ids:
        r = by_id.get(rid)
        if r is None:
            continue
        if not g.part or r.part_kind != g.part:
            loose.append(r)
            continue
        rec = inst.setdefault(r.part_instance, {"instance": r.part_instance, "regions": [], "area": 0, "sx": 0.0,
                                                "sy": 0.0, "boxes": []})
        rec["regions"].append(r.id)
        rec["area"] += r.area
        rec["sx"] += r.area * (r.bbox[0] + r.bbox[2]) / 2.0
        rec["sy"] += r.area * (r.bbox[1] + r.bbox[3]) / 2.0
        rec["boxes"].append(r.bbox)
    out = [inst[k] for k in sorted(inst)]
    for r in loose:
        if not out:
            break
        def overlap(rec) -> int:
            best = 0
            for b in rec["boxes"]:
                iw = max(0, min(b[2], r.bbox[2]) - max(b[0], r.bbox[0]))
                ih = max(0, min(b[3], r.bbox[3]) - max(b[1], r.bbox[1]))
                best = max(best, iw * ih)
            return best
        home = max(out, key=lambda rec: (overlap(rec), -rec["instance"]))
        home["regions"].append(r.id)
    return [{"instance": rec["instance"], "regions": sorted(rec["regions"]), "area": int(rec["area"]),
             "centroid": (rec["sx"] / max(rec["area"], 1), rec["sy"] / max(rec["area"], 1))} for rec in out]


def split_instances(groups: list[ColorGroup], regions: list[Region], group_map: np.ndarray,
                    labels: np.ndarray, gid: int) -> GroupState:
    """Split part group ``gid`` into one group per part instance (:func:`group_instances`),
    named by position (:func:`instance_names`), each keeping the group's lock (a part is never
    background). No colour clustering is involved, so two same-coloured calipers split
    cleanly; Merge joins them again under the kind's name. The first instance keeps ``gid``,
    the others get the next free ids; region ids and every other group are unchanged. A group
    with fewer than two instances comes back unchanged (as copies)."""
    gid = int(gid)
    if not (0 <= gid < len(groups)):
        raise ValueError(f"no group {gid}")
    g = groups[gid]
    carry = _carry_all(groups)
    assignment = {r.id: r.group_id for r in regions}
    inst = group_instances(g, regions) if g.part else []
    if len(inst) < 2:
        return _finalize(regions, labels, assignment, carry, sort_by_area=False)
    names = _instance_names_for(g, regions, [i["centroid"] for i in inst])
    next_gid = len(groups)
    for k, (i, nm) in enumerate(zip(inst, names)):
        new = gid if k == 0 else next_gid + k - 1
        for rid in i["regions"]:
            assignment[rid] = new
        # every instance is painted from the part's albedo, so the split alone changes no pixel
        carry[new] = {"name": nm, "locked": g.locked, "is_background": False,
                      "ref_lab": g.ref_lab if g.ref_lab is not None else g.albedo_lab}
    regions, groups, group_map = _finalize(regions, labels, assignment, carry, sort_by_area=False)
    _check_state(regions, groups)
    return regions, groups, group_map


# ---------------------------------------------------------------------- panel view

#: An object group below this share of the object area may be minor (partkit's TINY_FRAC) ...
MINOR_FRAC = 0.004
#: ... when its albedo is its parent's under other light: within MINOR_DE (CIEDE2000) of it, or
#: of the same body colour (lightness-normalised (a, b) within MINOR_CAST: a shadow is darker
#: but keeps the paint's body colour; partkit's junk test). A tiny group of a colour of its own
#: is a real small part (the Ducati's gold preload adjuster, 690 px next to silver at dE 25; the
#: Torana's amber tail light, dE 15 from the red paint), which the panel keeps among the colours.
MINOR_DE = 12.0
MINOR_CAST = 12.0
#: The body-colour test (MINOR_CAST) needs a body colour: both colours at least this chroma. Two
#: neutrals always have close lightness-normalised (a, b), so white next to black, or silver next
#: to charcoal, passed it at any lightness (a paper card in the coupe's window, L 62, was "minor
#: next to" a charcoal L 37 at dE 27); between neutrals only MINOR_DE decides.
MINOR_CAST_MIN_C = 8.0
#: Width (px) of the ring whose owner is a minor group's parent.
PARENT_RING_PX = 5
#: Region sources that are lettering: a tiny lettering group stays a colour row (decals get repainted).
LETTERING_SOURCES = ("text", "named")
#: The version of the panel view's rule (:func:`annotate_groups`) a job's groups were annotated
#: with; the pipeline stores it with the groups and annotates a job of an older rule again when
#: it is served (1: size alone; 2: tiny and its parent's colour under other light; 3: the body
#: colour test only between two colours).
PANEL_RULE = 3


def _lighting_variant(lab, parent_lab) -> bool:
    """True when colour ``lab`` is ``parent_lab`` under other light: within MINOR_DE of it, or,
    when both have a body colour (chroma >= MINOR_CAST_MIN_C), of the same body colour
    (lightness-normalised (a, b) within MINOR_CAST)."""
    a = np.asarray(lab, np.float32)[None]
    b = np.asarray(parent_lab, np.float32)[None]
    if float(imageio.delta_e(a, b)[0]) <= MINOR_DE:
        return True
    if min(_chroma(a[0]), _chroma(b[0])) < MINOR_CAST_MIN_C:
        return False
    return float(np.linalg.norm(normalise_lab(a)[0, 1:] - normalise_lab(b)[0, 1:])) <= MINOR_CAST


def _ring_owner(group_map: np.ndarray, gid: int, exclude: set[int]) -> int:
    import cv2
    m = group_map == gid
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return -1
    pad = PARENT_RING_PX + 2
    y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad + 1)
    sub = m[y0:y1, x0:x1]
    k = np.ones((2 * PARENT_RING_PX + 1, 2 * PARENT_RING_PX + 1), np.uint8)
    ring = cv2.dilate(sub.astype(np.uint8), k).astype(bool) & ~sub
    ids, cnt = np.unique(group_map[y0:y1, x0:x1][ring], return_counts=True)
    keep = [(c, i) for i, c in zip(ids.tolist(), cnt.tolist()) if i not in exclude]
    return int(max(keep)[1]) if keep else -1


def annotate_groups(groups: list[ColorGroup], regions: list[Region], group_map: np.ndarray) -> list[ColorGroup]:
    """In place (and returned): the Groups panel's view of each group. A group that is not
    background, not a part group and holds no lettering (sources LETTERING_SOURCES), covering
    less than MINOR_FRAC of the object area (the area of the non-background groups), is
    ``minor`` when it is its ``parent`` under other light (:func:`_lighting_variant`); the
    parent is the non-background group of at least that size owning most of a PARENT_RING_PX
    ring around it. A tiny group of a colour of its own, or with no such neighbour, is no minor
    group (a real small part stays among the colours). Every other group gets ``minor`` False
    and ``parent`` -1. Nothing is merged, hidden or locked by this."""
    object_px = sum(int(g.area) for g in groups if not g.is_background) or sum(int(g.area) for g in groups)
    src = {r.id: r.source for r in regions}
    gm = np.asarray(group_map)
    by_id = {g.id: g for g in groups}
    tiny: set[int] = set()
    for g in groups:
        g.minor, g.parent = False, -1
        if g.is_background or g.part or any(src.get(rid) in LETTERING_SOURCES for rid in g.region_ids):
            continue
        if g.area < MINOR_FRAC * object_px:
            tiny.add(g.id)
    bg = {g.id for g in groups if g.is_background}
    for g in groups:
        if g.id not in tiny or not (gm.shape and gm.size):
            continue
        parent = _ring_owner(gm, g.id, tiny | bg | {g.id})
        if parent >= 0 and parent in by_id and _lighting_variant(g.albedo_lab, by_id[parent].albedo_lab):
            g.minor, g.parent = True, int(parent)
    return groups
