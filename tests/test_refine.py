"""Group refinement (recolor/segmentation/refine.py) on synthetic label maps. No models:
the boundary snap is a stand-in callable; the real grouping code builds the groups."""
from __future__ import annotations

import numpy as np

from recolor import imageio
from recolor.segmentation import grouping, refine
from recolor.types import ColorGroup


def _info(n: int) -> list[dict]:
    return [{"id": i, "source": "sam", "confidence": 0.9} for i in range(n)]


def _check_partition(labels, regions, groups, group_map):
    assert labels.dtype == np.int32 and labels.min() == 0 and labels.max() == len(regions) - 1
    assert [r.id for r in regions] == list(range(len(regions)))
    assert group_map.dtype == np.int32 and group_map.min() >= 0
    grouping._check_state(regions, groups)
    lut = np.array([r.group_id for r in regions])
    assert np.array_equal(lut[labels], group_map)


# ------------------------------------------------------------------ absorb

def _washed_scene():
    """Region 0: red paint. Region 1: the same paint washed out by a highlight (same hue,
    0.57x chroma, +10 L, chroma falling as lightness rises). Region 2: washed the same way
    but 25 deg off the paint's hue. Region 3: the grey backdrop around them."""
    h, w = 60, 100
    labels = np.full((h, w), 3, np.int32)
    labels[10:50, 10:50] = 0
    labels[10:30, 50:70] = 1
    labels[30:50, 50:70] = 2
    rng = np.random.default_rng(0)
    lab = np.zeros((h, w, 3), np.float32)
    hue = np.radians(36.87)
    lab[labels == 0] = (45.0, 75.0 * np.cos(hue), 75.0 * np.sin(hue))
    for rid, dh in ((1, 0.0), (2, 25.0)):
        m = labels == rid
        L = rng.uniform(50.0, 62.0, int(m.sum())).astype(np.float32)
        C = 50.0 - 1.5 * (L - 50.0)
        hh = hue + np.radians(dh)
        lab[m] = np.stack([L, C * np.cos(hh), C * np.sin(hh)], 1)
    lab[labels == 3] = (60.0, 0.0, 0.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    return labels, albedo


def test_absorb_moves_only_the_washed_out_paint():
    labels, albedo = _washed_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(4))
    g_of = {r.id: r.group_id for r in regions}
    assert len({g_of[0], g_of[1], g_of[2]}) == 3                  # the plain linkage keeps them apart
    lab = imageio.linear_to_lab(albedo)
    regions, groups, gm, moves = refine.absorb_washed(regions, groups, labels, lab)
    g_of = {r.id: r.group_id for r in regions}
    assert [m["region"] for m in moves] == [1]
    assert g_of[1] == g_of[0] and g_of[2] != g_of[0] and g_of[3] != g_of[0]
    _check_partition(labels, regions, groups, gm)


def test_a_locked_group_absorbs_nothing():
    labels, albedo = _washed_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(4))
    paint = next(g for g in groups if 0 in g.region_ids)
    paint.locked = True
    _, _, _, moves = refine.absorb_washed(regions, groups, labels, imageio.linear_to_lab(albedo))
    assert moves == []


# ------------------------------------------------------------------ decals

DECAL = np.s_[20:32, 20:50]          # a white 12 x 30 lettering, 360 px
SMALL = np.s_[50:58, 20:28]          # a white blob below DECAL_MIN_PX
SHADOW = np.s_[45:65, 50:70]         # a dark blob: not a bright neutral


def _decal_scene():
    """Red paint (region 0) on a grey backdrop (1) carrying white lettering, a smaller white
    blob, a dark shadow blob and a dull dark red part (2, another material in its hue)."""
    h, w = 80, 120
    labels = np.full((h, w), 1, np.int32)
    labels[5:75, 5:100] = 0
    labels[58:72, 76:92] = 2
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (45.0, 60.0, 45.0)
    lab[DECAL] = (90.0, 0.0, 0.0)
    lab[SMALL] = (90.0, 0.0, 0.0)
    lab[SHADOW] = (15.0, 5.0, 3.0)
    lab[labels == 1] = (60.0, 0.0, 0.0)
    lab[labels == 2] = (30.0, 20.0, 15.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    photo = imageio.to_uint8(imageio.linear_to_srgb(albedo * 0.9))
    return labels, albedo, photo


def test_split_decals_carves_bright_neutral_blobs_only():
    labels, albedo, _ = _decal_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(3))
    lab = imageio.linear_to_lab(albedo)
    new_labels, regions, groups, gm, islands = refine.split_decals(labels, lab, regions, groups)
    assert islands.dtype == bool and islands.sum() == 360
    assert islands[DECAL].all() and not islands[SMALL].any() and not islands[SHADOW].any()
    assert len(np.unique(new_labels[islands])) == 1
    decal_id = int(new_labels[DECAL][0, 0])
    assert decal_id == 3 and regions[decal_id].source == "split"
    assert not np.array_equal(new_labels, labels) and labels[DECAL].max() == 0     # input untouched
    paint = next(g for g in groups if 0 in g.region_ids)
    assert decal_id not in paint.region_ids                         # the letters leave the paint
    _check_partition(new_labels, regions, groups, gm)


def test_split_decals_without_a_chromatic_paint_is_a_no_op():
    labels, albedo, _ = _decal_scene()
    albedo = np.full_like(albedo, 0.4)
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(3))
    out, regions2, groups2, gm2, islands = refine.split_decals(labels, imageio.linear_to_lab(albedo), regions, groups)
    assert np.array_equal(out, labels) and not islands.any() and len(regions2) == len(regions)


# ------------------------------------------------------------------ relabel after the snap

def test_relabel_compacts_ids_and_keeps_every_region_in_its_group():
    labels, albedo, _ = _decal_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(3))
    moved = labels.copy()
    moved[labels == 2] = 0                                          # the snap swallowed region 2
    moved[5:75, 99] = 1                                             # and region 1 grew by a column
    out, regions2, groups2, gm2 = refine.relabel(moved, imageio.linear_to_lab(albedo), regions)
    assert int(out.max()) == 1 and len(regions2) == 2
    assert regions2[1].area == int((moved == 1).sum())
    g_before = {r.id: r.group_id for r in regions}
    names = {g.id: g.name for g in groups}
    names2 = {g.id: g.name for g in groups2}
    assert names2[regions2[0].group_id] == names[g_before[0]]
    _check_partition(out, regions2, groups2, gm2)


# ------------------------------------------------------------------ material lock

def _g(gid, lab, area, bg=False) -> ColorGroup:
    lab = tuple(float(v) for v in lab)
    return ColorGroup(id=gid, name=f"g{gid}", albedo_lab=lab, albedo_hex=imageio.lab_to_hex(lab), area=area,
                      area_frac=0.1, region_ids=[gid], hue_family="yellow", is_background=bg)


def test_lock_materials_locks_other_materials_in_the_paints_hue():
    groups = [
        _g(0, (86.0, 0.3, -0.5), 90000, bg=True),                   # backdrop
        _g(1, (78.5, 3.9, 79.0), 10000),                            # the yellow paint
        _g(2, (58.5, 5.5, 25.2), 500),                              # gold caliper: dull and dark -> locked
        _g(3, (61.0, 14.5, 64.0), 800),                             # the paint in shadow: stays paint
        _g(4, (40.0, 60.0, 40.0), 900),                             # red: another hue
        _g(5, (20.0, 1.0, 2.0), 700),                               # black: neutral
        _g(6, (85.0, 2.0, 30.0), 400),                              # washed-out highlight: lighter, not locked
    ]
    assert refine.main_paint(groups).id == 1
    assert refine.paint_family(groups) == [1, 2, 3, 6]
    assert refine.lock_materials(groups) == [2]
    assert [g.locked for g in groups] == [False, False, True, False, False, False, False]
    assert refine.lock_materials([_g(0, (50.0, 1.0, 1.0), 10)]) == []


# ------------------------------------------------------------------ protected segments

def test_protect_mask_marks_own_coloured_objects_outside_the_paint():
    h, w = 60, 120
    labels = np.zeros((h, w), np.int32)
    labels[5:25, 60:80] = 1                                         # a yellow object (400 px) in a grey group
    labels[30:50, 60:80] = 2                                        # a grey part with a little yellow in it
    labels[5:25, 90:110] = 3                                        # a decal island, yellow-ish
    labels[40:47, 100:107] = 4                                      # a tiny yellow part (49 px)
    labels[:, 115:] = 5
    group_map = np.array([1, 0, 0, 0, 0, 0], np.int32)[labels]
    groups = [_g(0, (60.0, 0.0, 0.0), 5000), _g(1, (78.5, 3.9, 79.0), 2000)]
    photo = np.full((h, w, 3), 128, np.uint8)
    yellow = (230, 190, 30)
    photo[labels == 0] = yellow                                     # the paint itself
    photo[labels == 1] = yellow
    photo[30:34, 60:80] = yellow                                    # 80 of 400 px (20 %)
    photo[labels == 3] = yellow
    photo[labels == 4] = yellow
    islands = labels == 3
    prot = refine.protect_mask(photo, labels, group_map, groups, islands)
    assert prot.dtype == bool and prot.shape == labels.shape
    assert prot[labels == 1].all()
    assert not prot[labels == 0].any() and not prot[labels == 2].any()
    assert not prot[islands].any() and not prot[labels == 4].any()
    # with no chromatic paint there is nothing to protect
    assert not refine.protect_mask(photo, labels, group_map, [_g(0, (60.0, 0.0, 0.0), 5000)], islands).any()


# ------------------------------------------------------------------ the whole stage

def test_refine_groups_keeps_a_complete_partition_and_reports():
    labels, albedo, photo = _decal_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(3))
    calls = []

    def snap(image, lab_map, group_map, grps, protect=None, progress=None):
        calls.append(protect.copy())
        out = lab_map.copy()
        out[5:75, 99] = lab_map[0, 100]                             # the backdrop grows by one column
        if progress:
            progress(1.0, "snapped")
        return out, "vitmatte"

    msgs = []
    res = refine.refine_groups(photo, albedo, labels, regions, groups, gm, snap, progress=lambda f, m: msgs.append(m))
    _check_partition(res.labels, res.regions, res.groups, res.group_map)
    assert res.islands.dtype == bool and res.islands.sum() == 360
    assert np.array_equal(calls[0], res.islands)                    # the snap was told to keep the islands
    assert (res.labels[5:75, 99] == res.labels[0, 100]).all()
    assert res.protect.dtype == bool and res.protect.shape == labels.shape
    assert res.report["snap"] == "vitmatte" and res.report["decal_px"] == 360
    locked = [g for g in res.groups if g.locked]
    assert len(locked) == 1 and res.report["locked"] == [locked[0].id]
    assert locked[0].region_ids == [int(res.labels[65, 84])]        # the dull dark red material
    assert "snapped" in msgs


def test_refine_after_edit_reapplies_the_rules_after_a_regroup():
    labels, albedo, photo = _decal_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(3))
    res = refine.refine_groups(photo, albedo, labels, regions, groups, gm,
                               lambda image, l, g, gr, protect=None, progress=None: (l.copy(), "none"))
    regions, groups, gm = grouping.regroup(res.regions, res.labels, albedo, None, 10.0)
    assert not any(g.locked for g in groups)                        # a regroup drops user flags
    r2, g2, gm2, prot = refine.refine_after_edit(photo, albedo, res.labels, regions, groups, gm, res.islands, regrouped=True)
    assert sum(g.locked for g in g2) == 1
    _check_partition(res.labels, r2, g2, gm2)
    r3, g3, gm3, prot3 = refine.refine_after_edit(photo, albedo, res.labels, regions, groups, gm, res.islands, regrouped=False)
    assert g3 is groups and np.array_equal(gm3, gm) and prot3.shape == labels.shape


# ------------------------------------------------------------------ decals: parts at the paint's edge

def _lens_scene():
    """Yellow paint (region 0) in front of a light grey backdrop (region 1). Inside the
    paint's label: a clear lens at its edge, a little lighter than the paint and exactly
    the backdrop's colour (it continues into it), and the same colour once more in the
    middle of the paint, where it touches no neighbour."""
    h, w = 60, 120
    labels = np.full((h, w), 1, np.int32)
    labels[10:50, 20:100] = 0
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (84.0, 0.5, -0.5)                                   # the backdrop
    lab[labels == 0] = (78.0, 4.0, 78.0)                            # the paint
    lab[20:30, 20:32] = (84.0, 0.5, -0.5)                           # the lens, on the paint's left edge
    lab[30:40, 60:72] = (84.0, 0.5, -0.5)                           # the same colour, inside the paint
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    return labels, albedo


def test_split_decals_carves_a_clear_part_that_continues_into_its_neutral_neighbour():
    labels, albedo = _lens_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(2))
    lab = imageio.linear_to_lab(albedo)
    new_labels, regions, groups, gm, islands = refine.split_decals(labels, lab, regions, groups)
    lens = np.zeros(labels.shape, bool)
    lens[20:30, 20:32] = True
    inner = np.zeros(labels.shape, bool)
    inner[30:40, 60:72] = True
    assert islands[lens].mean() > 0.9                               # carved: the backdrop's material
    assert not islands[inner].any()                                 # only 6 L lighter and touching nothing
    lens_rid = int(np.bincount(new_labels[lens]).argmax())
    backdrop = next(g for g in groups if 1 in g.region_ids)
    assert lens_rid in backdrop.region_ids                          # it joins the backdrop's group
    _check_partition(new_labels, regions, groups, gm)


def test_main_paint_is_never_a_locked_group():
    groups = [
        _g(0, (86.0, 0.3, -0.5), 90000, bg=True),
        _g(1, (58.5, 5.5, 25.2), 20000),                            # a locked gold material, largest
        _g(2, (78.5, 3.9, 79.0), 10000),                            # the yellow paint
        _g(3, (61.0, 14.5, 64.0), 800),
    ]
    groups[1].locked = True
    assert refine.main_paint(groups).id == 2
    assert refine.paint_family(groups) == [2, 3]                    # a locked group is not the paint either
    groups[2].locked = groups[3].locked = True
    assert refine.main_paint(groups) is None and refine.paint_family(groups) == []


# ------------------------------------------------------------------ a regroup reproduces the analysis

def _snap_scene():
    """Red paint (region 0), a slightly different red part (region 1, clustered with the
    paint) and a grey backdrop (region 2)."""
    h, w = 60, 120
    labels = np.full((h, w), 2, np.int32)
    labels[5:55, 5:70] = 0
    labels[20:40, 70:90] = 1
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (60.0, 0.0, 0.0)
    lab[labels == 0] = (45.0, 60.0, 45.0)
    lab[labels == 1] = (47.0, 57.0, 44.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    photo = imageio.to_uint8(imageio.linear_to_srgb(albedo * 0.9))
    return labels, albedo, photo


def _greedy_snap(image, lab_map, group_map, grps, protect=None, progress=None):
    """A boundary snap that lets region 1 swallow a wide strip of the backdrop: afterwards
    most of region 1's pixels are grey, so clustering the snapped regions again would put
    it with the backdrop."""
    out = lab_map.copy()
    out[20:40, 90:118] = 1
    return out, "vitmatte"


def test_regroup_refined_reproduces_the_analysis_where_a_plain_regroup_does_not():
    labels, albedo, photo = _snap_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(3))
    res = refine.refine_groups(photo, albedo, labels, regions, groups, gm, _greedy_snap)
    paint = next(g for g in res.groups if 0 in g.region_ids)
    assert 1 in paint.region_ids                                    # the analysis clustered before the snap
    assert res.origin.tolist() == [0, 1, 2]
    r2, g2, gm2, prot2 = refine.regroup_refined(photo, albedo, labels, res.origin, res.labels, res.regions,
                                                res.islands, None, 10.0)
    assert [(g.name, g.area, g.region_ids, g.locked, g.is_background) for g in g2] == \
        [(g.name, g.area, g.region_ids, g.locked, g.is_background) for g in res.groups]
    assert np.array_equal(gm2, res.group_map) and np.array_equal(prot2, res.protect)
    _check_partition(res.labels, r2, g2, gm2)
    # clustering the snapped regions (the old regroup) splits region 1 off the paint
    _, g3, _ = grouping.regroup(res.regions, res.labels, albedo, None, 10.0)
    assert 1 not in next(g for g in g3 if 0 in g.region_ids).region_ids


def test_regroup_refined_takes_other_options_and_places_regions_without_an_origin():
    labels, albedo, photo = _decal_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(3))
    res = refine.refine_groups(photo, albedo, labels, regions, groups, gm,
                               lambda image, l, g, gr, protect=None, progress=None: (l.copy(), "none"))
    decal = int(res.labels[25, 30])
    assert res.origin[decal] == -1 and sorted(res.origin[res.origin >= 0].tolist()) == [0, 1, 2]
    r2, g2, gm2, _ = refine.regroup_refined(photo, albedo, labels, res.origin, res.labels, res.regions,
                                            res.islands, 2, 10.0)
    assert len(g2) <= 3                                             # max_groups caps the clustering (+ the decal's)
    _check_partition(res.labels, r2, g2, gm2)
    # a region beyond the origin array (a piece a pixel-level split cut off) joins by colour
    short = res.origin[:-1]
    r3, g3, gm3, _ = refine.regroup_refined(photo, albedo, labels, short, res.labels, res.regions, res.islands, None, 10.0)
    _check_partition(res.labels, r3, g3, gm3)


# ------------------------------------------------------------------ recovered parts stay out of the paint

def _part_scene():
    """A yellow paint panel (region 0), a gold part the regions stage cut out of the paint's
    scattered region with SAM (region 1, source 'part': darker and duller, but close enough
    to the paint's family), a grey machine (region 2) and a backdrop (region 3)."""
    h, w = 80, 160
    labels = np.full((h, w), 3, np.int32)
    labels[5:75, 5:80] = 0
    labels[20:60, 90:150] = 2
    labels[30:44, 110:130] = 1
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (88.0, 0.0, 0.0)
    lab[labels == 0] = (78.0, 4.0, 79.0)
    lab[labels == 1] = (72.0, 6.0, 70.0)            # within dE 10 of the paint: clustered with it
    lab[labels == 2] = (22.0, 0.5, 1.5)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    photo = imageio.to_uint8(imageio.linear_to_srgb(albedo * 0.9))
    info = _info(4)
    info[1]["source"] = refine.PART_SOURCE
    return labels, albedo, photo, info


def test_a_recovered_part_leaves_the_paint_and_is_locked():
    labels, albedo, photo, info = _part_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    g_of = {r.id: r.group_id for r in regions}
    assert g_of[1] == g_of[0]                                       # clustering alone paints it with the paint
    res = refine.refine_groups(photo, albedo, labels, regions, groups, gm,
                               lambda image, l, g, gr, protect=None, progress=None: (l.copy(), "none"))
    part = next(g for g in res.groups if 1 in g.region_ids)
    paint = next(g for g in res.groups if 0 in g.region_ids)
    assert part.id != paint.id and part.region_ids == [1] and part.locked and not paint.locked
    assert res.report["isolated_parts"] == 1 and part.id in res.report["locked"]
    assert refine.paint_family(res.groups) == [paint.id]
    # a part that is not in the paint is left where clustering put it
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(4))
    r2, g2, gm2, moved = refine.isolate_parts(list(regions), list(groups), labels, [2])
    assert moved == [] and [g.region_ids for g in g2] == [g.region_ids for g in groups]
    # a regroup with the analysis's options gives the same, locked part back
    seed_parts = [1]
    r3, g3, gm3, prot3 = refine.regroup_refined(photo, albedo, labels, res.origin, res.labels, res.regions,
                                                res.islands, None, 10.0, parts=seed_parts)
    assert [(g.area, g.region_ids, g.locked) for g in g3] == [(g.area, g.region_ids, g.locked) for g in res.groups]
    assert np.array_equal(gm3, res.group_map) and np.array_equal(prot3, res.protect)


# ------------------------------------------------------------------ decal gaps, majority locks, stamped islands

def _gap_scene():
    """Yellow paint (region 0) on a grey backdrop (region 2) with a red decal region (1)
    whose SAM mask swallowed the yellow gaps between its letters: two small yellow blobs
    inside the decal's region, and one big yellow patch (a real two-tone part, left alone)."""
    h, w = 90, 160
    labels = np.full((h, w), 2, np.int32)
    labels[5:85, 5:100] = 0
    labels[30:60, 40:90] = 1
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (60.0, 0.0, 0.0)
    lab[labels == 0] = (78.0, 4.0, 78.0)
    lab[labels == 1] = (40.0, 55.0, 40.0)
    lab[38:44, 48:52] = (78.0, 4.0, 78.0)                               # a 24 px gap of yellow
    lab[50:53, 60:80] = (78.0, 4.0, 78.0)                               # a 60 px gap
    labels[62:82, 20:40] = 3                                             # another red region, mostly yellow inside: not a gap
    lab[labels == 3] = (40.0, 55.0, 40.0)
    lab[64:80, 22:38] = (78.0, 4.0, 78.0)                               # 256 px of yellow inside a 400 px region: a real part? no: > GAP_MAX_PX
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    return labels, albedo


def test_fill_decal_gaps_gives_the_paint_its_gaps_back():
    labels, albedo = _gap_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(4))
    lab = imageio.linear_to_lab(albedo)
    out, r2, g2, gm2, moved = refine.fill_decal_gaps(labels, lab, regions, groups)
    assert moved == 24 + 60
    assert (out[38:44, 48:52] == 0).all() and (out[50:53, 60:80] == 0).all()   # the gaps are paint again
    assert (out[30:38, 40:90] == 1).all()                                       # the letters stay the decal's
    assert (out[64:80, 22:38] == 3).all()                                       # too big for a gap: untouched
    assert np.array_equal(out != labels, (labels == 1) & (out == 0))
    _check_partition(out, r2, g2, gm2)
    assert next(r for r in r2 if r.id == 1).area == 1500 - 84
    # nothing to do without a chromatic paint or without gaps
    same, *_ , n = refine.fill_decal_gaps(out, lab, r2, g2)
    assert n == 0 and np.array_equal(same, out)


def test_lock_parts_locks_a_group_that_is_mostly_parts():
    from recolor.types import Region
    groups = [_g(0, (78.0, 4.0, 78.0), 5000), _g(1, (60.0, 10.0, 40.0), 900), _g(2, (60.0, 10.0, 40.0), 400)]
    groups[1].region_ids = [1, 7]
    groups[2].region_ids = [2, 8]
    regions = [Region(id=i, area=a, bbox=(0, 0, 1, 1), albedo_lab=(50.0, 0.0, 0.0), albedo_hex="#777777", group_id=0,
                      touches_border=False, source="sam", confidence=1.0)
               for i, a in ((0, 5000), (1, 700), (7, 200), (2, 100), (8, 300))]
    assert refine.lock_parts(groups, [1, 2], regions) == [1]           # 700 of 900 px are the part; 100 of 400 are not
    assert groups[1].locked and not groups[2].locked and not groups[0].locked
    assert refine.lock_parts([_g(3, (60.0, 10.0, 40.0), 10)], [1, 2], regions) == []
    # without the regions every member must be a part (the older rule)
    groups[1].locked = False
    assert refine.lock_parts(groups, [1, 2]) == []
    assert refine.lock_parts(groups, [1, 7]) == [1]


def test_stamped_lettering_and_small_parts_are_islands():
    """Regions the regions stage stamped as lettering or small distinct parts are decal
    islands for the snap and the engine, whatever their colour."""
    labels, albedo, photo = _decal_scene()
    info = _info(3)
    labels = labels.copy()
    labels[40:50, 30:40] = 3                                            # a small red part inside the paint, source 'small'
    info.append({"id": 3, "source": "small", "confidence": 0.9})
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    seen = []

    def snap(image, lab_map, group_map, grps, protect=None, progress=None):
        seen.append(protect.copy())
        return lab_map.copy(), "none"

    res = refine.refine_groups(photo, albedo, labels, regions, groups, gm, snap)
    assert res.islands[40:50, 30:40].all() and res.islands[DECAL].all()
    assert seen[0][40:50, 30:40].all()                                  # the snap was told to keep it
    assert res.report["decal_px"] == int(res.islands.sum())


# ------------------------------------------------------------------ the seed's sources reach the highlight guard

def test_regroup_refined_hands_the_input_sources_to_the_highlight_absorb(monkeypatch):
    """The regions stage's sources ('text', 'small': never absorbed as a highlight) are not
    in the label map; a regroup gets them from the seed and the guard sees them again."""
    from recolor.segmentation import materials
    labels, albedo, photo = _snap_scene()
    info = _info(3)
    info[1]["source"] = "text"
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    res = refine.refine_groups(photo, albedo, labels, regions, groups, gm,
                               lambda image, l, g, gr, protect=None, progress=None: (l.copy(), "none"))
    seen = []
    real = materials.absorb_highlights

    def spy(regions, groups, lab, feats):
        seen.append([r.source for r in sorted(regions, key=lambda r: r.id)])
        return real(regions, groups, lab, feats)

    monkeypatch.setattr(materials, "absorb_highlights", spy)
    refine.regroup_refined(photo, albedo, labels, res.origin, res.labels, res.regions, res.islands, None, 10.0,
                           sources=["sam", "text", "sam"])
    refine.regroup_refined(photo, albedo, labels, res.origin, res.labels, res.regions, res.islands, None, 10.0)
    assert seen[0] == ["sam", "text", "sam"] and seen[1] == ["sam", "sam", "sam"]


# ------------------------------------------------------------------ letter counters and the sheen merge

def _counter_scene():
    """0: grey backdrop, 1: red paint, 2: a white "8" (lettering, source 'text') on it whose
    lower counter (a 6x6 hole of the paint mixed with the letter's edge: lighter, more orange)
    is part of the text mask, 3: the upper counter, a small distinct region ('small') inside the
    letter, 4: a blue sticker inside the letter (a real colour, not the paint)."""
    h, w = 120, 160
    labels = np.zeros((h, w), np.int32)
    labels[10:110, 10:150] = 1
    labels[30:90, 40:80] = 2                                     # the letter's box
    labels[40:48, 50:58] = 3                                     # the upper counter
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (60.0, 0.0, 0.0)
    lab[labels == 1] = (42.0, 66.0, 54.0)
    lab[labels == 2] = (92.0, 0.0, 3.0)
    lab[60:66, 55:61] = (56.0, 42.0, 56.0)                       # the lower counter, inside the text mask
    lab[labels == 3] = (57.0, 37.0, 56.0)
    labels[70:74, 65:69] = 4
    lab[labels == 4] = (50.0, -10.0, -45.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    info = [{"id": 0, "source": "sam", "bg": 2}, {"id": 1, "source": "sam"}, {"id": 2, "source": "text"},
            {"id": 3, "source": "small"}, {"id": 4, "source": "small"}]
    return labels, albedo, info


def test_letter_counters_go_back_to_the_paint():
    labels, albedo, info = _counter_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    lab = imageio.linear_to_lab(albedo)
    out, regions2, groups2, gm2, moved = refine.fill_letter_counters(labels, lab, list(regions), list(groups))
    assert moved == 36 + 64                                      # the lower counter's pixels and the upper counter
    assert (out[60:66, 55:61] == 1).all() and (out[40:48, 50:58] == 1).all()
    assert (out[labels == 4] == 4).all()                         # the blue sticker is no counter
    assert (out[(labels == 2) & ~((np.arange(120)[:, None] >= 60) & (np.arange(120)[:, None] < 66)
                                  & (np.arange(160)[None] >= 55) & (np.arange(160)[None] < 61))] == 2).all()
    paint = next(g for g in groups2 if 1 in g.region_ids)
    assert all(3 not in g.region_ids for g in groups2)           # the emptied counter region is gone
    assert paint.id == gm2[62, 57]
    # without a chromatic paint there is nothing to give back
    lab2 = lab.copy()
    lab2[labels != 2] = (50.0, 0.0, 0.0)
    albedo2 = np.clip(imageio.lab_to_linear(lab2), 0.0, 1.0).astype(np.float32)
    r3, g3, gm3 = grouping.group_regions(labels, albedo2, info)
    assert refine.fill_letter_counters(labels, lab2, list(r3), list(g3))[4] == 0


def test_the_sheen_merge_takes_the_paint_facing_the_light_only():
    """The Torana's roof: the albedo is the body's, darker and duller, and the photo is lighter
    and duller in the body's hue (a white sheen). absorb_sheen, the refinement's last step, joins
    it to the paint; a photo that is darker (another material) stays apart, and so does the pair
    in the clustering, which runs without the sheen."""
    h, w = 80, 120
    labels = np.zeros((h, w), np.int32)
    labels[10:70, 10:110] = 1                                    # the body
    labels[10:30, 20:80] = 2                                     # the roof
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (60.0, 0.0, 0.0)
    lab[labels == 1] = (43.7, 53.7, 47.8)
    lab[labels == 2] = (39.8, 26.5, 22.8)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    photo_lab = lab.copy()
    photo_lab[labels == 1] = (42.4, 47.3, 47.6)
    photo_lab[labels == 2] = (58.4, 28.1, 20.7)                  # lighter, duller, the same hue
    photo = imageio.to_uint8(imageio.lab_to_rgb(photo_lab))
    info = [{"id": 0, "source": "sam", "bg": 2}, {"id": 1, "source": "sam"}, {"id": 2, "source": "sam"}]
    regions, groups, gm = grouping.group_regions(labels, albedo, info, photo_rgb_u8=photo)
    g = {r.id: r.group_id for r in regions}
    assert g[1] != g[2]                                          # the clustering keeps them apart
    protect = np.zeros(labels.shape, bool)
    r2, g2, gm2, pr2, moves = refine.absorb_sheen(photo, imageio.linear_to_lab(albedo), labels, list(regions), list(groups),
                                                  gm, None, protect)
    assert len(moves) == 1 and len({r.group_id for r in r2 if r.id in (1, 2)}) == 1
    photo_lab[labels == 2] = (30.0, 20.0, 15.0)                  # darker in the photo: not a sheen
    photo = imageio.to_uint8(imageio.lab_to_rgb(photo_lab))
    regions, groups, gm = grouping.group_regions(labels, albedo, info, photo_rgb_u8=photo)
    assert refine.absorb_sheen(photo, imageio.linear_to_lab(albedo), labels, list(regions), list(groups), gm, None,
                               protect)[4] == []


def test_refine_groups_survives_a_counter_region_given_back_to_the_paint():
    """An emptied counter region leaves a gap in the ids until the relabel after the snap."""
    labels, albedo, info = _counter_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info)

    def snap(image, labels, group_map, groups, protect=None, progress=None):
        return labels.copy(), "none"

    photo = imageio.to_uint8(imageio.linear_to_srgb(albedo * 0.9))
    res = refine.refine_groups(photo, albedo, labels, regions, groups, gm, snap)
    _check_partition(res.labels, res.regions, res.groups, res.group_map)
    assert res.report["counter_px"] == 100
    paint = res.group_map[15, 15]
    assert res.group_map[44, 54] == paint and res.group_map[62, 57] == paint
    assert len(res.origin) == len(res.regions) and (res.origin[res.labels[44, 54]] == 1)


def test_only_lettering_printed_on_the_paint_gives_counters_back():
    """Navy letters on a white shoe beside blue jeans (the paint): the strokes are the paint's
    hue, the lettering is not on the paint, nothing moves."""
    labels, albedo, info = _counter_scene()
    lab = imageio.linear_to_lab(albedo)
    lab[labels == 1] = (35.0, 5.0, -40.0)                        # the "paint" is blue denim ...
    lab[labels == 2] = (30.0, 6.0, -38.0)                        # ... and the letters are navy
    lab[(labels == 2) & (np.arange(160)[None] < 60)] = (92.0, 0.0, 2.0)
    labels2 = labels.copy()
    labels2[26:94, 36:84] = np.where(labels[26:94, 36:84] == 1, 5, labels[26:94, 36:84])   # a white shoe around them
    lab[labels2 == 5] = (92.0, 0.0, 2.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    info2 = info + [{"id": 5, "source": "sam"}]
    regions, groups, gm = grouping.group_regions(labels2, albedo, info2)
    assert refine.fill_letter_counters(labels2, imageio.linear_to_lab(albedo), list(regions), list(groups))[4] == 0
