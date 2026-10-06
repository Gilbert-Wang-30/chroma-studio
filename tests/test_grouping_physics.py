"""Grouping beyond the clustering (recolor/segmentation/grouping.py): one paint under
different light merged at group level, the matte rule's background flags and the unique
group names. Synthetic label maps, no models."""
from __future__ import annotations

import numpy as np

from recolor import imageio
from recolor.segmentation import grouping
from recolor.types import Region


def _info(n, bg=None):
    out = [{"id": i, "source": "sam", "confidence": 0.9} for i in range(n)]
    for i, k in (bg or {}).items():
        out[i]["bg"] = k
    return out


def _photo_of(albedo, shading):
    return imageio.to_uint8(imageio.linear_to_srgb(np.clip(albedo * shading, 0, 1)))


# ------------------------------------------------------------------ one paint under different light

def _lit_scene():
    """Region 0: a red paint panel in the light. Region 1: the same paint in shadow, whose
    albedo kept part of the shading (darker and a little duller, same lightness-normalised
    chromaticity). Region 2: a chrome part reflecting the paint (much duller and darker: the
    material veto). Region 3: a grey backdrop."""
    h, w = 80, 160
    labels = np.full((h, w), 3, np.int32)
    labels[10:70, 10:60] = 0
    labels[10:70, 70:100] = 1
    labels[10:70, 110:140] = 2
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (70.0, 0.0, 0.0)
    lab[labels == 0] = (48.0, 62.0, 40.0)
    shading = np.full((h, w, 3), 0.85, np.float32)
    # the shadowed panel: the same body colour at 40 % of the light; the albedo keeps a third of that
    lit = np.clip(imageio.lab_to_linear(lab[labels == 0][:1]), 0, 1)
    shadowed_alb = imageio.linear_to_lab((lit * 0.55)[None])[0, 0]
    lab[labels == 1] = shadowed_alb
    shading[labels == 1] = 0.85 * 0.7
    lab[labels == 2] = (30.0, 16.0, 10.0)                                # chrome: chroma 19 (< 0.6 x 74), 18 L darker
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    return labels, albedo, _photo_of(albedo, shading)


def test_absorb_lit_merges_the_shadowed_paint_and_refuses_the_chrome():
    labels, albedo, photo = _lit_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(4))
    g_of = {r.id: r.group_id for r in regions}
    assert len({g_of[0], g_of[1], g_of[2]}) == 3                         # CIEDE2000 alone keeps all three apart
    lab = imageio.linear_to_lab(albedo)
    r2, g2, gm2, log = grouping.absorb_lit(regions, groups, labels, lab, grouping.photo_lab_of(photo))
    g_of = {r.id: r.group_id for r in r2}
    assert g_of[1] == g_of[0], log                                       # the shadowed panel joined its paint
    assert g_of[2] != g_of[0] and g_of[3] != g_of[0]                     # the chrome and the backdrop did not
    assert [m["group"] for m in log] and all(m["into"] == g_of[0] for m in log)
    grouping._check_state(r2, g2)
    assert np.array_equal(gm2, np.array([r.group_id for r in r2])[labels])
    # the same through the public entry point, and not without the photo
    _, g3, _ = grouping.group_regions(labels, albedo, _info(4), photo_rgb_u8=photo)
    assert len(g3) == len(groups) - 1
    _, g4, _ = grouping.group_regions(labels, albedo, _info(4))
    assert len(g4) == len(groups)


def test_absorb_lit_never_touches_neutral_locked_or_background_groups():
    labels, albedo, photo = _lit_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(4))
    lab = imageio.linear_to_lab(albedo)
    paint = next(g for g in groups if 0 in g.region_ids)
    paint.locked = True
    _, _, _, log = grouping.absorb_lit(regions, groups, labels, lab, grouping.photo_lab_of(photo))
    assert log == []                                                     # a locked anchor absorbs nothing
    paint.locked = False
    shadow = next(g for g in groups if 1 in g.region_ids)
    shadow.is_background = True
    _, _, _, log = grouping.absorb_lit(regions, groups, labels, lab, grouping.photo_lab_of(photo))
    assert log == []                                                     # a background candidate never moves


# ------------------------------------------------------------------ the matte rule

def _matte_scene():
    """A red object (regions 0 and 1) on a two-tone backdrop (a dark floor, region 2, and a
    light wall, region 3, which owns most of the border) with a see-through hole (region
    4, the wall seen through the object) and a grey part of the object touching the wall
    (region 5, the same grey as the wall)."""
    h, w = 100, 160
    labels = np.full((h, w), 3, np.int32)
    labels[70:, :] = 2                                                  # the floor
    labels[15:65, 30:90] = 0                                            # the object
    labels[15:65, 90:120] = 1                                           # a second panel of the object
    labels[30:40, 50:60] = 4                                            # the hole: wall colour, inside the object
    labels[20:60, 120:135] = 5                                          # a grey object part
    lab = np.zeros((h, w, 3), np.float32)
    lab[labels == 3] = (86.0, 0.0, 0.0)
    lab[labels == 2] = (30.0, 0.0, 0.0)
    lab[labels == 0] = (45.0, 60.0, 45.0)
    lab[labels == 1] = (46.0, 58.0, 44.0)
    lab[labels == 4] = (86.0, 0.0, 0.0)
    lab[labels == 5] = (86.0, 0.0, 0.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    fg = np.zeros((h, w), np.float32)
    fg[15:65, 30:135] = 1.0                                             # the matte: the whole object, hole included ...
    fg[30:40, 50:60] = 0.0                                              # ... except the hole
    return labels, albedo, fg


def test_backdrop_decisions_follow_the_matte_and_the_border_set():
    labels, albedo, fg = _matte_scene()
    kinds = grouping.backdrop_decisions(labels, albedo, _info(6), fg)
    assert kinds.dtype == np.int8 and len(kinds) == 6
    assert kinds[3] == grouping.BG_MATTE and kinds[2] == grouping.BG_MATTE      # wall and floor: the matte says backdrop
    assert kinds[0] == grouping.BG_OBJECT and kinds[1] == grouping.BG_OBJECT
    assert kinds[5] == grouping.BG_OBJECT                                # the same grey as the wall, but matte 1.0
    assert kinds[4] == grouping.BG_MATTE                                 # the hole: backdrop seen through the object


def test_the_backdrop_reaches_through_chains_of_regions_the_matte_calls_backdrop():
    """A car parked behind the subject, a piece at a time: its body (region 6) touches the wall,
    its headlight (region 7) touches only the body. Both are off the matte; the headlight, beyond
    the border set's first ring, was decided object and became a colour row of the subject."""
    labels, albedo, fg = _matte_scene()
    labels[75:95, 5:25] = 6                                             # the other car, on the floor
    labels[80:86, 10:16] = 7                                            # its headlight, inside it
    lab = imageio.linear_to_lab(albedo)
    lab[labels == 6] = (35.0, 10.0, -40.0)
    lab[labels == 7] = (70.0, 5.0, 60.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    kinds = grouping.backdrop_decisions(labels, albedo, _info(8), fg)
    assert kinds[6] == grouping.BG_MATTE and kinds[7] == grouping.BG_MATTE
    assert kinds[0] == grouping.BG_OBJECT and kinds[5] == grouping.BG_OBJECT


def test_group_regions_flags_every_backdrop_group_and_keeps_the_object_out():
    labels, albedo, fg = _matte_scene()
    kinds = grouping.backdrop_decisions(labels, albedo, _info(6), fg)
    info = _info(6, {i: int(k) for i, k in enumerate(kinds)})
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    grouping._check_state(regions, groups)
    by_region = {r.id: groups[r.group_id] for r in regions}
    assert by_region[2].is_background and by_region[3].is_background     # more than one background group
    assert by_region[2].id != by_region[3].id
    assert not by_region[0].is_background and not by_region[1].is_background
    assert not by_region[5].is_background                                # never in a background group, whatever its colour
    assert by_region[5].id != by_region[3].id
    assert by_region[4].is_background                                    # the hole stays backdrop (see-through)
    assert [r.backdrop for r in sorted(regions, key=lambda r: r.id)] == [False, False, True, True, True, False]
    # the border rule alone would have flagged just the wall's group
    _, plain, _ = grouping.group_regions(labels, albedo, _info(6))
    assert sum(g.is_background for g in plain) == 1
    # a regroup keeps the backdrop apart and flagged
    r2, g2, gm2 = grouping.regroup(regions, labels, albedo, None, 30.0)
    assert {g.is_background for g in g2} == {True, False}
    assert all(not g2[r.group_id].is_background for r in r2 if r.id in (0, 1, 5))
    assert all(g2[r.group_id].is_background for r in r2 if r.id in (2, 3, 4))


def test_a_border_set_region_off_the_border_leaves_the_background():
    """A region flagged only through the border set (not by the matte) that is not
    connected to the image border through flagged regions is object: it rejoins the
    nearest unflagged group or gets one of its own."""
    h, w = 60, 100
    labels = np.zeros((h, w), np.int32)
    labels[20:40, 40:60] = 1                                            # an island of the wall's colour inside the object
    labels[10:50, 10:90] = np.where(labels[10:50, 10:90] == 1, 1, 2)    # the object around it
    lab = np.zeros((h, w, 3), np.float32)
    lab[labels == 0] = (86.0, 0.0, 0.0)
    lab[labels == 1] = (86.0, 0.0, 0.0)
    lab[labels == 2] = (45.0, 60.0, 45.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    info = _info(3, {0: grouping.BG_MATTE, 1: grouping.BG_BORDER})       # the island is in the wall's border-set group ...
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    by_region = {r.id: groups[r.group_id] for r in regions}
    assert by_region[0].is_background and not by_region[1].is_background and not by_region[2].is_background
    assert by_region[1].id != by_region[0].id                            # ... but not connected to the border: object


# ------------------------------------------------------------------ names

def test_duplicate_names_get_a_suffix_and_stay_automatic():
    h, w = 40, 120
    labels = np.zeros((h, w), np.int32)
    labels[:, 40:80] = 1
    labels[:, 80:] = 2
    lab = np.zeros((h, w, 3), np.float32)
    lab[labels == 0] = (60.0, 40.0, 50.0)
    lab[labels == 1] = (60.0, 40.0, 50.0)                                # the same colour twice: two groups at dE 0? no: one
    lab[labels == 2] = (20.0, 0.0, 0.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    regions, groups, gm = grouping.group_regions(labels, albedo, _info(3), delta_e=0.5)
    # force two groups of one colour by splitting the first
    regions, groups, gm = grouping.move_regions(groups, regions, gm, labels, [1], groups[-1].id)
    lab_names = [g.name for g in groups]
    assert len(set(lab_names)) == len(lab_names)
    regions, groups, gm = grouping.move_regions(groups, regions, gm, labels, [1], groups[0].id)
    regions, groups, gm = grouping.split_group(groups, regions, gm, labels, albedo, groups[0].id, 2)
    names = [g.name for g in groups]
    assert len(set(names)) == len(names)
    base = grouping._auto_name(groups[0].albedo_lab)
    assert base in names and any(n.startswith(base + " ") for n in names) or len(groups) == 2
    # a suffixed automatic name is not a custom name: a rebuild may rename it
    g = groups[-1]
    g.name = grouping._auto_name(g.albedo_lab) + " 2"
    assert grouping._keep_name(g) is None
    g.name = "Racing stripe"
    assert grouping._keep_name(g) == "Racing stripe"


# ------------------------------------------------------------------ the cap with a backdrop side

def _capped_scene():
    """Two object colours (red, blue) and two backdrop tones (light and dark grey, more than
    dE 10 apart) on a matte that calls the surround backdrop."""
    h, w = 100, 160
    labels = np.zeros((h, w), np.int32)
    labels[:, 80:] = 1                                                   # the darker half of the backdrop
    labels[20:80, 10:70] = 2                                             # red
    labels[20:80, 90:150] = 3                                            # blue
    lab = np.zeros((h, w, 3), np.float32)
    lab[labels == 0] = (82.0, 0.0, 0.0)
    lab[labels == 1] = (40.0, 0.0, 0.0)
    lab[labels == 2] = (48.0, 62.0, 40.0)
    lab[labels == 3] = (35.0, 20.0, -60.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    return labels, albedo, _info(4, bg={0: 2, 1: 2})


def test_max_groups_caps_the_total_and_the_object_keeps_its_colours():
    labels, albedo, info = _capped_scene()
    _, g_auto, _ = grouping.group_regions(labels, albedo, info)
    assert len(g_auto) == 4 and sum(g.is_background for g in g_auto) == 2
    for cap in (1, 2, 3, 4, 8):
        regions, groups, gm = grouping.group_regions(labels, albedo, info, max_groups=cap)
        grouping._check_state(regions, groups)
        assert len(groups) <= cap, cap
        assert len(groups) == min(cap, 4)
        if cap == 1:
            assert not groups[0].is_background          # a lone group is never background (nothing to paint otherwise)
        if cap >= 3:
            obj = [g for g in groups if not g.is_background]
            assert len(obj) == 2 and not any(g.is_background for g in obj)      # red and blue survive
            assert all(set(g.region_ids) <= {0, 1} for g in groups if g.is_background)
        if cap == 2:
            assert sorted(g.is_background for g in groups) == [False, True]      # one object, one backdrop
    # a regroup of the same regions honours the cap the same way
    regions, groups, gm = grouping.group_regions(labels, albedo, info)
    r2, g2, gm2 = grouping.regroup(regions, labels, albedo, max_groups=2, delta_e=10.0)
    grouping._check_state(r2, g2)
    assert len(g2) == 2 and sorted(g.is_background for g in g2) == [False, True]
    # a regroup at max_groups 1 clusters everything into one group, which is not flagged
    # background (it is mostly backdrop): with the background ignored nothing could be painted
    r1, g1, gm1 = grouping.regroup(regions, labels, albedo, max_groups=1, delta_e=10.0)
    grouping._check_state(r1, g1)
    assert len(g1) == 1 and not g1[0].is_background and gm1.max() == 0
    assert sum(r.area for r in r1 if r.backdrop) > 0.5 * labels.size


# ------------------------------------------------------------------ split keeps the parent's flags

def test_split_keeps_the_parents_background_and_lock_flags():
    labels, albedo, info = _capped_scene()
    regions, groups, gm = grouping.group_regions(labels, albedo, info, delta_e=60.0)   # one backdrop group of two tones
    bg = [g for g in groups if g.is_background]
    assert len(bg) == 1 and sorted(bg[0].region_ids) == [0, 1]
    r2, g2, gm2 = grouping.split_group(groups, regions, gm, labels.copy(), albedo, bg[0].id, k=2)
    grouping._check_state(r2, g2)
    halves = [g for g in g2 if set(g.region_ids) & {0, 1}]
    assert len(halves) == 2 and all(g.is_background for g in halves)                  # region-level split
    assert all(r.backdrop for r in r2 if r.id in (0, 1))
    # a pixel-level split of a locked two-tone region: the new group is locked too
    labels2 = np.zeros((60, 80), np.int32)
    lab = np.zeros((60, 80, 3), np.float32)
    lab[:, :40] = (30.0, 40.0, 30.0)
    lab[:, 40:] = (70.0, 10.0, 60.0)
    alb2 = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    regions, groups, gm = grouping.group_regions(labels2, alb2, _info(1))
    groups[0].locked = True
    lab_map = labels2.copy()
    r3, g3, gm3 = grouping.split_group(groups, regions, gm, lab_map, alb2, 0, k=2)
    grouping._check_state(r3, g3)
    assert len(g3) == 2 and int(lab_map.max()) == 1 and all(g.locked for g in g3)
