"""Segmentation tests: region hierarchy on synthetic masks and grouping invariants.

No SAM here: masks are hand-made. SLIC (skimage) does run on tiny images.
"""
from __future__ import annotations

import numpy as np
import pytest

from recolor import imageio
from recolor.segmentation import (
    build_regions, group_regions, merge_groups, move_regions, regroup, slic_labels, split_group,
)
from recolor.segmentation import DETAIL_PRESETS, REGION_PRESETS
from recolor.segmentation.grouping import cluster_colors
from recolor.segmentation.labelops import adjacency, compact, region_medians

H, W = 160, 200
GRAY = (0.45, 0.45, 0.45)
BLUE = (0.10, 0.20, 0.80)
RED = (0.85, 0.10, 0.10)
GREEN = (0.10, 0.70, 0.20)


def _scene(seed: int = 0):
    """Gray background, a blue square (30..130 x 40..140) with a red hole in its middle
    (60..100 x 70..110), and a green rectangle in the corner (5..25 x 160..195)."""
    rng = np.random.default_rng(seed)
    alb = np.empty((H, W, 3), np.float32)
    alb[:] = GRAY
    alb[30:130, 40:140] = BLUE
    alb[60:100, 70:110] = RED
    alb[5:25, 160:195] = GREEN
    alb = np.clip(alb + rng.normal(0, 0.01, alb.shape).astype(np.float32), 0, 1)
    # a shading gradient so the image is not identical to the albedo
    shade = np.linspace(0.6, 1.0, W, dtype=np.float32)[None, :, None]
    image = imageio.to_uint8(imageio.linear_to_srgb(alb * shade))
    return image, alb


def _mask(seg: np.ndarray, iou: float = 0.9) -> dict:
    ys, xs = np.nonzero(seg)
    return {"segmentation": seg, "area": int(seg.sum()),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())],
            "predicted_iou": iou, "stability_score": 0.95}


def _masks_with_hole_and_duplicate():
    square = np.zeros((H, W), bool)
    square[30:130, 40:140] = True
    square[60:100, 70:110] = False          # hole: SAM missed the red inset
    dup = np.zeros((H, W), bool)
    dup[31:130, 41:139] = True              # near-identical proposal of the same blue part
    dup[60:100, 70:110] = False
    green = np.zeros((H, W), bool)
    green[5:25, 160:195] = True
    return [_mask(square), _mask(dup, 0.8), _mask(green)]


def test_presets_exist():
    assert set(DETAIL_PRESETS) == {"fast", "balanced", "max"} == set(REGION_PRESETS)
    for p in DETAIL_PRESETS.values():
        for k in ("points_per_side", "crop_n_layers", "pred_iou_thresh", "stability_score_thresh",
                  "min_mask_region_area", "use_m2m", "points_per_batch"):
            assert k in p


def test_slic_labels_contiguous():
    image, _ = _scene()
    sp = slic_labels(image, 40)
    assert sp.dtype == np.int32 and sp.shape == (H, W)
    ids = np.unique(sp)
    assert ids[0] == 0 and ids[-1] == len(ids) - 1
    assert 10 <= len(ids) <= 80


def test_region_medians_matches_numpy():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 5, size=(20, 30)).astype(np.int32)
    labels[0, :10] = -1
    lab = rng.uniform(-100, 100, size=(20, 30, 3)).astype(np.float32)
    med = region_medians(labels, lab, 6)
    for i in range(5):
        pix = lab[labels == i]
        lower = np.sort(pix, axis=0)[(len(pix) - 1) // 2]
        assert np.allclose(med[i], lower, atol=0.011)
    assert np.isnan(med[5]).all()


def test_adjacency_and_compact():
    labels = np.array([[0, 0, 2], [0, 5, 2]], np.int32)
    pairs, blen = adjacency(labels, 6)
    assert pairs.tolist() == [[0, 2], [0, 5], [2, 5]]
    assert blen.tolist() == [1, 2, 1]
    out, mapping = compact(labels, 6)
    assert out.tolist() == [[0, 0, 1], [0, 2, 1]]
    assert mapping.tolist() == [0, -1, 1, -1, -1, 2]


def test_build_regions_hole_and_duplicate():
    image, alb = _scene()
    labels, info = build_regions(image, alb, _masks_with_hole_and_duplicate(), detail="balanced")
    assert labels.dtype == np.int32 and labels.shape == (H, W)
    assert labels.min() == 0 and labels.max() == len(info) - 1
    assert [d["id"] for d in info] == list(range(len(info)))
    assert {d["source"] for d in info} <= {"sam", "superpixel", "split"}
    # the duplicate proposal was dropped: only two SAM regions survive
    assert sum(d["source"] == "sam" for d in info) == 2
    # the hole is filled, by exactly one region that is not the blue square
    hole = labels[62:98, 72:108]
    assert len(np.unique(hole)) == 1
    blue_id = labels[40, 50]
    assert hole[0, 0] != blue_id
    assert info[hole[0, 0]]["source"] == "superpixel"
    # the blue square is intact and the background is one region
    assert (labels[32:58, 42:138] == blue_id).all()
    bg = labels[140:, :]
    assert len(np.unique(bg)) == 1 and len(np.unique(labels[:3, :150])) == 1
    assert len(info) <= 6
    for d in info:
        assert d["area"] > 0 and len(d["albedo_lab"]) == 3


def test_build_regions_no_masks_is_a_partition():
    image, alb = _scene()
    labels, info = build_regions(image, alb, [], detail="fast")
    assert labels.min() >= 0 and labels.max() == len(info) - 1
    assert all(d["source"] == "superpixel" for d in info)
    # same-colored superpixels coalesce: four colors -> a handful of regions
    assert len(info) <= 8


def test_bimodal_split():
    image, alb = _scene()
    alb = alb.copy()
    alb[30:130, 40:90] = RED
    alb[30:130, 90:140] = BLUE
    alb[60:100, 70:110] = np.where(np.arange(70, 110)[None, :, None] < 90, RED, BLUE)
    image = imageio.to_uint8(imageio.linear_to_srgb(alb))
    two_tone = np.zeros((H, W), bool)
    two_tone[30:130, 40:140] = True
    labels, info = build_regions(image, alb, [_mask(two_tone)], detail="balanced")
    assert sum(d["source"] == "split" for d in info) == 1
    left = labels[35:125, 45:85]
    right = labels[35:125, 95:135]
    assert len(np.unique(left)) == 1 and len(np.unique(right)) == 1
    assert left[0, 0] != right[0, 0]


def test_split_products_are_split_again(monkeypatch):
    """Step 2 examines each SAM region once, so the smaller half of a split kept whatever
    else it held (the BMW's nose shared a region with every blown specular of the bike).
    Step 5 re-runs the split on the regions step 2 produced: a blue / red / orange part
    comes out as three regions, a complete int32 partition."""
    from recolor.segmentation import hierarchy
    rng = np.random.default_rng(0)
    alb = np.empty((H, W, 3), np.float32)
    alb[:] = GRAY
    alb[30:130, 40:112] = BLUE
    alb[30:130, 112:136] = RED
    alb[30:130, 136:160] = (0.90, 0.45, 0.05)
    alb = np.clip(alb + rng.normal(0, 0.01, alb.shape).astype(np.float32), 0, 1)
    image = imageio.to_uint8(imageio.linear_to_srgb(alb * np.linspace(0.6, 1.0, W, dtype=np.float32)[None, :, None]))
    seg = np.zeros((H, W), bool)
    seg[30:130, 40:160] = True

    def parts(labels):
        return [np.unique(labels[40:120, x0:x1]).tolist() for x0, x1 in ((45, 105), (116, 132), (140, 156))]

    monkeypatch.setattr(hierarchy, "RESPLIT_ROUNDS", 0)
    labels, _ = build_regions(image, alb, [_mask(seg)], detail="balanced")
    blue, red, orange = parts(labels)
    assert red == orange and blue != red                     # without step 5: red + orange share a region
    monkeypatch.setattr(hierarchy, "RESPLIT_ROUNDS", 3)
    labels, info = build_regions(image, alb, [_mask(seg)], detail="balanced")
    blue, red, orange = parts(labels)
    assert len({blue[0], red[0], orange[0]}) == 3 and len(blue) == len(red) == len(orange) == 1
    assert labels.dtype == np.int32 and labels.min() == 0 and labels.max() == len(info) - 1
    assert [d["id"] for d in info] == list(range(len(info)))
    assert sum(d["source"] == "split" for d in info) == 2


def test_gradient_is_not_split():
    image, alb = _scene()
    alb = alb.copy()
    ramp = np.linspace(0.2, 0.8, 100, dtype=np.float32)[None, :, None]
    alb[30:130, 40:140] = ramp
    image = imageio.to_uint8(imageio.linear_to_srgb(alb))
    seg = np.zeros((H, W), bool)
    seg[30:130, 40:140] = True
    labels, info = build_regions(image, alb, [_mask(seg)], detail="balanced")
    assert not any(d["source"] == "split" for d in info)


def test_cluster_colors_threshold_and_cap():
    lab = np.array([[50, 60, 40], [52, 58, 42], [50, -60, 40], [90, 0, 0]], np.float32)
    areas = np.array([100, 50, 100, 400])
    cl = cluster_colors(lab, areas, delta_e=10.0)
    assert cl[0] == cl[1] and len(set(cl.tolist())) == 3
    assert len(set(cluster_colors(lab, areas, delta_e=10.0, max_groups=2).tolist())) == 2
    assert len(set(cluster_colors(lab, areas, delta_e=0.5).tolist())) == 4


def _grouped():
    image, alb = _scene()
    labels, info = build_regions(image, alb, _masks_with_hole_and_duplicate(), detail="balanced")
    regions, groups, group_map = group_regions(labels, alb, info)
    return image, alb, labels, regions, groups, group_map


def _check(regions, groups, group_map, labels):
    assert [g.id for g in groups] == list(range(len(groups)))
    assert group_map.dtype == np.int32 and group_map.shape == labels.shape
    assert group_map.min() == 0 and group_map.max() == len(groups) - 1
    assert sorted(r.id for r in regions) == list(range(len(regions)))
    assert abs(sum(g.area_frac for g in groups) - 1.0) < 1e-6
    lut = np.array([r.group_id for r in sorted(regions, key=lambda r: r.id)])
    assert np.array_equal(lut[labels], group_map)
    for g in groups:
        assert g.region_ids == sorted(r.id for r in regions if r.group_id == g.id)
        assert g.area == sum(r.area for r in regions if r.group_id == g.id)
        assert g.name and g.albedo_hex.startswith("#") and g.hue_family
    assert sum(g.is_background for g in groups) <= 1


def test_group_regions_invariants():
    image, alb, labels, regions, groups, group_map = _grouped()
    _check(regions, groups, group_map, labels)
    assert len(groups) == 4                                  # gray, blue, red, green
    assert [g.area for g in groups] == sorted((g.area for g in groups), reverse=True)
    assert groups[0].is_background and groups[0].hue_family == "neutral"
    families = {g.hue_family for g in groups}
    assert {"blue", "red", "green", "neutral"} <= families
    for r in regions:
        assert r.source in ("sam", "superpixel", "split")
        assert r.bbox[0] < r.bbox[2] and r.bbox[1] < r.bbox[3]


def test_regroup_cap_keeps_region_ids():
    image, alb, labels, regions, groups, group_map = _grouped()
    r2, g2, gm2 = regroup(regions, labels, alb, max_groups=2, delta_e=10.0)
    _check(r2, g2, gm2, labels)
    assert len(g2) == 2
    assert [r.id for r in r2] == [r.id for r in regions]
    r3, g3, gm3 = regroup(regions, labels, alb, max_groups=None, delta_e=200.0)
    assert len(g3) == 1 and gm3.max() == 0


def test_merge_split_move():
    image, alb, labels, regions, groups, group_map = _grouped()
    blue = next(g for g in groups if g.hue_family == "blue")
    red = next(g for g in groups if g.hue_family == "red")
    n_regions = len(regions)

    # merge: one fewer group, region ids untouched, ids contiguous
    r1, g1, gm1 = merge_groups(groups, regions, group_map, labels, [blue.id, red.id])
    _check(r1, g1, gm1, labels)
    assert len(g1) == len(groups) - 1 and len(r1) == n_regions
    merged = g1[min(blue.id, red.id)]
    assert set(merged.region_ids) == set(blue.region_ids) | set(red.region_ids)

    # split the merged (two-region) group at region level: back to the original count
    r2, g2, gm2 = split_group(g1, r1, gm1, labels.copy(), alb, merged.id, k=2)
    _check(r2, g2, gm2, labels)
    assert len(g2) == len(groups) and len(r2) == n_regions
    assert g2[-1].id == len(g2) - 1

    # split a single-region group at pixel level: new region ids are appended
    lab2 = labels.copy()
    green = next(g for g in g2 if g.hue_family == "green")
    assert len(green.region_ids) == 1
    r3, g3, gm3 = split_group(g2, r2, gm2, lab2, alb, green.id, k=2)
    _check(r3, g3, gm3, lab2)
    assert len(r3) >= n_regions and lab2.max() == len(r3) - 1
    assert sorted(r.id for r in r3)[:n_regions] == list(range(n_regions))

    # move: the region moves, an emptied group disappears, flags survive
    g2[0].locked = True
    g2[0].name = "My gray"
    r4, g4, gm4 = move_regions(g2, r2, gm2, labels, green.region_ids, g2[0].id)
    _check(r4, g4, gm4, labels)
    assert len(g4) == len(g2) - 1
    assert g4[0].locked and g4[0].name == "My gray"
    assert set(green.region_ids) <= set(g4[0].region_ids)


def test_group_regions_rejects_incomplete_labels():
    _, alb = _scene()
    labels = np.zeros((H, W), np.int32)
    labels[0, 0] = -1
    with pytest.raises(ValueError):
        group_regions(labels, alb, [])


@pytest.mark.parametrize("shape", [(1, 64), (64, 1), (1, 1), (2, 2), (3, 40)])
def test_build_regions_degenerate_shapes(shape):
    """Thin / tiny images (1xW in particular) must still come out fully labelled: the
    guided-filter edge snap is skipped instead of crashing on a degenerate guide."""
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, shape + (3,), np.uint8)
    alb = imageio.srgb_to_linear(imageio.to_float(image))
    for detail in ("fast", "balanced", "max"):
        labels, info = build_regions(image, alb, [], detail)
        assert labels.shape == shape and labels.min() >= 0 and labels.max() == len(info) - 1


def test_split_flat_group_is_noop():
    """Splitting a group that is genuinely one colour (flat + sensor noise) must not
    invent two identical swatches: state and label map come back unchanged."""
    image, alb, labels, regions, groups, group_map = _grouped()
    green = next(g for g in groups if g.hue_family == "green")
    assert len(green.region_ids) == 1
    lab2 = labels.copy()
    for k in (2, 3):
        r2, g2, gm2 = split_group(groups, regions, group_map, lab2, alb, green.id, k=k)
        _check(r2, g2, gm2, lab2)
        assert len(g2) == len(groups) and len(r2) == len(regions)
        assert np.array_equal(lab2, labels) and np.array_equal(gm2, group_map)
        assert [g.name for g in g2] == [g.name for g in groups]
    # region-level branch: a multi-region group of one colour is not split either
    gray = next(g for g in groups if g.is_background)
    if len(gray.region_ids) >= 2:
        r3, g3, gm3 = split_group(groups, regions, group_map, labels.copy(), alb, gray.id, k=2)
        assert len(g3) == len(groups) and np.array_equal(gm3, group_map)


def test_split_two_tone_region_at_pixel_level():
    """A single region holding two clearly different colours is cut at the pixel level:
    a new region id is appended, a new group created, existing ids untouched. The label
    map is hand-made so the region-level branch cannot take over."""
    image, alb = _scene()
    alb = alb.copy()
    alb[5:25, 160:178] = (0.85, 0.75, 0.10)        # left half of the green part turns yellow
    labels = np.zeros((H, W), np.int32)
    labels[30:130, 40:140] = 1
    labels[5:25, 160:195] = 2                       # one region, two tones
    regions, groups, group_map = group_regions(labels, alb, [], delta_e=10.0)
    _check(regions, groups, group_map, labels)
    gid = next(r.group_id for r in regions if r.id == 2)
    assert [r.id for r in regions if r.group_id == gid] == [2]
    n_regions, n_groups = len(regions), len(groups)
    lab2 = labels.copy()
    r2, g2, gm2 = split_group(groups, regions, group_map, lab2, alb, gid, k=2)
    _check(r2, g2, gm2, lab2)
    assert len(g2) == n_groups + 1 and len(r2) == n_regions + 1
    assert sorted(r.id for r in r2)[:n_regions] == list(range(n_regions))
    assert g2[-1].id == n_groups and lab2.max() == n_regions
    assert [g.id for g in g2[:n_groups]] == [g.id for g in groups]
    hues = {g.hue_family for g in g2}
    assert "green" in hues and (("yellow" in hues) or ("orange" in hues))
    new_region = next(r for r in r2 if r.id == n_regions)
    assert new_region.source == "split" and 16 <= new_region.area <= 20 * 35
    # a piece cut off lettering stays lettering (the pruning's exemption, the island treatment)
    info = [{"id": 0, "source": "sam"}, {"id": 1, "source": "sam"}, {"id": 2, "source": "text"}]
    regions, groups, group_map = group_regions(labels, alb, info, delta_e=10.0)
    gid = next(r.group_id for r in regions if r.id == 2)
    lab3 = labels.copy()
    r3, _, _ = split_group(groups, regions, group_map, lab3, alb, gid, k=2)
    assert next(r for r in r3 if r.id == n_regions).source == "text"


# ---------------------------------------------------------------------- part recovery

def _pocket_scene():
    """A dark grey machine (region 0, 120 x 160) holding a small gold part SAM never
    proposed (20 x 20 at 50..70 x 60..80), on a light backdrop (region 1)."""
    h, w = 140, 200
    labels = np.ones((h, w), np.int32)
    labels[10:130, 20:180] = 0
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (85.0, 0.0, 0.0)
    lab[labels == 0] = (20.0, 0.5, 1.0)
    lab[50:70, 60:80] = (35.0, 6.0, 32.0)                           # the gold part
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    image = imageio.to_uint8(imageio.linear_to_srgb(albedo))
    info = [{"id": 0, "source": "sam", "confidence": 0.9}, {"id": 1, "source": "sam", "confidence": 0.9}]
    return image, albedo, labels, info


def _cand(full: np.ndarray, score: float = 0.9, clipped: bool = False, x0: int = 0, y0: int = 0, size: int = 96) -> dict:
    return {"mask": full[y0:y0 + size, x0:x0 + size].copy(), "x0": x0, "y0": y0, "score": score, "clipped": clipped}


def test_find_pockets_finds_a_coloured_part_inside_a_neutral_region():
    from recolor.segmentation.hierarchy import find_pockets
    image, albedo, labels, info = _pocket_scene()
    pockets = find_pockets(labels, imageio.linear_to_lab(albedo))
    assert len(pockets) == 1
    p = pockets[0]
    assert p["region"] == 0 and 300 <= p["px"] <= 400
    x, y = p["point"]
    assert 55 <= x <= 75 and 45 <= y <= 65                          # a point well inside the part
    # a coloured host region is not searched (its pockets are its own colours)
    lab2 = imageio.linear_to_lab(albedo).copy()
    lab2[labels == 0] = (40.0, 50.0, 30.0)
    assert find_pockets(labels, lab2) == []


def test_recover_parts_stamps_the_part_sam_returns_and_rejects_the_rest():
    from recolor.segmentation.hierarchy import recover_parts
    image, albedo, labels, info = _pocket_scene()
    part = np.zeros(labels.shape, bool)
    part[50:70, 60:80] = True
    machine = labels == 0
    calls = []

    def prompter(img, points):
        calls.append(list(points))
        # per point: the part itself, a clipped mask of the whole machine, a low-score blob
        return [[_cand(part, 0.95, x0=30, y0=20), _cand(machine, 0.99, clipped=True, x0=30, y0=20),
                 _cand(part, 0.3, x0=30, y0=20)] for _ in points]

    before = labels.copy()
    out, out_info, n = recover_parts(image, albedo, labels, info, prompter)
    assert len(calls) == 1 and n == 1
    assert out.dtype == np.int32 and out.min() == 0 and int(out.max()) + 1 == len(out_info) == 3
    new_id = int(out[60, 70])
    assert new_id not in (int(out[20, 30]), int(out[0, 0]))
    assert (out == new_id).sum() == 400 and out_info[new_id]["source"] == "prompt"
    assert out_info[new_id]["area"] == 400 and out_info[int(out[20, 30])]["area"] == int(machine.sum()) - 400
    assert np.array_equal(labels, before)                           # the input is not modified


def test_recover_parts_keeps_the_partition_when_nothing_is_a_part():
    from recolor.segmentation.hierarchy import recover_parts
    image, albedo, labels, info = _pocket_scene()
    grey = np.zeros(labels.shape, bool)
    grey[40:80, 50:90] = True                                        # neutral median: a chrome part, not the gold

    def prompter(img, points):
        return [[_cand(grey, 0.95, x0=30, y0=20)] for _ in points]

    out, out_info, n = recover_parts(image, albedo, labels, info, prompter)
    assert n == 0 and np.array_equal(out, labels) and len(out_info) == 2
    # no pocket at all: the prompter is never asked
    flat = np.full_like(albedo, 0.2)
    out, _, n = recover_parts(image, flat, labels, info, lambda img, pts: pytest.fail("prompted"))
    assert n == 0 and np.array_equal(out, labels)


def _colour_pocket_scene():
    """A yellow paint region (0) made of a large panel plus two pieces far from it that SAM's
    colour mode swept in: a gold part (darker, duller: another material) and a sliver of the
    same yellow seen through a gap; a grey machine (1) around the pieces; a backdrop (2)."""
    h, w = 140, 220
    labels = np.full((h, w), 2, np.int32)
    labels[10:130, 110:210] = 1                                      # the machine
    labels[10:130, 10:100] = 0                                       # the panel
    labels[40:60, 130:152] = 0                                       # the gold part (440 px)
    labels[90:100, 170:180] = 0                                      # a sliver of the paint (100 px)
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (88.0, 0.0, 0.0)
    lab[labels == 1] = (22.0, 0.5, 1.5)
    lab[labels == 0] = (78.0, 4.0, 79.0)
    lab[40:60, 130:152] = (60.0, 9.5, 37.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    image = imageio.to_uint8(imageio.linear_to_srgb(albedo))
    info = [{"id": i, "source": "split" if i == 0 else "sam", "confidence": 0.9} for i in range(3)]
    return image, albedo, labels, info


def test_find_colour_pockets_finds_an_off_colour_piece_of_a_chromatic_region():
    from recolor.segmentation.hierarchy import find_colour_pockets, find_pockets
    image, albedo, labels, info = _colour_pocket_scene()
    lab = imageio.linear_to_lab(albedo)
    pockets = find_colour_pockets(labels, lab)
    assert len(pockets) == 1                                         # not the panel, not the paint sliver
    p = pockets[0]
    assert p["region"] == 0 and p["kind"] == "colour" and p["px"] == 440
    x, y = p["point"]
    assert 130 <= x < 152 and 40 <= y < 60
    assert find_pockets(labels, lab) == []                           # the neutral-host search does not see it


def test_recover_parts_gives_an_off_colour_piece_its_own_region():
    from recolor.segmentation.hierarchy import recover_parts
    image, albedo, labels, info = _colour_pocket_scene()
    gold = np.zeros(labels.shape, bool)
    gold[38:62, 128:154] = True                                      # SAM's part: the piece and its rim
    grey = np.zeros(labels.shape, bool)
    grey[30:70, 120:160] = True                                      # mostly machine: a neutral median

    def prompter(img, points):
        return [[_cand(grey, 0.97, x0=100, y0=0, size=100), _cand(gold, 0.9, x0=100, y0=0, size=100)]
                for _ in points]

    out, out_info, n = recover_parts(image, albedo, labels, info, prompter)
    assert n == 1 and int(out.max()) + 1 == len(out_info) == 4
    new_id = int(out[50, 140])
    assert out_info[new_id]["source"] == "part" and (out == new_id).sum() == int(gold.sum())
    assert int(out[50, 50]) != new_id and int(out[95, 175]) == int(out[50, 50])   # panel and sliver stay paint

    # a mask that covers the piece but is mostly the host's own colour is not a part of its own
    from recolor.segmentation import hierarchy
    lab = imageio.linear_to_lab(albedo)
    pocket = hierarchy.find_colour_pockets(labels, lab)[0]
    host = np.median(lab[labels == 0], axis=0)
    around = np.zeros(labels.shape, bool)
    around[30:70, 120:160] = True
    lab_y = lab.copy()
    lab_y[around & (labels == 1)] = (78.0, 4.0, 79.0)                # the paint's yellow all around it
    assert hierarchy._accept_part(_cand(around, 0.96, x0=100, y0=0, size=100), pocket, labels, lab_y, host) is None
    assert hierarchy._accept_part(_cand(gold, 0.9, x0=100, y0=0, size=100), pocket, labels, lab, host) is not None
    small = np.zeros(labels.shape, bool)
    small[45:50, 135:141] = True                                     # 30 px of the 440 px piece: not the part
    assert hierarchy._accept_part(_cand(small, 0.9, x0=100, y0=0, size=100), pocket, labels, lab, host) is None
