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
