"""Mapping tests: uniqueness when G <= N, determinism, locked/background handling,
luminance ordering preserved by ``contrast``, overflow when G > N, neutral preference."""
from __future__ import annotations

import pytest

from recolor import colornames, imageio, mapping
from recolor.types import ColorGroup, PaletteColor


def group(gid: int, hex_color: str, area: int, locked: bool = False, background: bool = False) -> ColorGroup:
    lab = imageio.hex_to_lab(hex_color)
    return ColorGroup(id=gid, name=colornames.nearest_name(lab), albedo_lab=lab, albedo_hex=hex_color,
                      area=area, area_frac=area / 10000.0, region_ids=[gid], hue_family=colornames.hue_family(lab),
                      locked=locked, is_background=background)


def palette(hexes: list[str], weights: list[float] | None = None) -> list[PaletteColor]:
    weights = weights or [float(len(hexes) - i) for i in range(len(hexes))]
    tot = sum(weights)
    return [PaletteColor(hex=h, lab=imageio.hex_to_lab(h), weight=w / tot, name=colornames.nearest_name(imageio.hex_to_lab(h)))
            for h, w in zip(hexes, weights)]


GROUPS = [
    group(0, "#b0171f", 5000),   # cherry red, biggest
    group(1, "#f2f0eb", 2500),   # off-white
    group(2, "#1a1a1a", 1500),   # black
    group(3, "#4682b4", 1000),   # steel blue
]
HEXES = ["#1f5fb6", "#ffd400", "#f4f4f4", "#101010", "#ff8c1a"]


@pytest.mark.parametrize("strategy", mapping.STRATEGIES)
def test_unique_when_fewer_groups_than_colors(strategy):
    m = mapping.suggest_mapping(GROUPS, palette(HEXES), strategy=strategy, keep_background=False)
    assert set(m) == {0, 1, 2, 3}
    assigned = [v for v in m.values()]
    assert None not in assigned
    assert len(set(assigned)) == 4
    assert set(assigned) <= set(HEXES)


@pytest.mark.parametrize("strategy", mapping.STRATEGIES)
def test_deterministic(strategy):
    a = mapping.suggest_mapping(GROUPS, palette(HEXES), strategy=strategy)
    b = mapping.suggest_mapping(list(GROUPS), palette(list(HEXES)), strategy=strategy)
    assert a == b


def test_hex_strings_accepted_and_normalized():
    m = mapping.suggest_mapping(GROUPS, ["#ABC", "1f5fb6", "#ffd400", "#101010"], strategy="area")
    assert set(m.values()) <= {"#aabbcc", "#1f5fb6", "#ffd400", "#101010"}
    assert len(set(m.values())) == 4


def test_locked_and_background_map_to_none():
    gs = [group(0, "#b0171f", 5000), group(1, "#dddddd", 4000, background=True), group(2, "#1a1a1a", 500, locked=True)]
    m = mapping.suggest_mapping(gs, palette(HEXES))
    assert m[1] is None and m[2] is None and m[0] is not None
    m2 = mapping.suggest_mapping(gs, palette(HEXES), keep_background=False, keep_locked=False)
    assert all(v is not None for v in m2.values()) and len(set(m2.values())) == 3
    m3 = mapping.suggest_mapping(gs, palette(HEXES), keep_background=False, keep_locked=True)
    assert m3[1] is not None and m3[2] is None


def test_area_pairs_in_rank_order():
    gs = [group(0, "#777777", 100), group(1, "#777777", 900), group(2, "#777777", 500)]
    cols = palette(["#ff0000", "#00ff00", "#0000ff"], [0.2, 0.5, 0.3])
    m = mapping.suggest_mapping(gs, cols, strategy="area")
    assert m == {1: "#00ff00", 2: "#0000ff", 0: "#ff0000"}


def test_contrast_preserves_luminance_order():
    gs = [group(0, "#101010", 100), group(1, "#505050", 200), group(2, "#909090", 300), group(3, "#e0e0e0", 400)]
    cols = palette(["#ffe0a0", "#c04030", "#203050", "#803030"])
    m = mapping.suggest_mapping(gs, cols, strategy="contrast")
    order = [m[g.id] for g in sorted(gs, key=lambda g: g.albedo_lab[0])]
    ls = [imageio.hex_to_lab(h)[0] for h in order]
    assert ls == sorted(ls)
    ml = mapping.suggest_mapping(gs, cols, strategy="luminance")
    ls2 = [imageio.hex_to_lab(ml[g.id])[0] for g in sorted(gs, key=lambda g: g.albedo_lab[0])]
    assert ls2 == sorted(ls2)


def test_luminance_spreads_over_the_palette_when_fewer_groups():
    gs = [group(0, "#202020", 100), group(1, "#e0e0e0", 100)]
    cols = palette(["#101010", "#404040", "#808080", "#c0c0c0", "#f0f0f0"])
    m = mapping.suggest_mapping(gs, cols, strategy="luminance")
    assert m == {0: "#101010", 1: "#f0f0f0"}


def test_hue_matches_nearest_hue_and_neutrals():
    gs = [group(0, "#d62828", 100), group(1, "#1f5fd6", 100), group(2, "#888888", 100)]
    cols = palette(["#4169e1", "#c0c0c0", "#ff2400"])
    m = mapping.suggest_mapping(gs, cols, strategy="hue")
    assert m == {0: "#ff2400", 1: "#4169e1", 2: "#c0c0c0"}


def test_more_groups_than_colors_all_assigned():
    gs = [group(i, h, 1000 - i * 50) for i, h in enumerate(
        ["#b0171f", "#f2f0eb", "#1a1a1a", "#4682b4", "#2e8b3d", "#ff8c1a", "#8e44ad"])]
    cols = palette(["#1f5fb6", "#ffd400", "#f4f4f4"])
    for strategy in mapping.STRATEGIES:
        m = mapping.suggest_mapping(gs, cols, strategy=strategy)
        assert all(m[g.id] is not None for g in gs)
        assert set(m.values()) == set(c.hex for c in cols)      # every color used once before reuse
    # nearest-in-Lab for the overflow: a white group left over gets the white color.
    gs2 = [group(0, "#d62828", 900), group(1, "#1f5fd6", 800), group(2, "#fafafa", 10)]
    m2 = mapping.suggest_mapping(gs2, palette(["#ff2400", "#4169e1"]), strategy="hue")
    assert m2[2] in ("#ff2400", "#4169e1")


def test_neutral_group_prefers_neutral_color():
    gs = [group(0, "#c0392b", 3000), group(1, "#9a9a9a", 2900)]
    cols = palette(["#ff8c1a", "#d3d3d3"])
    for strategy in ("balanced", "area", "hue"):
        m = mapping.suggest_mapping(gs, cols, strategy=strategy)
        assert m[1] == "#d3d3d3", strategy
        assert m[0] == "#ff8c1a"


def test_balanced_uses_size_and_lightness():
    gs = [group(0, "#f5f5f5", 6000), group(1, "#2a2a2a", 1000)]
    cols = palette(["#f0e6d0", "#202840"])
    m = mapping.suggest_mapping(gs, cols, strategy="balanced")
    assert m == {0: "#f0e6d0", 1: "#202840"}


def test_edge_cases_and_errors():
    assert mapping.suggest_mapping([], palette(HEXES)) == {}
    assert mapping.suggest_mapping(GROUPS, []) == {0: None, 1: None, 2: None, 3: None}
    with pytest.raises(ValueError):
        mapping.suggest_mapping(GROUPS, palette(HEXES), strategy="random")
    assert mapping.hue_distance((50, 10, 0), (50, -10, 0)) == pytest.approx(180.0)
    assert mapping.hue_distance((50, 10, 0), (50, 10, 0)) == pytest.approx(0.0)


def test_duplicate_ids_and_non_finite_lab_are_rejected():
    dup = [group(0, "#b0171f", 5000), group(0, "#f2f0eb", 2500)]
    with pytest.raises(ValueError, match="duplicate group ids"):
        mapping.suggest_mapping(dup, palette(HEXES))
    bad = group(1, "#4682b4", 1000)
    bad.albedo_lab = (float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError, match="non-finite"):
        mapping.suggest_mapping([GROUPS[0], bad], palette(HEXES))
    nan_col = palette(["#1f5fb6", "#ffd400"])
    nan_col[1].lab = (float("inf"), 0.0, 0.0)
    with pytest.raises(ValueError, match="non-finite"):
        mapping.suggest_mapping(GROUPS, nan_col)
