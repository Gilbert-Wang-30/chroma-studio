"""Suggest which palette color each color group should receive.

Every strategy builds a ``G x N`` cost matrix (eligible groups x palette colors), solves
it with the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``) so that no
two groups share a color while there are colors to spare, then gives every group left
over (``G > N``) the palette color nearest to its own albedo in Lab.  Locked groups and
the background are mapped to ``None`` ("keep the original") when the flags say so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import colornames, imageio
from .types import ColorGroup, Mapping, PaletteColor

STRATEGIES = ["balanced", "area", "luminance", "hue", "contrast"]
NEUTRAL_CHROMA = 9.0
# How strongly a neutral group is steered toward the palette's neutral colors, per
# strategy.  Small for the value-structure strategies so lightness order still rules.
NEUTRAL_PENALTY = {"balanced": 0.30, "area": 0.30, "hue": 1.00, "luminance": 0.05, "contrast": 0.05}


@dataclass(frozen=True)
class _Color:
    hex: str
    lab: tuple[float, float, float]
    weight: float

    @property
    def chroma(self) -> float:
        return math.hypot(self.lab[1], self.lab[2])

    @property
    def is_neutral(self) -> bool:
        return self.chroma < NEUTRAL_CHROMA


def _as_colors(colors: list[PaletteColor] | list[str]) -> list[_Color]:
    out: list[_Color] = []
    for i, c in enumerate(colors):
        if isinstance(c, PaletteColor):
            hx = imageio.rgb01_to_hex(imageio.hex_to_rgb01(c.hex))
            out.append(_Color(hx, tuple(float(v) for v in c.lab), float(c.weight)))
        else:
            hx = imageio.rgb01_to_hex(imageio.hex_to_rgb01(str(c)))
            out.append(_Color(hx, imageio.hex_to_lab(hx), float(len(colors) - i)))
    return out


def hue_distance(lab_a, lab_b) -> float:
    """Shortest angular distance between two Lab hue angles, in degrees [0, 180]."""
    ha = math.degrees(math.atan2(float(lab_a[2]), float(lab_a[1])))
    hb = math.degrees(math.atan2(float(lab_b[2]), float(lab_b[1])))
    d = abs(ha - hb) % 360.0
    return min(d, 360.0 - d)


def _group_is_neutral(g: ColorGroup) -> bool:
    return g.hue_family == "neutral" and math.hypot(g.albedo_lab[1], g.albedo_lab[2]) < NEUTRAL_CHROMA


def _ranks(values: list[float], descending: bool = True) -> np.ndarray:
    """0-based rank of each value (stable: ties keep list order)."""
    order = sorted(range(len(values)), key=lambda i: (-values[i] if descending else values[i], i))
    ranks = np.empty(len(values), np.float64)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def _cost_matrix(groups: list[ColorGroup], colors: list[_Color], strategy: str) -> np.ndarray:
    G, N = len(groups), len(colors)
    gL = np.array([g.albedo_lab[0] for g in groups], np.float64)
    cL = np.array([c.lab[0] for c in colors], np.float64)
    dL = np.abs(gL[:, None] - cL[None, :]) / 100.0                                   # (G, N)
    g_rank = _ranks([float(g.area) for g in groups]) / max(G - 1, 1)                 # 0..1 by area
    c_rank = _ranks([c.weight for c in colors]) / max(N - 1, 1)                       # 0..1 by weight
    d_rank = np.abs(g_rank[:, None] - c_rank[None, :])
    hue = np.array([[hue_distance(g.albedo_lab, c.lab) / 180.0 for c in colors] for g in groups], np.float64)
    g_neutral = np.array([_group_is_neutral(g) for g in groups], bool)
    c_neutral = np.array([c.is_neutral for c in colors], bool)
    # Hue is undefined for neutrals: treat a neutral/chromatic pair as a full hue miss
    # and a neutral/neutral pair as a hue match.
    both_neutral = g_neutral[:, None] & c_neutral[None, :]
    one_neutral = g_neutral[:, None] ^ c_neutral[None, :]
    hue = np.where(both_neutral, 0.0, np.where(one_neutral, 1.0, hue))

    if strategy == "area":
        # Pair in rank order; unnormalized rank difference so identity pairing is optimal.
        cost = np.abs(_ranks([float(g.area) for g in groups])[:, None]
                      - _ranks([c.weight for c in colors])[None, :]) / max(G, N)
    elif strategy == "luminance":
        gl_rank = _ranks(gL.tolist(), descending=False) / max(G - 1, 1)
        cl_rank = _ranks(cL.tolist(), descending=False) / max(N - 1, 1)
        cost = np.abs(gl_rank[:, None] - cl_rank[None, :]) + 1e-3 * dL
    elif strategy == "hue":
        cost = hue + 0.02 * dL
    elif strategy == "contrast":
        cost = dL + 1e-3 * hue
    elif strategy == "balanced":
        cost = 0.45 * d_rank + 0.45 * dL + 0.10 * hue
    else:
        raise ValueError(f"unknown mapping strategy {strategy!r}; choose one of {STRATEGIES}")

    # Neutral groups prefer neutral palette colors when the palette has any.
    if c_neutral.any() and g_neutral.any():
        penalty = NEUTRAL_PENALTY[strategy]
        cost = cost + penalty * (g_neutral[:, None] & ~c_neutral[None, :])
    return cost


def _nearest_color(g: ColorGroup, colors: list[_Color]) -> _Color:
    labs = np.asarray([c.lab for c in colors], np.float32)
    d = imageio.delta_e(np.repeat(np.asarray(g.albedo_lab, np.float32)[None], len(colors), 0), labs)
    return colors[int(np.argmin(d))]


def suggest_mapping(groups: list[ColorGroup], colors: list[PaletteColor] | list[str],
                    strategy: str = "balanced", keep_background: bool = True,
                    keep_locked: bool = True) -> Mapping:
    """Propose a ``{group_id: hex | None}`` mapping.

    Guarantees: the result has an entry for every group in ``groups``; locked groups
    (``keep_locked``) and background groups (``keep_background``) map to ``None``;
    when the number of eligible groups is at most the number of colors every eligible
    group gets a *different* color; with more groups than colors the Hungarian matches
    are made first and the remaining groups take the color nearest their albedo
    (CIEDE2000).  The output is deterministic for identical inputs.  ``colors`` may be
    ``PaletteColor`` records (weights taken from them) or hex strings (first = heaviest).
    Raises ``ValueError`` for an unknown strategy, for duplicate group ids (the
    uniqueness guarantee could not hold) and for a non-finite group or palette Lab.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown mapping strategy {strategy!r}; choose one of {STRATEGIES}")
    ids = [g.id for g in groups]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate group ids {dupes}: every ColorGroup needs a unique id")
    for g in groups:
        if len(g.albedo_lab) != 3 or not all(math.isfinite(float(v)) for v in g.albedo_lab):
            raise ValueError(f"group {g.id} has a non-finite albedo_lab {tuple(g.albedo_lab)!r}")
    cols = _as_colors(colors)
    for c in cols:
        if not all(math.isfinite(v) for v in c.lab):
            raise ValueError(f"palette color {c.hex} has a non-finite lab {c.lab!r}")
    mapping: Mapping = {g.id: None for g in groups}
    eligible = [g for g in groups
                if not (keep_locked and g.locked) and not (keep_background and g.is_background)]
    if not eligible or not cols:
        return mapping
    cost = _cost_matrix(eligible, cols, strategy)
    rows, cols_idx = linear_sum_assignment(cost)
    assigned: set[int] = set()
    for r, c in zip(rows.tolist(), cols_idx.tolist()):
        mapping[eligible[r].id] = cols[c].hex
        assigned.add(r)
    for i, g in enumerate(eligible):
        if i not in assigned:
            mapping[g.id] = _nearest_color(g, cols).hex
    return mapping


def describe_strategy(strategy: str) -> str:
    """One-line human description of a strategy for UI tooltips."""
    return {
        "balanced": "Match by size, lightness and hue together",
        "area": "Biggest part gets the palette's main color, and so on",
        "luminance": "Keep the light/dark structure of the original",
        "hue": "Closest hue for each part; neutrals stay neutral",
        "contrast": "Preserve light-dark relationships between parts",
    }.get(strategy, strategy)


def group_hue_family(g: ColorGroup) -> str:
    """Hue family of a group's albedo, recomputed from Lab (ignores a stale field)."""
    return colornames.hue_family(g.albedo_lab)
