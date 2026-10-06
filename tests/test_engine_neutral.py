"""Regression tests for the engine's neutral-source rules (recolor/engine.py, ``NEUTRAL_*``, ``GLINT_*``,
``ISLAND_*``, ``EXPOSURE_*``): a white or near-white paint repainted to a colour. No models, no network.

Each scene is a white panel whose decomposition behaves like the Careaga layers on white paint (an
albedo below a real white, a broad achromatic residual floor that follows the shading) beside a
saturated red panel and a black glossy panel, so every test can also check that a saturated source is
untouched. "The saturated-paint rules" are the same renderer with every group's neutral weight forced
to 0. For a source of CIELAB chroma 8 or more that is exactly the engine before the neutral rules
existed; below chroma 8 rule 8 also brings the source's reflections in with its hue confidence (the old
engine switched them on at full weight at chroma 2), which the saturated-paint renderer keeps
(test_a_near_neutral_sources_reflections_come_in_with_its_hue). The rules must apply to white paint
only (a glossy grey keeps its reflections), fade out as the target nears the source colour, and come in
continuously: no step of the source paint, its gloss, the target or a glint's standout may flip a rule on
at once. A glint the old hard tests kept whole (1.5x its ring, 1.44x the shading's step, 8 px) is kept
whole: the ramps lie below those cuts.
"""
from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np
import pytest
import torch

from recolor import engine, imageio
from recolor.types import ColorGroup, RenderOptions

H, W = 64, 96
WHITE, RED, BLACK, LETTER = 0, 1, 2, 3
NAVY, PASTEL = "#123f9e", "#7fb2ff"
LIT = (slice(40, 60), slice(8, 40))          # lit rows of the white panel, away from its edges
GLINT = (slice(22, 27), slice(20, 25))       # a 5 x 5 px sensor-clipped glint on the white panel


def _group(gid, albedo, gm):
    lab = tuple(float(v) for v in np.median(imageio.linear_to_lab(albedo[gm == gid]), axis=0))
    area = int((gm == gid).sum())
    return ColorGroup(id=gid, name=f"g{gid}", albedo_lab=lab, albedo_hex=imageio.lab_to_hex(lab), area=area,
                      area_frac=area / float(H * W), region_ids=[gid], hue_family="neutral")


def _clip_residual(albedo, shading, residual):
    """The residual that makes albedo * shading + residual a photo clipped at 1 (sensor clipping)."""
    lin = albedo * shading + residual
    return np.where(lin > 1.0, 1.0 - albedo * shading, residual).astype(np.float32)


def white_scene(white_albedo: float = 0.72, floor_share: float = 0.15, sun: float = 1.25):
    """(albedo, shading, residual, group_map, groups): a white panel (left, 0..47), a red panel (right,
    48..71) and a black glossy panel (72..95). The white panel's residual is an achromatic floor of
    ``floor_share`` x its diffuse (the leftover of an albedo below a real white); the black panel's is a
    sheen streak many times its diffuse."""
    gm = np.full((H, W), WHITE, np.int32)
    gm[:, 48:72] = RED
    gm[:, 72:] = BLACK
    albedo = np.zeros((H, W, 3), np.float32)
    albedo[gm == WHITE] = white_albedo
    albedo[gm == RED] = imageio.srgb_to_linear(imageio.hex_to_rgb01("#c0392b"))
    albedo[gm == BLACK] = 0.03
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    shade = 0.45 + (sun - 0.45) * (yy / (H - 1))
    shading = np.repeat(shade[..., None], 3, axis=2).astype(np.float32)
    residual = np.zeros((H, W, 3), np.float32)
    residual[gm == WHITE] = floor_share * (albedo * shading)[gm == WHITE]
    streak = (gm == BLACK) & (np.abs(xx - 84) < 2)
    residual[streak] = 0.35
    residual = _clip_residual(albedo, shading, residual)
    groups = [_group(g, albedo, gm) for g in (WHITE, RED, BLACK)]
    return albedo, shading, residual, gm, groups


def glint_scene():
    """The white scene with a small clipped glint on the white panel (in the residual, the shading smooth
    under it: the decomposition does not explain it) and a clipped lit bevel (a band the *shading* makes
    clip, 3 x 20 px), each standing well out of the paint around it."""
    albedo, shading, residual, gm, groups = white_scene()
    shading = shading.copy()
    shading[34:37, 12:32] *= 1.8                       # the lit bevel: the shading explains it
    residual = residual.copy()
    residual[GLINT] = 0.8                              # the glint: nothing but the residual explains it
    residual = _clip_residual(albedo, shading, residual)
    return albedo, shading, residual, gm, groups


def island_scene(lighter: bool = False):
    """The white scene with a dark decal carved out of the white paint as an island (group LETTER). The
    decal's label also holds white paint next to the white label (rows 26..29 of the decal: the analysis
    took the paint into the decal), which the photo shows lit like the paint around it, or, with
    ``lighter``, much lighter than the paint (a reflection in glass, not paint)."""
    albedo, shading, residual, gm, _ = white_scene()
    gm = gm.copy()
    albedo = albedo.copy()
    residual = residual.copy()
    gm[16:30, 10:30] = LETTER
    albedo[16:26, 10:30] = 0.03                         # the decal itself: dark
    albedo[26:30, 10:30] = 0.72                         # white paint inside the decal's label
    residual[16:26, 10:30] = 0.0
    if lighter:
        residual[26:30, 10:30] = 0.6                    # a white reflection in glass: clipped
    residual = _clip_residual(albedo, shading, residual)
    islands = gm == LETTER
    groups = [_group(g, albedo, gm) for g in (WHITE, RED, BLACK, LETTER)]
    return (albedo, shading, residual, gm, groups), islands


def glossy_scene(panel_albedo: float = 0.34, streak: float = 0.25):
    """The white scene with the white panel's albedo set to ``panel_albedo`` (0.34: a mid grey, CIELAB L
    ~65) and no floor, but a broad achromatic streak in its residual on about a third of it: the
    reflections of a glossy grey (metal, a clear coat), which glints on ~27 % of its pixels."""
    albedo, shading, residual, gm, _ = white_scene(white_albedo=panel_albedo, floor_share=0.0)
    residual = residual.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    band = (np.abs((xx - 0.6 * yy) - 10) < 7) & (gm == WHITE)
    residual[band] = streak
    residual = _clip_residual(albedo, shading, residual)
    groups = [_group(g, albedo, gm) for g in (WHITE, RED, BLACK)]
    return (albedo, shading, residual, gm, groups), band


def source_scene(L: float, C: float = 2.0, hue: float = 70.0):
    """The glint scene with the white panel's albedo set to CIELAB (L, C, hue) and no sheen on the black panel,
    so the white's floor is the image's largest residual and reads as glints on most of the panel (no other
    highlight sets the 99th percentile): a sweep of L runs through every neutral-weight ramp (lightness, the
    white evidence that overrides the gloss fade, the exposure bound's share) and through the glint's
    standout, a sweep of C through the chroma ramp."""
    albedo, shading, _, gm, _ = white_scene()
    albedo = albedo.copy()
    lab = np.array([L, C * np.cos(np.radians(hue)), C * np.sin(np.radians(hue))], np.float32)
    albedo[gm == WHITE] = imageio.lab_to_linear(lab[None, None])[0, 0]
    residual = np.zeros_like(albedo)
    residual[gm == WHITE] = 0.15 * (albedo * shading)[gm == WHITE]
    residual[GLINT] = 0.8
    residual = _clip_residual(albedo, shading, residual)
    groups = [_group(g, albedo, gm) for g in (WHITE, RED, BLACK)]
    return albedo, shading, residual, gm, groups


def _ok(u8):
    return engine.linear_to_oklab_t(torch.from_numpy(imageio.srgb_to_linear(imageio.to_float(u8)).astype(np.float32))).numpy()


def _chroma(u8):
    ok = _ok(u8)
    return np.hypot(ok[..., 1], ok[..., 2])


def _renderer(scene, saturated_rules: bool = False, **kw):
    r = engine.Renderer(*scene, **kw)
    if saturated_rules:
        n = max(r.n_groups, 1)
        r._group_neutral = np.zeros(n, np.float32)    # every source is a saturated paint
        r._group_white = np.zeros(n, np.float32)
    return r


def _render(scene, mapping, saturated_rules: bool = False, options: RenderOptions | None = None, long_side=None, **kw):
    r = _renderer(scene, saturated_rules, **kw)
    try:
        opts = options or RenderOptions(feather_px=0.0)
        return r.render(mapping, opts) if long_side is None else r.render_at(long_side, mapping, opts)
    finally:
        r.free()


@pytest.fixture(scope="module")
def scene():
    return white_scene()


# ------------------------------------------------------------------ identity, weights, saturated sources

def test_identity_unchanged(scene):
    assert np.array_equal(_render(scene, {}), engine.recompose(*scene[:3]))
    sc, islands = island_scene()
    assert np.array_equal(_render(sc, {}, islands=islands), engine.recompose(*sc[:3]))
    g = glint_scene()
    assert np.array_equal(_render(g, {}), engine.recompose(*g[:3]))


def test_neutral_weights(scene):
    """White paint is a neutral source; a real colour and a black paint are not."""
    r = engine.Renderer(*scene)
    try:
        w = r._group_neutral_weights()
        assert w[WHITE] == 1.0 and w[RED] == 0.0 and w[BLACK] == 0.0
    finally:
        r.free()
    assert engine._neutral_weight(2.0, 90.0, 0.0) == 1.0
    assert engine._neutral_weight(25.0, 90.0, 1.0) == 0.0              # a real colour
    assert engine._neutral_weight(2.0, 30.0, 0.0) == 0.0               # a dark grey, no white in the photo
    assert engine._neutral_weight(2.0, 30.0, 0.5) == 1.0               # ... a white paint under a grey albedo
    assert 0.0 < engine._neutral_weight(14.0, 90.0, 0.0) < 1.0
    assert engine._neutral_weight(2.0, 65.0, 0.10, 0.05) == 1.0        # a matte light grey
    assert engine._neutral_weight(2.0, 65.0, 0.10, 0.30) == 0.0        # a glossy grey: metal, a gloss
    assert engine._neutral_weight(2.0, 88.0, 0.60, 0.40) == 1.0        # glossy white paint is white paint
    assert engine._neutral_weight(2.0, 48.0, 0.60, 0.40) == 1.0        # ... and so is a white photo
    assert 0.0 < engine._neutral_weight(2.0, 65.0, 0.10, 0.17) < 1.0


def test_saturated_source_is_unchanged(scene):
    """A saturated source (CIELAB chroma >= NEUTRAL_C1) renders exactly as under the saturated-paint rules,
    alone and next to a repainted white paint (away from their shared boundary), and so does a black
    paint, whose neutral floor is its sheen."""
    for sc, kw in ((scene, {}), (glint_scene(), {}), (island_scene()[0], {"islands": island_scene()[1]})):
        assert np.array_equal(_render(sc, {RED: NAVY}, **kw), _render(sc, {RED: NAVY}, saturated_rules=True, **kw))
        assert np.array_equal(_render(sc, {BLACK: "#c1121f"}, **kw),
                              _render(sc, {BLACK: "#c1121f"}, saturated_rules=True, **kw))
    both = _render(scene, {RED: "#1b4d3e", WHITE: NAVY})
    both_sat = _render(scene, {RED: "#1b4d3e", WHITE: NAVY}, saturated_rules=True)
    assert np.array_equal(both[:, 56:], both_sat[:, 56:])


def test_a_chrome_group_keeps_the_saturated_paint_rules(scene):
    """A group the analysis tagged chrome is never a neutral source: its floor is its reflections."""
    albedo, shading, residual, gm, groups = scene
    chrome = [dataclasses.replace(g, finish="chrome") if g.id == WHITE else g for g in groups]
    sc = (albedo, shading, residual, gm, chrome)
    r = engine.Renderer(*sc)
    try:
        assert r._group_neutral_weights()[WHITE] == 0.0
    finally:
        r.free()
    assert np.array_equal(_render(sc, {WHITE: NAVY}), _render(sc, {WHITE: NAVY}, saturated_rules=True))


def test_a_glossy_grey_keeps_its_reflections():
    """A mid-grey glossy panel (L 65, reflections on a third of it: it glints on more than NEUTRAL_GLOSS_G1 of
    its pixels) with no evidence of white paint is no neutral source: it renders exactly as under the
    saturated-paint rules, which keep its streak (the white-paint rules repainted the streak as diffuse
    light: the BMW's cast fork leg and brake disc read as matte plastic, a concrete wall as flat paint)."""
    sc, band = glossy_scene()
    r = engine.Renderer(*sc)
    try:
        assert r._group_neutral_weights()[WHITE] == 0.0
        assert r.neutral_sources({WHITE: NAVY}) == ()
    finally:
        r.free()
    new = _render(sc, {WHITE: NAVY})
    assert np.array_equal(new, _render(sc, {WHITE: NAVY}, saturated_rules=True))
    L = _ok(new)[..., 0]
    assert L[band].mean() > L[(sc[3] == WHITE) & ~band].mean() + 0.15          # the streak is still there


def test_glossy_white_paint_is_still_white_paint():
    """... while the same reflections on a white paint (albedo L 88) leave it a neutral source: evidence of
    white paint overrides the gloss (the RX-78's white armour glints on a third of its pixels)."""
    sc, _ = glossy_scene(panel_albedo=0.72)
    r = engine.Renderer(*sc)
    try:
        assert r._group_neutral_weights()[WHITE] == 1.0
    finally:
        r.free()


def test_a_map_to_the_swatch_colour_stays_the_photo():
    """A neutral source mapped to its own swatch colour (``albedo_hex``) repaints nothing: every neutral-source
    rule fades out as the target nears the source colour (NEUTRAL_NEAR_DE*), so the render is exactly the
    saturated-paint rules' and within dE 1 of the photo, also for a white paint with a grey albedo under an
    inflated light (for that map the exposure bound took 43 L* off the lit faces of the Unicorn's armour)."""
    for sc in (white_scene(), white_scene(white_albedo=0.33, floor_share=0.25, sun=2.9), glint_scene()):
        hexc = next(g for g in sc[4] if g.id == WHITE).albedo_hex
        new = _render(sc, {WHITE: hexc})
        assert np.array_equal(new, _render(sc, {WHITE: hexc}, saturated_rules=True))
        m = sc[3] == WHITE
        de = np.linalg.norm(imageio.rgb_to_lab(imageio.to_float(new))[m]
                            - imageio.rgb_to_lab(imageio.to_float(engine.recompose(*sc[:3])))[m], axis=-1)
        assert de.mean() < 1.0
        r = engine.Renderer(*sc)
        try:
            assert r.neutral_sources({WHITE: hexc}) == () and r.neutral_sources({WHITE: NAVY}) == (WHITE,)
        finally:
            r.free()
    src = imageio.hex_to_lab("#b8b8b8")
    fades = [engine._near_source_fade(imageio.hex_to_lab(h), src) for h in ("#b8b8b8", "#b0b0b0", "#a0a0a0", "#808080")]
    assert fades[0] == 0.0 and 0.0 < fades[1] < fades[2] < 1.0 and fades[3] == 1.0


# ------------------------------------------------------------------ rules 2, 6, 7a': the white's own light

def test_white_panel_lit_faces_read_navy(scene):
    """Rule 6: the white paint's neutral floor is its diffuse light and follows the new paint, so the lit
    faces of a navy repaint carry the target's chroma and hue (kept as a glint / veil they read
    cornflower) and are no lighter than navy lit by the photo's light."""
    new = _render(scene, {WHITE: NAVY})
    old = _render(scene, {WHITE: NAVY}, saturated_rules=True)
    t = engine.hex_to_oklab(NAVY)
    c_t = float(np.hypot(t[1], t[2]))
    ok = _ok(new[LIT])
    assert _chroma(new[LIT]).mean() > 0.85 * c_t
    assert _chroma(new[LIT]).mean() > _chroma(old[LIT]).mean() + 0.02
    hue = np.degrees(np.arctan2(ok[..., 2], ok[..., 1]))
    assert np.abs((hue - np.degrees(np.arctan2(t[2], t[1])) + 180.0) % 360.0 - 180.0).max() < 6.0
    # the lit navy is the target under the photo's own light, not lifted toward white
    shade = scene[1][LIT]
    lit_target = imageio.srgb_to_linear(imageio.hex_to_rgb01(NAVY)) * shade
    L_ref = engine.linear_to_oklab_t(torch.from_numpy(lit_target.astype(np.float32))).numpy()[..., 0]
    assert np.median(ok[..., 0] - L_ref) < 0.03
    assert ok[..., 0].mean() < _ok(old[LIT])[..., 0].mean() - 0.03


def test_black_target_is_black_not_marble(scene):
    """Black on white paint: the floor no longer survives as a grey veil (grey marble)."""
    new = _render(scene, {WHITE: "#000000"})
    old = _render(scene, {WHITE: "#000000"}, saturated_rules=True)
    assert _ok(new[LIT])[..., 0].mean() < _ok(old[LIT])[..., 0].mean() - 0.03
    assert _ok(new[LIT])[..., 0].mean() < 0.25


def test_no_texture_above_the_white_anchor(scene):
    """Rule 2: what lies above a white paint's own lightness is light the decomposition left in the
    albedo; flat above the anchor (NEUTRAL_UP_K 0), so lighter albedo stripes do not lift the navy."""
    albedo, shading, residual, gm, groups = scene
    albedo = albedo.copy()
    albedo[40:60:4, 8:40] = 0.85                       # lighter albedo stripes above the anchor
    sc = (albedo, shading, _clip_residual(albedo, shading, residual), gm, groups)
    new = _render(sc, {WHITE: NAVY})
    old = _render(sc, {WHITE: NAVY}, saturated_rules=True)
    stripe = (slice(44, 45), slice(8, 40))
    between = (slice(45, 46), slice(8, 40))
    lift_new = _ok(new[stripe])[..., 0].mean() - _ok(new[between])[..., 0].mean()
    lift_old = _ok(old[stripe])[..., 0].mean() - _ok(old[between])[..., 0].mean()
    assert lift_old > 0.02                              # the saturated-paint rule copies it 1:1
    assert lift_new < 0.5 * lift_old


def test_exposure_bound_keeps_a_sunlit_pastel_off_white():
    """Rule 7a': a white paint the decomposition gave a grey albedo under an inflated light (the photo
    white) is bounded by the light a white paint implies, so a pastel stays a pastel."""
    sc = white_scene(white_albedo=0.33, floor_share=0.25, sun=2.9)
    new = _render(sc, {WHITE: PASTEL})
    old = _render(sc, {WHITE: PASTEL}, saturated_rules=True)
    lit = (slice(50, 62), slice(8, 40))
    assert _chroma(new[lit]).mean() > _chroma(old[lit]).mean() + 0.03       # blown toward white without it
    assert _chroma(new[lit]).mean() > 0.06


def test_exposure_bound_needs_white_paint():
    """... but a grey paint whose photo clipped in a small corner is not white paint there: with a white
    share below EXPOSURE_S0 its white-paint weight is 0, and the bound does nothing (a renderer whose
    bound had no light at all renders the same)."""
    albedo, shading, residual, gm, groups = white_scene(white_albedo=0.33, floor_share=0.0, sun=0.9)
    shading = shading.copy()
    shading[2:6, 2:6] = 3.0                             # a tiny blown corner, 16 px of 3072
    sc = (albedo, shading, _clip_residual(albedo, shading, residual), gm, groups)
    r = engine.Renderer(*sc)
    dark = engine.Renderer(*sc)
    try:
        r._group_neutral_weights()
        assert r._group_white[WHITE] == 0.0
        zero = torch.zeros(gm.shape, device=dark.device)
        dark._base.white_light = (zero, torch.ones_like(zero))
        opts = RenderOptions(feather_px=0.0)
        assert np.array_equal(r.render({WHITE: "#202020"}, opts), dark.render({WHITE: "#202020"}, opts))
    finally:
        r.free()
        dark.free()


def test_warm_lit_off_white_has_no_gloss_floor():
    """Rule 7b: an off-white under a warm lamp passes the saturated-paint gloss test (photo min/max below
    GLOSS_RATIO_MAX, the light's colour), but a neutral group gets no gloss floor of its own."""
    albedo, shading, residual, gm, groups = white_scene(floor_share=0.0)
    shading = (shading * np.array([1.0, 0.55, 0.3], np.float32)).astype(np.float32)   # a warm lamp
    residual = residual.copy()
    residual[30:32, :48] = 0.08                       # the white's own light: whiter than the warm diffuse
    sc = (albedo, shading, _clip_residual(albedo, shading, residual), gm, groups)
    r = engine.Renderer(*sc)
    try:
        assert r._group_neutral_weights()[WHITE] == 1.0
        white = gm == WHITE
        assert float(r._gloss(r._base).cpu().numpy()[white].max()) > 0.0
        assert float(r._gloss_neutral(r._base).cpu().numpy()[white].max()) == 0.0
    finally:
        r.free()


def test_white_source_does_not_reflect(scene):
    """Rule 8: a white paint's faint cast is the light's; its repaint recolours no neighbour."""
    albedo, shading, residual, gm, groups = scene
    albedo = albedo.copy()
    albedo[gm == WHITE] = albedo[gm == WHITE] * np.array([0.97, 0.99, 1.03], np.float32)   # faint blue cast
    groups = [_group(g, albedo, gm) for g in (WHITE, RED, BLACK)]
    r = engine.Renderer(albedo, shading, residual, gm, groups)
    try:
        assert not r._group_params({WHITE: NAVY}, RenderOptions()).refl
    finally:
        r.free()


# ------------------------------------------------------------------ rule 7e: a white paint's own glints

def _near_white(u8, l_min: float = 0.9, c_max: float = 0.05) -> np.ndarray:
    ok = _ok(u8)
    return (ok[..., 0] >= l_min) & (np.hypot(ok[..., 1], ok[..., 2]) <= c_max)


def test_small_clipped_glint_stays_white():
    """A small clipped white spot that the shading does not explain is the lamp's glint: it stays white on
    a navy and on a black repaint (the saturated-paint rules and a plain repaint turn it navy), while the
    white panel around it takes the target."""
    g = glint_scene()
    for target in (NAVY, "#000000"):
        new = _render(g, {WHITE: target})
        assert _near_white(new[GLINT]).all(), target
        assert not _near_white(new[18:31, 16:29]).all()         # a glint, not a white patch
        assert not _near_white(new[LIT]).any()
    old = _render(g, {WHITE: NAVY}, saturated_rules=True)
    assert not _near_white(old[GLINT]).any()
    r = engine.Renderer(*g)
    try:
        field = r._white_glints(r._base).cpu().numpy()
        assert field[GLINT].min() > 0.8
        assert field[:, 48:].max() == 0.0                          # only on the neutral paint
        assert (field > 0.01).sum() < 150                          # the glint and its halo only
    finally:
        r.free()


def test_a_lit_bevel_is_repainted_not_kept_white():
    """A clipped band the shading layer explains (a lit bevel or crease) is paint, and so is a clipped
    face too large for a glint: both are repainted."""
    g = glint_scene()
    new = _render(g, {WHITE: NAVY})
    bevel = (slice(34, 37), slice(12, 32))
    assert not _near_white(new[bevel]).any()
    albedo, shading, residual, gm, groups = white_scene()
    residual = residual.copy()
    residual[20:36, 10:30] = 0.8                                  # a clipped face of 16 x 20 px
    face = (albedo, shading, _clip_residual(albedo, shading, residual), gm, groups)
    out = _render(face, {WHITE: NAVY})
    assert not _near_white(out[20:36, 10:30]).any()


def test_white_lettering_in_the_paints_group_is_no_glint():
    """A small clipped white spot whose surround is not lit white paint (white lettering on a black
    backdrop that the analysis put into the white paint's group) is repainted with the group, not kept."""
    albedo, shading, residual, gm, groups = white_scene()
    albedo = albedo.copy()
    albedo[20:34, 10:34] = 0.02                         # a black plate inside the white group's label
    albedo[25:29, 18:24] = 0.9                          # white lettering on it
    residual = residual.copy()
    residual[20:34, 10:34] = 0.0
    residual[25:29, 18:24] = 0.6                        # clipped
    sc = (albedo, shading, _clip_residual(albedo, shading, residual), gm, groups)
    r = engine.Renderer(*sc)
    try:
        assert r._white_glints(r._base).cpu().numpy()[20:34, 10:34].max() == 0.0
    finally:
        r.free()
    assert not _near_white(_render(sc, {WHITE: NAVY})[25:29, 18:24]).any()


def test_an_export_keeps_the_previews_glints():
    """A renderer given the working-resolution glint field (a full-resolution export) keeps exactly those
    glints, resized to its layers, and none where the field has none."""
    g = glint_scene()
    r = engine.Renderer(*g)
    try:
        field = r.white_glints()
    finally:
        r.free()
    assert field.shape == (H, W) and field[GLINT].min() > 0.8
    big = tuple(np.repeat(np.repeat(a, 2, axis=0), 2, axis=1) for a in g[:4]) + (g[4],)
    kept = _render(big, {WHITE: NAVY}, glints=field, reference_long_side=W)
    none = _render(big, {WHITE: NAVY}, glints=np.zeros_like(field), reference_long_side=W)
    core = (slice(46, 52), slice(42, 48))               # the glint's core, twice the size
    assert _near_white(kept[core]).all()
    assert not _near_white(none[core]).any()


def test_glints_survive_the_preview():
    """The glints are found at the layers' resolution and resized, so a preview keeps them."""
    g = glint_scene()
    small = _render(g, {WHITE: NAVY}, long_side=64)
    assert small.shape[:2] == (43, 64)
    assert _near_white(small[14:18, 13:17], l_min=0.85, c_max=0.06).any()


def test_a_glint_does_not_darken_the_paint_around_it():
    """Rule 7a': a small clipped spot on a face that is not white in the photo is no source of the exposure
    bound's light (the Unicorn's sun streak switched the bound on 4 px around it with a light of 1.0, and its
    navy sat in a dark pad at 0.57 of the face): the ring 1-6 px around the spot renders like the face."""
    albedo, shading, residual, gm, _ = white_scene(white_albedo=0.33, floor_share=0.25, sun=2.9)
    shading = shading.copy()
    shading[:30, :48] = 1.1                        # a grey face (photo Y ~0.45) above the sunlit white rows
    residual = residual.copy()
    residual[gm == WHITE] = (0.25 * albedo * shading)[gm == WHITE]
    residual[12:16, 22:26] = 0.8                   # a small clipped spot in the face
    sc = (albedo, shading, _clip_residual(albedo, shading, residual), gm, [_group(g, albedo, gm) for g in (WHITE, RED, BLACK)])
    r = engine.Renderer(*sc)
    try:
        r._group_neutral_weights()
        assert r._group_white[WHITE] == 1.0         # the bound is on for this paint
    finally:
        r.free()
    Y = imageio.luminance(imageio.srgb_to_linear(imageio.to_float(_render(sc, {WHITE: NAVY}))))
    core = np.zeros(gm.shape, bool)
    core[12:16, 22:26] = True
    d = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    face = (gm == WHITE) & (np.arange(H)[:, None] < 30)
    ring, far = face & (d >= 1) & (d <= 6), face & (d >= 8) & (d <= 20)
    assert np.median(Y[ring]) >= 0.97 * np.median(Y[far])


def test_a_lone_white_pixel_does_not_darken_the_paint_around_it():
    """Rule 7a' acts as far as a pixel's neighbourhood is white (EXPOSURE_NEAR): a few noise pixels that cross
    the white threshold on a grey face (the Unicorn's sit at OK L 0.79-0.80) switched the bound on at full
    strength over a 4 px disc each and dotted the navy with dark discs."""
    albedo, shading, residual, gm, _ = white_scene(white_albedo=0.33, floor_share=0.25, sun=2.9)
    shading = shading.copy()
    shading[:30, :48] = 1.1                        # a grey face above the sunlit white rows
    residual = residual.copy()
    residual[gm == WHITE] = (0.25 * albedo * shading)[gm == WHITE]
    lone = np.zeros(gm.shape, bool)
    for y, x in ((8, 12), (14, 30), (22, 18)):
        residual[y, x] = 0.3                       # white in the photo, unclipped
        lone[y, x] = True
    sc = (albedo, shading, _clip_residual(albedo, shading, residual), gm, [_group(g, albedo, gm) for g in (WHITE, RED, BLACK)])
    Y = imageio.luminance(imageio.srgb_to_linear(imageio.to_float(_render(sc, {WHITE: NAVY}))))
    d = cv2.distanceTransform((~lone).astype(np.uint8), cv2.DIST_L2, 5)
    face = (gm == WHITE) & (np.arange(H)[:, None] < 30)
    assert np.median(Y[face & (d >= 1) & (d <= 4)]) >= 0.97 * np.median(Y[face & (d >= 7)])


def test_a_clipped_corner_of_a_lit_face_is_no_glint():
    """A small clipped piece at the edge of a lit face whose ring falls mostly on darker paint (so it stands out
    from the ring's median) is a piece of that face when the face beside it is as bright as it (the RX-78's one
    kept "glint", 1.10x its lit face): rule 7e gives it a small weight."""
    albedo, shading, residual, gm, _ = white_scene(white_albedo=0.6, floor_share=0.0)
    shading = np.full_like(shading, 0.8)
    shading[26:32, :48] = 1.0                      # a thin lit face ...
    residual = np.zeros_like(residual)
    residual[26:32, :48] = 0.32                    # ... with the white's floor on it: photo 0.92, unclipped
    residual[29:32, 20:26] = 0.5                   # a clipped piece in its lower edge
    sc = (albedo, shading, _clip_residual(albedo, shading, residual), gm, [_group(g, albedo, gm) for g in (WHITE, RED, BLACK)])
    r = engine.Renderer(*sc)
    try:
        assert r._white_glints(r._base).cpu().numpy()[29:32, 20:26].max() < 0.25
    finally:
        r.free()


def test_a_glint_dims_gradually():
    """Rule 7e weighs each glint with smoothsteps: as the paint around a small clipped glint brightens, its
    standout falls through GLINT_STANDOUT0..1 and the kept glint dims over many steps (under hard cuts it
    went from white to navy in one step of the paint's albedo)."""
    levels = []
    for a in np.arange(0.55, 0.90, 0.005):
        albedo, shading, residual, gm, _ = glint_scene()
        albedo = albedo.copy()
        albedo[gm == WHITE] = a
        residual = residual.copy()
        residual[gm == WHITE] = 0.15 * (albedo * shading)[gm == WHITE]
        residual[GLINT] = 0.8
        sc = (albedo, shading, _clip_residual(albedo, shading, residual), gm, [_group(g, albedo, gm) for g in (WHITE, RED, BLACK)])
        levels.append(float(_render(sc, {WHITE: NAVY})[GLINT].mean()))
    levels = np.array(levels)
    assert levels.max() - levels.min() > 120                        # white at first, navy at the end
    assert np.abs(np.diff(levels)).max() < 0.15 * (levels.max() - levels.min())


def test_renders_change_gradually_with_the_source_paint():
    """Neighbouring source paints render alike: sweeps of the source's CIELAB lightness (a quarter of a unit a
    step, through the white-evidence and exposure-bound ramps and, from L 92 on, through the glint's standout
    ramp until the glint is painted) and chroma (through NEUTRAL_C0/C1) never move a pixel of the repainted
    panel by a fraction of what a switched rule did (a glint switching off moved 237 levels in one step of
    0.25 L; the soft ramps' steepest step is 17 levels, the frozen engine's own steps are up to 5)."""
    def steps(frames):
        return [int(np.abs(b.astype(int) - a.astype(int))[:, :48].max()) for a, b in zip(frames, frames[1:])]
    frames = [_render(source_scene(L), {WHITE: NAVY}) for L in np.arange(64.0, 99.01, 0.25)]
    assert _near_white(frames[0][GLINT]).all() and not _near_white(frames[-1][GLINT]).any()    # it crossed the ramp
    assert max(steps(frames)) <= 20
    assert max(steps([_render(source_scene(80.0, C), {WHITE: "#000000"}) for C in np.arange(8.0, 20.01, 0.5)])) <= 12


def test_renders_change_gradually_with_the_target_and_the_white_share():
    """The near-source fade and the exposure bound's white-share ramp come in gradually too: a sweep of the target
    away from the swatch colour (a quarter of a CIELAB unit a step, through NEUTRAL_NEAR_DE0..1, which brings in
    rules worth 119 levels on this sunlit grey-albedo white) and a sweep of the sun over the same paint (its photo's
    white share crossing the bound's ramp: its white-paint weight goes from 0 to 1) never move a pixel by more than a
    small step (the steepest, 9 and 17 levels)."""
    def steps(frames):
        return [int(np.abs(b.astype(int) - a.astype(int))[:, :48].max()) for a, b in zip(frames, frames[1:])]
    sc = white_scene(white_albedo=0.33, floor_share=0.25, sun=2.9)
    src = np.array(next(g for g in sc[4] if g.id == WHITE).albedo_lab, np.float64)
    far = np.array(imageio.hex_to_lab("#5a6f9a"), np.float64)
    toward = (far - src) / np.linalg.norm(far - src)
    frames = [_render(sc, {WHITE: imageio.lab_to_hex(tuple(src + d * toward))}) for d in np.arange(0.0, 16.01, 0.25)]
    assert int(np.abs(frames[-1].astype(int) - frames[0].astype(int))[:, :48].max()) > 60
    assert max(steps(frames)) <= 15
    whites = []
    for sun in (1.0, 3.0):
        r = engine.Renderer(*white_scene(white_albedo=0.33, floor_share=0.25, sun=sun))
        try:
            r._group_neutral_weights()
            whites.append(float(r._group_white[WHITE]))
        finally:
            r.free()
    assert whites == [0.0, 1.0]                                             # the sweep crosses the bound's ramp
    frames = [_render(white_scene(white_albedo=0.33, floor_share=0.25, sun=float(s)), {WHITE: PASTEL})
              for s in np.arange(1.0, 3.001, 0.02)]
    assert max(steps(frames)) <= 24


def test_the_gloss_gate_comes_in_gradually():
    """A grey's glint share is counted softly (GLINT_PX_*), so a streak whose residual sits at one value does not
    move the whole share at once: as a uniform streak on a mid-grey brightens from a sheen to a glint, the grey's
    neutral weight falls from 1 to 0 over many steps (counted with hard cuts it fell from 1 to 0, and the render
    moved 81 levels, for one 0.005 step of the streak)."""
    for albedo in (0.34, 0.45):
        weights, frames = [], []
        for s in np.arange(0.10, 0.3001, 0.005):
            sc, _ = glossy_scene(panel_albedo=albedo, streak=float(s))
            r = engine.Renderer(*sc)
            try:
                weights.append(float(r._group_neutral_weights()[WHITE]))
                frames.append(r.render({WHITE: "#c1121f"}, RenderOptions(feather_px=0.0)))
            finally:
                r.free()
        assert weights[0] == 1.0 and weights[-1] < 0.15                    # a matte grey at first, glossy at the end
        assert np.abs(np.diff(weights)).max() < 0.25
        assert max(int(np.abs(b.astype(int) - a.astype(int))[:, :48].max()) for a, b in zip(frames, frames[1:])) <= 25


def test_a_glint_just_past_the_old_cuts_is_kept_whole():
    """Rule 7e's ramps lie below the hard tests that first kept the real glints: a small clipped spot 1.57x the
    paint around it, which the shading does not explain, is kept whole and stays white on navy and black, its
    whole clipped core at the photo's brightness, the pixels clipped in one channel only included (with ramps
    centred on those cuts the Alpine's roof streak, 1.54x its ring, was kept at half weight and read as a pale
    blue stripe; scaled by its own profile a dimmer clipped pixel came out pale blue)."""
    albedo, shading, residual, gm, groups = glint_scene()
    lin = albedo * shading
    core = np.zeros(gm.shape, bool)
    core[GLINT] = True
    ring = (cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5) > 2) & (
        cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5) <= 6) & (gm == WHITE)
    ring_y = float(np.median(imageio.luminance(lin + residual)[ring]))
    residual = residual.copy()
    tgt = (1.57 * ring_y - 0.2126) / 0.7874               # a core 1.57x its ring, clipped in every channel ...
    residual[GLINT] = (np.array([1.0, tgt, tgt], np.float32) - lin[GLINT]).astype(np.float32)
    dim = (slice(23, 27, 2), GLINT[1])                   # ... with two rows clipped in red only, dimmer
    residual[dim] = (np.array([1.0, 0.9, 0.9], np.float32) - lin[dim]).astype(np.float32)
    sc = (albedo, shading, residual, gm, groups)
    r = engine.Renderer(*sc)
    try:
        field = r._white_glints(r._base).cpu().numpy()
        top = float(np.median(imageio.luminance(np.clip(lin + residual, 0, 1))[GLINT]))
        assert field[GLINT].min() >= 0.97 * top
    finally:
        r.free()
    for target in (NAVY, "#000000"):
        assert _near_white(_render(sc, {WHITE: target})[GLINT], l_min=0.85, c_max=0.06).all(), target


def test_a_glint_with_a_bright_halo_on_evenly_lit_paint_is_kept():
    """Rule 7e's lit test (is the spot brighter than the brightest unclipped paint around it?) applies only as far
    as the shading steps up under the spot (GLINT_LIT_SHADE*): on evenly lit paint the bright pixels around a glint
    are its own halo, and the test painted the model ship's brightest lamp dot (1.04x its halo, which runs on to
    one side of it)."""
    albedo, shading, residual, gm, groups = glint_scene()
    residual = residual.copy()
    lin = albedo * shading
    halo = (slice(21, 28), slice(14, 20))                # an unclipped tail to the left, nearly as bright as the core
    residual[halo] = (0.95 - lin[halo]).astype(np.float32)
    sc = (albedo, shading, _clip_residual(albedo, shading, residual), gm, groups)
    r = engine.Renderer(*sc)
    try:
        assert r._white_glints(r._base).cpu().numpy()[GLINT].min() > 0.9
    finally:
        r.free()


def test_a_near_neutral_sources_reflections_come_in_with_its_hue():
    """Rule 8 for a source that is no white paint (a dark grey, L 40) with a faint hue: its reflections come in with
    its hue confidence (CIELAB chroma 2 to 8) instead of switching on at full weight at chroma 2 (a 38-level step on
    the reflection), and from chroma 8 on they are exactly the old rule's (weight 1)."""
    sh, sw = 64, 96
    patch = (slice(20, 40), slice(52, 60))

    def scene(C, hue=40.0):
        gm = np.zeros((sh, sw), np.int32)
        gm[:, 48:] = 1
        h = math.radians(hue)
        albedo = np.zeros((sh, sw, 3), np.float32)
        albedo[gm == 0] = imageio.lab_to_linear(np.array([[[40.0, C * math.cos(h), C * math.sin(h)]]], np.float32))[0, 0]
        albedo[gm == 1] = imageio.lab_to_linear(np.array([[[60.0, 0.0, 0.0]]], np.float32))[0, 0]
        albedo[patch] = imageio.lab_to_linear(np.array([[[36.0, 22 * math.cos(h), 22 * math.sin(h)]]], np.float32))[0, 0]
        groups = []
        for g in (0, 1):
            lab = tuple(float(v) for v in np.median(imageio.linear_to_lab(albedo[gm == g]), axis=0))
            groups.append(ColorGroup(id=g, name=f"g{g}", albedo_lab=lab, albedo_hex=imageio.lab_to_hex(lab),
                                     area=int((gm == g).sum()), area_frac=float((gm == g).mean()), region_ids=[g],
                                     hue_family="neutral"))
        return albedo, np.ones_like(albedo), np.zeros_like(albedo), gm, groups

    change, weights = [], []
    for C in np.arange(0.0, 12.01, 0.5):
        r = engine.Renderer(*scene(float(C)))
        try:
            out = r.render({0: NAVY}, RenderOptions(feather_px=0.0)).astype(int)
            ident = r.render({}, RenderOptions()).astype(int)
            refl = r._group_params({0: NAVY}, RenderOptions()).refl
            assert r._group_neutral_weights()[0] == 0.0                      # a dark grey: no neutral source
        finally:
            r.free()
        change.append(float(np.abs(out[patch] - ident[patch]).mean()))
        weights.append(refl[0].weight if refl else 0.0)
    change, weights = np.array(change), np.array(weights)
    full = change[-1]
    assert full > 20.0                                                      # a reflection, recoloured
    assert change[:4].max() == 0.0                                          # no hue below chroma 2
    assert np.abs(np.diff(change)).max() < 0.25 * full                      # it comes in gradually
    assert (weights[16:] == 1.0).all() and np.abs(change[16:] - full).max() <= 1.0   # from chroma 8: the old rule


def test_an_export_takes_the_previews_neutral_weights():
    """``neutral_weights()`` is the table a full-resolution export is handed: a renderer given it reports and
    uses it instead of measuring its own (a table of zeros renders as the saturated-paint rules)."""
    g = glint_scene()
    r = engine.Renderer(*g)
    try:
        table = r.neutral_weights()
    finally:
        r.free()
    assert table.shape == (3, 2) and table[WHITE, 0] == 1.0 and table[RED, 0] == 0.0
    big = tuple(np.repeat(np.repeat(a, 2, axis=0), 2, axis=1) for a in g[:4]) + (g[4],)
    r = engine.Renderer(*big, neutral=table, reference_long_side=W)
    try:
        assert np.array_equal(r.neutral_weights(), table)
    finally:
        r.free()
    none = _render(big, {WHITE: NAVY}, neutral=np.zeros_like(table), reference_long_side=W)
    assert np.array_equal(none, _render(big, {WHITE: NAVY}, saturated_rules=True, reference_long_side=W))


# ------------------------------------------------------------------ rule 1: decal islands on white paint

def test_white_paint_inside_a_decal_is_repainted():
    """The white paint the analysis took into a decal next to the white label is repainted with it; the
    decal itself keeps its dark colour. Under the saturated-paint rules it stayed pale."""
    sc, islands = island_scene()
    new = _render(sc, {WHITE: NAVY}, islands=islands)
    old = _render(sc, {WHITE: NAVY}, islands=islands, saturated_rules=True)
    paint_in_decal = (slice(27, 30), slice(12, 28))
    decal = (slice(17, 24), slice(12, 28))
    assert _chroma(new[paint_in_decal]).mean() > 0.08
    assert _chroma(old[paint_in_decal]).mean() < 0.03
    photo = engine.recompose(*sc[:3])
    assert np.abs(new[decal].astype(int) - photo[decal].astype(int)).max() <= 2


def test_a_lighter_pixel_inside_a_decal_is_not_paint():
    """... but a pixel of the decal much lighter in the photo than the paint around it (a reflection in
    glass next to the paint) is not entered."""
    sc, islands = island_scene(lighter=True)
    new = _render(sc, {WHITE: NAVY}, islands=islands)
    photo = engine.recompose(*sc[:3])
    inner = (slice(26, 28), slice(12, 28))
    assert np.abs(new[inner].astype(int) - photo[inner].astype(int)).max() <= 2


# ------------------------------------------------------------------ robustness

def test_every_option_combination_is_finite(scene):
    g = glint_scene()
    sc, islands = island_scene()
    for s, kw in ((g, {}), (sc, {"islands": islands})):
        for mode in ("shift", "flat"):
            for tex in (0.0, 1.0):
                for tint in (0.0, 1.0):
                    for strength in (0.6, 1.4):
                        r = engine.Renderer(*s, **kw)
                        try:
                            opts = RenderOptions(mode=mode, texture=tex, residual_tint=tint, shading_strength=strength)
                            out = r.render({WHITE: PASTEL, RED: NAVY}, opts)
                            small = r.render_at(48, {WHITE: "#000000"}, RenderOptions(mode=mode))
                        finally:
                            r.free()
                        assert out.shape == (H, W, 3) and out.dtype == np.uint8
                        assert small.shape[2] == 3
