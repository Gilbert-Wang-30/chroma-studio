"""Unit tests for the recoloring engine (recolor/engine.py). No models, no network.

A small synthetic scene stands in for an analyzed job: three groups (a red panel, a
grey panel and a green stripe), smooth colored shading, a white specular blob and a
clipped (negative residual) patch. Later sections build one small scene per engine rule
(module docstring of recolor/engine.py), each asserting what the rule guarantees.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest
import torch

from recolor import engine, filters, imageio
from recolor.types import ColorGroup, RenderOptions

H, W = 72, 96
RED, GREY, GREEN = 0, 1, 2


def _group(gid: int, albedo_lab, area: int, locked: bool = False) -> ColorGroup:
    lab = tuple(float(v) for v in albedo_lab)
    return ColorGroup(id=gid, name=f"g{gid}", albedo_lab=lab, albedo_hex=imageio.lab_to_hex(lab),
                      area=area, area_frac=area / float(H * W), region_ids=[gid], hue_family="red",
                      locked=locked)


def make_scene(seed: int = 0):
    """-> (albedo, shading, residual, group_map, groups). albedo*shading+residual is a
    valid image in [0,1] except in the deliberately clipped patch."""
    rng = np.random.default_rng(seed)
    group_map = np.zeros((H, W), np.int32)
    group_map[:, W // 2:] = GREY
    group_map[H // 2 - 8:H // 2 + 8, :] = GREEN

    base = np.zeros((H, W, 3), np.float32)
    base[group_map == RED] = imageio.srgb_to_linear(imageio.hex_to_rgb01("#c0392b"))
    base[group_map == GREY] = imageio.srgb_to_linear(imageio.hex_to_rgb01("#9a9a9a"))
    base[group_map == GREEN] = imageio.srgb_to_linear(imageio.hex_to_rgb01("#2e8b3d"))
    albedo = np.clip(base * rng.uniform(0.85, 1.15, (H, W, 1)).astype(np.float32), 0.0, 1.0)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    shade = 0.35 + 0.75 * (0.5 * xx / W + 0.5 * yy / H)
    shading = np.stack([shade * 1.05, shade, shade * 0.95], axis=2).astype(np.float32)   # warm light

    residual = np.zeros((H, W, 3), np.float32)
    residual[10:18, 10:18] = 0.25                        # white specular on the red panel
    shading[60:68, 80:92] = 2.5                          # blown-out corner of the grey panel
    lin = albedo * shading + residual
    residual = np.where(lin > 1.0, 1.0 - albedo * shading, residual).astype(np.float32)   # clip -> negative residual

    def med(gid):
        return np.median(imageio.linear_to_lab(albedo[group_map == gid]), axis=0)

    groups = [_group(RED, med(RED), int((group_map == RED).sum())),
              _group(GREY, med(GREY), int((group_map == GREY).sum())),
              _group(GREEN, med(GREEN), int((group_map == GREEN).sum()))]
    return albedo, shading, residual, group_map, groups


def _interior(group_map: np.ndarray, gid: int, margin: int = 6) -> np.ndarray:
    """Boolean mask of a group's pixels at least `margin` px away from other groups
    (6 > the 5 px kernel radius of a sigma-1.5 feather)."""
    m = (group_map == gid).astype(np.float32)
    eroded = filters.box_filter(torch.from_numpy(m)[None].to(filters._dev()), margin)[0].cpu().numpy()
    return eroded > 0.999


@pytest.fixture(scope="module")
def scene():
    return make_scene()


@pytest.fixture(scope="module")
def renderer(scene):
    r = engine.Renderer(*scene)
    yield r
    r.free()


# ------------------------------------------------------------------ color math

def test_oklab_matches_the_reference_and_round_trips():
    # Ottosson's published values for linear sRGB white, red, green and blue
    lin = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    want = torch.tensor([[1.0, 0.0, 0.0], [0.6279554, 0.2248631, 0.1258463],
                         [0.8664396, -0.2338876, 0.1794985], [0.4520137, -0.0324570, -0.3115281]])
    assert torch.allclose(engine.linear_to_oklab_t(lin), want, atol=2e-4)
    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.random((500, 3)).astype(np.float32))
    ok = engine.linear_to_oklab_t(x)
    assert torch.allclose(engine.oklab_to_linear_t(ok), x, atol=1e-4)
    assert np.abs(engine.oklab_from_linear_np(x.numpy()) - ok.numpy()).max() < 1e-4       # numpy twin
    # group colours arrive as CIELAB of the linear albedo: the conversion inverts that stage
    lab = imageio.linear_to_lab(x.numpy()[:5])
    back = np.stack([engine.cielab_to_oklab(v) for v in lab])
    assert np.abs(back - ok.numpy()[:5]).max() < 2e-3


def test_gamut_compression_keeps_ok_lightness_and_hue():
    lab = torch.tensor([[0.95, 0.20, 0.20], [0.30, -0.25, -0.25], [0.5, 0.0, 0.0], [0.6, 0.05, -0.08]])
    lin = engine.oklab_to_linear_gamut_t(lab)
    assert torch.isfinite(lin).all() and float(lin.min()) >= 0.0 and float(lin.max()) <= 1.0
    back = engine.linear_to_oklab_t(lin)
    assert torch.allclose(back[:, 0], lab[:, 0], atol=5e-3)                  # OK L preserved
    hue_in = torch.atan2(lab[:2, 2], lab[:2, 1])
    hue_out = torch.atan2(back[:2, 2], back[:2, 1])
    assert torch.allclose(hue_in, hue_out, atol=0.03)                         # OK hue preserved
    assert torch.hypot(back[0, 1], back[0, 2]) < torch.hypot(lab[0, 1], lab[0, 2])   # chroma gave way
    assert torch.allclose(back[2:], lab[2:], atol=1e-4)                       # in-gamut untouched


# ------------------------------------------------------------------ identity

def test_identity_reproduces_recomposition(scene, renderer):
    albedo, shading, residual, _, _ = scene
    ref = engine.recompose(albedo, shading, residual)
    for opts in (RenderOptions(), RenderOptions(feather_px=0.0), RenderOptions(feather_px=4.0),
                 RenderOptions(mode="flat"), RenderOptions(texture=0.3, saturation=2.0)):
        out = renderer.render({}, opts)
        assert out.shape == (H, W, 3) and out.dtype == np.uint8
        assert np.abs(out.astype(np.int16) - ref.astype(np.int16)).max() <= 2
    # mapping everything to None / "" is the identity too
    out = renderer.render({RED: None, "1": "", 2: None}, RenderOptions())
    assert np.abs(out.astype(np.int16) - ref.astype(np.int16)).max() <= 2


def test_identity_without_residual(scene, renderer):
    albedo, shading, _, _, _ = scene
    ref = imageio.to_uint8(imageio.linear_to_srgb(np.clip(albedo * shading, 0, 1)))
    out = renderer.render({}, RenderOptions(keep_residual=False))
    assert np.abs(out.astype(np.int16) - ref.astype(np.int16)).max() <= 2


# ------------------------------------------------------------------ repaint

@pytest.mark.parametrize("feather", [0.0, 1.5])
def test_flat_repaint_puts_group_mean_at_target(scene, renderer, feather):
    _, _, _, group_map, _ = scene
    target = "#1f5fd6"
    alb = renderer.recolor_albedo({RED: target}, RenderOptions(mode="flat", feather_px=feather))
    interior = _interior(group_map, RED)
    mean_lab = imageio.linear_to_lab(alb[interior].mean(axis=0)[None])[0]
    assert float(imageio.delta_e(mean_lab, np.array(imageio.hex_to_lab(target)))) < 3.0
    # other groups untouched away from the border
    for gid in (GREY, GREEN):
        sel = _interior(group_map, gid)
        assert np.abs(alb[sel] - scene[0][sel]).max() < 1e-4


def test_shift_repaint_keeps_texture_and_hits_target(scene, renderer):
    albedo, _, _, group_map, groups = scene
    target = "#1f5fd6"
    alb = renderer.recolor_albedo({RED: target}, RenderOptions(mode="shift", texture=1.0, feather_px=0.0))
    sel = group_map == RED
    med = np.median(imageio.linear_to_lab(alb[sel]), axis=0)
    assert float(imageio.delta_e(med, np.array(imageio.hex_to_lab(target)))) < 3.0
    # lightness texture is preserved: L spread of the repaint equals the source's
    src_L = imageio.linear_to_lab(albedo[sel])[:, 0]
    new_L = imageio.linear_to_lab(alb[sel])[:, 0]
    assert abs(np.std(new_L) - np.std(src_L)) < 0.15 * np.std(src_L) + 0.2
    # texture 0 in shift mode is the flat repaint
    flat = renderer.recolor_albedo({RED: target}, RenderOptions(mode="shift", texture=0.0, feather_px=0.0))
    assert np.std(imageio.linear_to_lab(flat[sel])[:, 0]) < 0.05


def test_saturation_scales_target_chroma(scene, renderer):
    _, _, _, group_map, _ = scene
    sel = group_map == RED
    lo = renderer.recolor_albedo({RED: "#d62828"}, RenderOptions(mode="flat", saturation=0.2, feather_px=0.0))
    hi = renderer.recolor_albedo({RED: "#d62828"}, RenderOptions(mode="flat", saturation=1.0, feather_px=0.0))
    c_lo = np.hypot(*imageio.linear_to_lab(lo[sel]).mean(0)[1:])
    c_hi = np.hypot(*imageio.linear_to_lab(hi[sel]).mean(0)[1:])
    assert c_lo < 0.35 * c_hi


def test_locked_and_out_of_range_groups_are_skipped(scene):
    albedo, shading, residual, group_map, groups = scene
    locked = [g if g.id != RED else _group(RED, g.albedo_lab, g.area, locked=True) for g in groups]
    r = engine.Renderer(albedo, shading, residual, group_map, locked)
    try:
        ref = engine.recompose(albedo, shading, residual)
        out = r.render({RED: "#00ff00", 42: "#00ff00", -1: "#00ff00"}, RenderOptions())
        assert np.abs(out.astype(np.int16) - ref.astype(np.int16)).max() <= 2
    finally:
        r.free()


def test_group_missing_from_list_uses_pixel_median(scene):
    albedo, shading, residual, group_map, groups = scene
    r = engine.Renderer(albedo, shading, residual, group_map, groups[:2])      # GREEN unknown
    try:
        assert r.n_groups == 3
        alb = r.recolor_albedo({GREEN: "#ffcc00"}, RenderOptions(mode="shift", feather_px=0.0))
        med = np.median(imageio.linear_to_lab(alb[group_map == GREEN]), axis=0)
        assert float(imageio.delta_e(med, np.array(imageio.hex_to_lab("#ffcc00")))) < 3.0
    finally:
        r.free()


def test_bright_repaint_of_dark_group_stays_in_gamut(scene, renderer):
    _, _, _, group_map, _ = scene
    alb = renderer.recolor_albedo({GREY: "#fff44f"}, RenderOptions(mode="shift", feather_px=0.0))
    assert np.isfinite(alb).all() and alb.min() >= 0.0 and alb.max() <= 1.0
    out = renderer.render({GREY: "#fff44f"}, RenderOptions())
    assert out.dtype == np.uint8


# ------------------------------------------------------------------ residual and shading

def test_negative_residual_scales_with_repaint(scene, renderer):
    """Darkening the blown-out grey corner must not punch a dark hole into it."""
    out = renderer.render({GREY: "#202020"}, RenderOptions(feather_px=0.0))
    corner = out[61:67, 82:90].astype(np.float32).mean(axis=(0, 1))
    around = out[40:50, 70:90].astype(np.float32).mean(axis=(0, 1))
    assert corner.mean() >= around.mean() - 1.0


def test_residual_tint_moves_highlight_toward_target(scene, renderer):
    plain = renderer.render({RED: "#1f5fd6"}, RenderOptions(residual_tint=0.0, feather_px=0.0))
    tinted = renderer.render({RED: "#1f5fd6"}, RenderOptions(residual_tint=1.0, feather_px=0.0))
    spec = (slice(11, 17), slice(11, 17))
    lab_p = imageio.rgb_to_lab(imageio.to_float(plain[spec]).reshape(-1, 3)).mean(0)
    lab_t = imageio.rgb_to_lab(imageio.to_float(tinted[spec]).reshape(-1, 3)).mean(0)
    assert lab_t[2] < lab_p[2] - 3.0                      # bluer highlight
    assert abs(lab_t[0] - lab_p[0]) < 6.0                 # about as bright
    # tint does nothing outside repainted groups
    assert np.array_equal(plain[:, W // 2 + 8:], tinted[:, W // 2 + 8:])


def test_shading_strength_changes_contrast_not_exposure(scene, renderer):
    outs = {s: renderer.render({}, RenderOptions(shading_strength=s, keep_residual=False)).astype(np.float32)
            for s in (0.5, 1.0, 1.6)}
    means = {s: o.mean() for s, o in outs.items()}
    assert abs(means[0.5] - means[1.0]) < 0.08 * means[1.0]
    assert abs(means[1.6] - means[1.0]) < 0.08 * means[1.0]
    sel = np.s_[:H // 2 - 8, W // 2 + 8:, :]              # grey panel above the stripe, no clipped corner
    spread = {s: o[sel].mean(axis=2).std() for s, o in outs.items()}    # luminance gradient across it
    assert spread[0.5] < spread[1.0] < spread[1.6]


# ------------------------------------------------------------------ options / API surface

def test_all_options_accepted_and_finite(scene, renderer):
    grid = itertools.product(("shift", "flat"), (0.0, 0.5, 1.0), (0.0, 2.5), (True, False),
                             (0.0, 1.0), (0.6, 1.4), (0.0, 1.8), (False, True))
    for mode, tex, feather, keep, tint, strength, sat, sharpen in grid:
        opts = RenderOptions(mode=mode, texture=tex, feather_px=feather, keep_residual=keep,
                             residual_tint=tint, shading_strength=strength, saturation=sat,
                             sharpen_edges=sharpen)
        out = renderer.render({RED: "#1f5fd6", GREEN: "#f2b705"}, opts)
        assert out.shape == (H, W, 3) and out.dtype == np.uint8
        assert np.isfinite(out.astype(np.float32)).all()


def test_options_from_json_and_string_keys(scene, renderer):
    opts = RenderOptions.from_dict({"mode": "shift", "texture": 0.4, "feather_px": 2, "bogus": 1})
    out = renderer.render({"0": "#1f5fd6", "2": "f2b705"}, opts)
    assert out.shape == (H, W, 3)


def test_bad_mode_and_bad_color_raise(scene, renderer):
    with pytest.raises(ValueError):
        renderer.render({RED: "#1f5fd6"}, RenderOptions(mode="sideways"))
    with pytest.raises(ValueError):
        renderer.render({RED: "not-a-color"}, RenderOptions())


def test_constructor_validates_shapes(scene):
    albedo, shading, residual, group_map, groups = scene
    with pytest.raises(ValueError):
        engine.Renderer(albedo, shading[:-1], residual, group_map, groups)
    bad = group_map.copy()
    bad[0, 0] = -1
    with pytest.raises(ValueError):
        engine.Renderer(albedo, shading, residual, bad, groups)


def test_render_at_resizes_and_caches(scene, renderer):
    out = renderer.render_at(48, {RED: "#1f5fd6"}, RenderOptions())
    assert out.shape == (36, 48, 3)
    assert (48, 36) in renderer._levels
    again = renderer.render_at(48, {RED: "#1f5fd6"}, RenderOptions())
    assert np.array_equal(out, again)
    big = renderer.render_at(4000, {}, RenderOptions())                 # never upscales
    assert big.shape == (H, W, 3)
    assert renderer.last_render_ms >= 0.0


def test_render_once_matches_renderer(scene, renderer):
    out_once = engine.render_once(*scene, {RED: "#1f5fd6"}, RenderOptions())
    out = renderer.render({RED: "#1f5fd6"}, RenderOptions())
    assert np.array_equal(out_once, out)


def test_free_releases_and_cpu_device_works(scene):
    albedo, shading, residual, group_map, groups = scene
    r = engine.Renderer(albedo, shading, residual, group_map, groups, device="cpu")
    out = r.render({RED: "#1f5fd6"}, RenderOptions(feather_px=2.0, residual_tint=0.5))
    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    r.free()
    assert r._base.albedo.numel() == 0


def test_feathered_lookup_equals_soft_group_weights(scene):
    """The engine's blur-of-lookup equals sum_g W_g * const_g with filters.soft_group_weights."""
    _, _, _, group_map, _ = scene
    consts = torch.tensor([[1.0, 5.0], [0.0, -2.0], [1.0, 0.5]], device=filters._dev())
    w = filters.soft_group_weights(group_map, 3, 2.0)                       # [G,H,W]
    expected = torch.einsum("ghw,gc->hwc", w, consts)
    lookup = consts[torch.from_numpy(group_map.astype(np.int64)).to(consts.device)]     # [H,W,C]
    got = filters.gaussian_blur(lookup.permute(2, 0, 1).contiguous(), 2.0).permute(1, 2, 0)
    assert torch.allclose(got, expected, atol=1e-4)
    # the engine's own feather agrees with filters.gaussian_blur on an unambiguous shape
    mine = engine._feather_chw(lookup.permute(2, 0, 1).contiguous(), 2.0).permute(1, 2, 0)
    assert torch.allclose(mine, expected, atol=1e-4)


# ------------------------------------------------------------------ reviewer regressions

@pytest.mark.parametrize("shape", [(9, 2, 3), (9, 3, 1), (9, 1, 4), (9, 4, 4), (9, 1, 1)])
def test_feather_keeps_shape_for_ambiguous_widths(shape):
    """filters.gaussian_blur guesses channels from the last dim; the engine's feather
    must not, so a field whose W is 1, 3 or 4 keeps its [C,H,W] shape."""
    x = torch.rand(*shape, device=filters._dev())
    y = engine._feather_chw(x, 1.5)
    assert tuple(y.shape) == shape
    assert torch.isfinite(y).all()
    assert engine._feather_chw(x, 0.0) is x


@pytest.mark.parametrize("hw", [(1, 3), (3, 1), (4, 4), (1, 1), (2, 3)])
def test_tiny_images_render_with_feather(hw):
    """A 1-px-wide upload / render_at(long_side<=4) with feathering renders finite
    output instead of raising or corrupting the CUDA context."""
    h, w = hw
    rng = np.random.default_rng(3)
    albedo = rng.random((h, w, 3)).astype(np.float32)
    shading = np.full((h, w, 3), 0.8, np.float32)
    residual = np.zeros((h, w, 3), np.float32)
    gm = (np.arange(h * w).reshape(h, w) % 2).astype(np.int32)
    r = engine.Renderer(albedo, shading, residual, gm, [])
    try:
        out = r.render({0: "#1f5fd6", 1: "#f2b705"}, RenderOptions(feather_px=2.5, residual_tint=0.5))
        assert out.shape == (h, w, 3) and out.dtype == np.uint8
        assert np.isfinite(out.astype(np.float32)).all()
    finally:
        r.free()


def test_render_at_tiny_long_side_with_feather(scene, renderer):
    for side in (1, 2, 3, 4):
        out = renderer.render_at(side, {RED: "#1f5fd6", GREEN: "#f2b705"}, RenderOptions(feather_px=2.0))
        assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8
        assert max(out.shape[:2]) == side
        assert np.isfinite(out.astype(np.float32)).all()
    # the GPU context is still healthy afterwards
    assert renderer.render({RED: "#1f5fd6"}, RenderOptions()).shape == (H, W, 3)


@pytest.mark.parametrize("key", [None, "abc", "", 1.5, object()])
def test_normalize_mapping_bad_keys_raise_value_error(key):
    with pytest.raises(ValueError, match="not a group id"):
        engine.normalize_mapping({key: "#ff0000"})


def test_normalize_mapping_accepts_int_like_keys_and_drops_empties():
    got = engine.normalize_mapping({"0": "#ff0000", 1: "0f0", 2.0: "#0000FF", 3: None, "4": "  "})
    assert got == {0: "#ff0000", 1: "#00ff00", 2: "#0000ff"}
    with pytest.raises(ValueError):
        engine.normalize_mapping({0: "not-a-color"})
    with pytest.raises(ValueError):
        engine.normalize_mapping({0: 12345})


def test_render_after_free_raises(scene):
    albedo, shading, residual, group_map, groups = scene
    r = engine.Renderer(albedo, shading, residual, group_map, groups)
    assert not r.freed
    r.free()
    assert r.freed
    r.free()                                                   # idempotent
    for call in (lambda: r.render({}, RenderOptions()),
                 lambda: r.render_at(48, {}, RenderOptions()),
                 lambda: r.recolor_albedo({RED: "#1f5fd6"})):
        with pytest.raises(RuntimeError, match="freed"):
            call()


# ------------------------------------------------------------------ repaint across hue
#
# The albedo's deviation from its group's colour is the paint's texture: a little more or
# less saturated here, a hint of hue drift there. It is expressed in the source's a/b
# frame, so carried over unrotated it lands beside the new colour instead of along it: the
# lit flank of a red tank painted navy came out mauve, and the white decal on it cyan.

DECAL = (slice(26, 38), slice(14, 34))


def _textured_scene(paint_hex: str = "#c8140a"):
    """A saturated panel whose albedo varies in saturation only (dull on the left, full on
    the right, same hue throughout) with a white decal in the middle, next to a neutral
    panel. Flat neutral light, no residual, so the output hue is the repainted albedo's."""
    h, w = 64, 96
    gm = np.zeros((h, w), np.int32)
    gm[:, w // 2:] = 1
    L, a, b = imageio.hex_to_lab(paint_hex)
    lab = np.empty((h, w, 3), np.float32)
    lab[..., 0] = L
    factor = np.linspace(0.55, 1.0, w, dtype=np.float32)[None, :]
    lab[..., 1] = a * factor
    lab[..., 2] = b * factor
    lab[gm == 1] = (60.0, 0.0, 0.0)
    lab[DECAL] = (85.0, 0.0, 0.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    shading = np.full((h, w, 3), 0.85, np.float32)
    residual = np.zeros((h, w, 3), np.float32)
    lab_alb = imageio.linear_to_lab(albedo)
    groups = [_group(0, np.median(lab_alb[gm == 0], axis=0), int((gm == 0).sum())),
              _group(1, np.median(lab_alb[gm == 1], axis=0), int((gm == 1).sum()))]
    return albedo, shading, residual, gm, groups


def _hue_deg(lab: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) + 360.0) % 360.0


def test_texture_follows_the_new_hue():
    """Every repainted pixel must sit on the target's hue; the saturation modelling is
    carried over as saturation of the *new* colour (more saturated red -> more saturated
    blue), not as a hue offset from it."""
    scene = _textured_scene()
    target = "#3a5fa8"
    r = engine.Renderer(*scene)
    try:
        out = r.render({0: target}, RenderOptions())
    finally:
        r.free()
    inside = _interior(scene[3], 0)
    inside[20:44, 8:40] = False                       # the decal and a margin around it
    lab = imageio.rgb_to_lab(imageio.to_float(out))[inside]
    want = float(_hue_deg(np.array(imageio.hex_to_lab(target), np.float32)))
    dev = np.abs((_hue_deg(lab) - want + 180.0) % 360.0 - 180.0)
    assert np.percentile(dev, 98) < 6.0, f"hue drifted up to {np.percentile(dev, 98):.1f} deg off the target"
    src_c = np.hypot(*imageio.linear_to_lab(scene[0])[inside][:, 1:].T)
    new_c = np.hypot(lab[:, 1], lab[:, 2])
    assert _rank_corr(src_c, new_c) > 0.9, "saturation modelling did not carry over"


@pytest.mark.parametrize("target", ["#123f9e", "#3a5fa8", "#2e8b3d"])
def test_white_decal_stays_neutral_under_a_repaint(target):
    """A zero-chroma pixel inside the paint deviates from the group by exactly -A_ab; in
    the target's frame that is 'fully desaturated', so it must come out neutral (it used
    to come out as a saturated cyan on a navy repaint)."""
    scene = _textured_scene()
    r = engine.Renderer(*scene)
    try:
        out = r.render({0: target}, RenderOptions())
    finally:
        r.free()
    lab = imageio.rgb_to_lab(imageio.to_float(out[29:35, 18:30]))
    chroma = float(np.hypot(lab[..., 1], lab[..., 2]).mean())
    assert chroma < 10.0, f"{target} tinted the white decal: chroma {chroma:.1f}"


def test_hue_turn_fades_out_for_neutral_sources():
    """The same CIELAB colours as the engine's CIELAB era, converted to OK units (the engine's
    working space): the turn is exactly the OK hue difference, and a near-neutral side
    (CIELAB chroma 0.5, below the confidence ramp) turns nothing."""
    red = engine.cielab_to_oklab((45.0, 60.0, 45.0))
    navy = engine.cielab_to_oklab((30.0, 24.0, -56.0))
    grey = engine.cielab_to_oklab((50.0, 0.4, -0.3))
    want = np.arctan2(navy[2], navy[1]) - np.arctan2(red[2], red[1])          # navy hue minus red hue
    assert abs(engine._hue_turn(red, navy) - want) < 1e-4
    assert engine._hue_turn(grey, navy) == 0.0
    assert engine._hue_turn(red, grey) == 0.0
    assert abs(engine._hue_turn(red, red)) < 1e-6


# ------------------------------------------------------------------ repaint across lightness
#
# A real decomposition of a saturated surface leaks the paint into both the residual and
# the shading chromaticity. Added back untouched, that leak survives any repaint, which is
# what made a bright red part asked to become black come out muddy maroon.

def _leaky_scene(paint_hex: str = "#c8140a", leak_gradient: bool = False):
    """A saturated panel next to a neutral one, with the paint leaked into the shading and
    the positive residual the way the intrinsic model leaks it on real photographs. With
    ``leak_gradient`` the leak grows from 0.2x at the top to 1.8x at the bottom, the way
    real bounce is strongest where a part is darkest."""
    h, w = 64, 96
    gm = np.zeros((h, w), np.int32)
    gm[:, w // 2:] = 1                                   # neutral reference half
    paint = imageio.srgb_to_linear(imageio.hex_to_rgb01(paint_hex))
    albedo = np.empty((h, w, 3), np.float32)
    albedo[gm == 0] = paint
    albedo[gm == 1] = imageio.srgb_to_linear(np.float32([0.55, 0.55, 0.55]))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Real albedo is not flat: the decomposition leaves some modelling in it, and that is
    # exactly what an additive lightness shift used to clip away on a dark target.
    albedo = np.clip(albedo * (0.55 + 0.9 * (xx / w))[..., None], 0.0, 1.0).astype(np.float32)
    shade = (0.25 + 0.95 * (yy / h)).astype(np.float32)  # strong top-to-bottom modelling
    shading = np.repeat(shade[..., None], 3, axis=2)
    leak = np.float32([1.22, 0.97, 0.78])                # paint leaked into the light
    if leak_gradient:
        strength = (0.2 + 1.6 * yy / h)[..., None]
        shading[gm == 0] *= (1.0 + (leak - 1.0) * strength)[gm == 0]
    else:
        shading[gm == 0] *= leak
    residual = np.zeros((h, w, 3), np.float32)
    residual[gm == 0] = np.float32([0.045, 0.008, 0.006])    # broad veil in the paint's hue
    residual[6:12, 6:14] = 0.30                              # a white glint on top of it
    lab = imageio.linear_to_lab(albedo)
    groups = [_group(0, np.median(lab[gm == 0], axis=0), int((gm == 0).sum())),
              _group(1, np.median(lab[gm == 1], axis=0), int((gm == 1).sum()))]
    return albedo, shading, residual, gm, groups


@pytest.mark.parametrize("target,max_chroma,max_L", [("#000000", 6.0, 22.0), ("#141414", 6.0, 26.0)])
def test_dark_repaint_of_saturated_group_is_neutral(target, max_chroma, max_L):
    scene = _leaky_scene()
    r = engine.Renderer(*scene)
    try:
        out = r.render({0: target}, RenderOptions())
    finally:
        r.free()
    inside = _interior(scene[3], 0)
    lab = imageio.rgb_to_lab(imageio.to_float(out[inside][None]))[0]
    chroma = float(np.hypot(lab[..., 1], lab[..., 2]).mean())
    assert chroma < max_chroma, f"{target} kept the original hue: chroma {chroma:.1f}"
    assert float(lab[..., 0].mean()) < max_L, f"{target} came out washed: L {lab[..., 0].mean():.1f}"


def test_light_repaint_of_saturated_group_is_neutral():
    """The bound (mean CIELAB chroma < 9) is the original test's. The first OKLab port of the
    engine missed it at 9.16 (the paint-direction bounce retint leaves a warm remainder in
    OKLab); the integrated engine measures 4.3 because rule 7c caps a repainted pixel's OK
    chroma at 1.25 C_T + 0.003 (the target's own CIELAB chroma is 2.7). The second assertion
    pins that cap rather than loosening the original bound."""
    scene = _leaky_scene()
    r = engine.Renderer(*scene)
    try:
        out = r.render({0: "#f2f0eb"}, RenderOptions())
    finally:
        r.free()
    inside = _interior(scene[3], 0)
    lab = imageio.rgb_to_lab(imageio.to_float(out[inside][None]))[0]
    chroma = float(np.hypot(lab[..., 1], lab[..., 2]).mean())
    assert chroma < 9.0
    assert chroma < 6.0, f"the envelope's chroma cap no longer holds: {chroma:.2f}"
    assert float(lab[..., 0].mean()) > 55.0


def test_bounce_light_follows_the_new_paint():
    """The paint's leak into the shading is strongest in the shadows. Removing only its
    per-group median left the excess red light on the blue albedo, so a navy repaint of
    a red part came out teal in its shadows; the tint aligned with the old paint has to
    be recoloured per pixel."""
    scene = _leaky_scene(leak_gradient=True)
    target = "#123f9e"
    r = engine.Renderer(*scene)
    try:
        out = r.render({0: target}, RenderOptions())
    finally:
        r.free()
    inside = _interior(scene[3], 0)
    inside[2:16, 2:18] = False                        # the glint
    lab = imageio.rgb_to_lab(imageio.to_float(out))
    want = float(_hue_deg(np.array(imageio.hex_to_lab(target), np.float32)))
    dark = inside & (lab[..., 0] < np.median(lab[..., 0][inside]))
    dev = (_hue_deg(lab[dark]) - want + 180.0) % 360.0 - 180.0
    assert abs(float(np.median(dev))) < 4.0, f"shadows drifted {np.median(dev):+.1f} deg off the target hue"
    assert np.percentile(np.abs(dev), 90) < 8.0


def test_retint_leaves_the_light_alone_for_a_neutral_source():
    scene = _leaky_scene()
    r = engine.Renderer(*scene)
    try:
        level = r._base
        shading = level.shading
        hw = shading.shape[:2]
        ones = torch.ones(hw, device=r.device)
        bounce = torch.zeros(hw + (4,), device=r.device)
        bounce[..., 0] = bounce[..., 3] = 1.0
        paint = engine._Repaint(albedo=level.albedo, coverage=ones, hard=ones,
                                target=torch.zeros(hw + (3,), device=r.device), bounce=bounce,
                                src_ab=torch.zeros(hw + (2,), device=r.device))
        out = r._retint_shading(level, shading, paint)
        assert torch.allclose(out, shading, atol=2e-3)
        assert torch.allclose(engine.luminance_t(out), engine.luminance_t(shading), atol=1e-5)
    finally:
        r.free()


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


@pytest.mark.parametrize("target", ["#141414", "#1a2a5a", "#2c3539"])
def test_dark_repaint_keeps_albedo_modelling(target):
    """The old additive lightness shift sent every pixel below the group's own lightness
    to L <= 0, so more than half a red part clipped to featureless black on a dark target.
    The anchored map compresses that half instead of clipping it, so the modelling keeps
    its order and almost nothing lands on the floor."""
    scene = _leaky_scene()
    albedo, _, _, gm, _ = scene
    r = engine.Renderer(*scene)
    try:
        painted = r.recolor_albedo({0: target}, RenderOptions())
    finally:
        r.free()
    inside = _interior(gm, 0)
    src_L = imageio.linear_to_lab(albedo)[..., 0][inside]
    new_L = imageio.linear_to_lab(painted)[..., 0][inside]
    floor = float(np.mean(new_L <= 0.05))
    assert floor < 0.05, f"{target} crushed {floor:.0%} of the part onto a flat floor"
    assert _rank_corr(src_L, new_L) > 0.99, f"{target} lost the modelling's ordering"


def test_pure_black_target_zeroes_the_albedo_without_going_negative():
    """A #000000 paint reflects nothing, so its diffuse albedo is meant to collapse; what
    must not happen is a negative lightness that clips into flat patches with hard edges.
    What is still visible at pure black is the specular, covered by the glint test."""
    scene = _leaky_scene()
    r = engine.Renderer(*scene)
    try:
        painted = r.recolor_albedo({0: "#000000"}, RenderOptions())
    finally:
        r.free()
    inside = _interior(scene[3], 0)
    assert float(painted[inside].min()) >= 0.0
    assert float(imageio.linear_to_lab(painted)[..., 0][inside].mean()) < 12.0


def test_glint_survives_a_black_repaint():
    """The neutral part of the residual is the lamp, not the paint: a specular highlight
    must still read as a highlight after the part is painted black."""
    scene = _leaky_scene()
    r = engine.Renderer(*scene)
    try:
        out = r.render({0: "#000000"}, RenderOptions())
    finally:
        r.free()
    glint = imageio.to_float(out[7:11, 7:13]).mean()
    around = imageio.to_float(out[20:30, 6:14]).mean()
    assert glint > around * 2.0, f"glint {glint:.3f} vs surroundings {around:.3f}"


# ------------------------------------------------------------------ boundary coverage
#
# Repainting only *some* of a photo's groups used to leave a rim of the old paint around
# every repainted part: the coverage came from Gaussian-blurring the mapped-group
# indicator, so it fell below 1 inside the part itself and those pixels were only
# partially repainted. On a yellow motorcycle painted black that rim was a yellow outline
# around every panel and vent.

def _two_panel_scene(paint_hex: str = "#e8c010", neighbour_hex: str = "#1e1e1e"):
    """A saturated panel meeting a dark one, with the anti-aliased boundary a real
    photograph has: the labels are hard, the pixels between them are a mix."""
    h, w = 64, 96
    gm = np.zeros((h, w), np.int32)
    gm[:, w // 2:] = 1
    paint = imageio.srgb_to_linear(imageio.hex_to_rgb01(paint_hex))
    other = imageio.srgb_to_linear(imageio.hex_to_rgb01(neighbour_hex))
    albedo = np.empty((h, w, 3), np.float32)
    albedo[gm == 0] = paint
    albedo[gm == 1] = other
    # two columns of genuine mixture straddling the label boundary
    b = w // 2
    albedo[:, b - 1] = 0.66 * paint + 0.34 * other
    albedo[:, b] = 0.34 * paint + 0.66 * other
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    shading = np.repeat((0.45 + 0.7 * (yy / h))[..., None], 3, axis=2).astype(np.float32)
    residual = np.zeros((h, w, 3), np.float32)
    lab = imageio.linear_to_lab(albedo)
    groups = [_group(0, np.median(lab[gm == 0], axis=0), int((gm == 0).sum())),
              _group(1, np.median(lab[gm == 1], axis=0), int((gm == 1).sum()))]
    return albedo, shading, residual, gm, groups


def _source_hue_pixels(rgb_u8: np.ndarray, source_lab, chroma_min: float = 25.0) -> np.ndarray:
    lab = imageio.rgb_to_lab(imageio.to_float(rgb_u8))
    c = np.hypot(lab[..., 1], lab[..., 2])
    h = (np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) + 360.0) % 360.0
    src_h = (np.degrees(np.arctan2(source_lab[2], source_lab[1])) + 360.0) % 360.0
    d = np.abs(h - src_h) % 360.0
    return (c > chroma_min) & (np.minimum(d, 360.0 - d) < 30.0)


# feather 0 asks for no soft edge at all, so the one genuinely anti-aliased column that
# the label map hands to the neighbour is only partly reached; every other setting must
# clear the rim outright.
@pytest.mark.parametrize("feather,max_frac", [(0.0, 0.03), (1.5, 0.02), (3.0, 0.02)])
def test_partial_repaint_leaves_no_rim_of_the_old_paint(feather, max_frac):
    """Repaint one group of two and the old colour must not survive as an outline. The
    neighbour is deliberately left unmapped, which is the case that produced the rim."""
    scene = _two_panel_scene()
    gm, groups = scene[3], scene[4]
    r = engine.Renderer(*scene)
    try:
        out = r.render({0: "#000000"}, RenderOptions(feather_px=feather))
    finally:
        r.free()
    # everything the repainted group covers, plus the mixed columns beside it
    region = np.zeros(gm.shape, bool)
    region[:, : gm.shape[1] // 2 + 1] = True
    rim = _source_hue_pixels(out, np.array(groups[0].albedo_lab)) & region
    frac = float(rim.sum()) / float(region.sum())
    assert frac < max_frac, f"feather {feather}: {frac:.1%} of the repainted panel kept the old hue"


def test_repaint_reaches_the_edge_of_its_own_group():
    """Every pixel the label map assigns to a repainted group must be fully repainted,
    including the column right against the boundary. Partial coverage there is exactly
    what drew the outline."""
    scene = _two_panel_scene()
    albedo, _, _, gm, groups = scene
    r = engine.Renderer(*scene)
    try:
        painted = r.recolor_albedo({0: "#000000"}, RenderOptions(feather_px=1.5))
    finally:
        r.free()
    b = gm.shape[1] // 2
    interior = imageio.linear_to_lab(painted[:, 4:b - 6])
    border = imageio.linear_to_lab(painted[:, b - 4:b - 2])     # own group, hard against the edge
    assert float(np.hypot(border[..., 1], border[..., 2]).mean()) < 8.0, "border column kept the old chroma"
    assert abs(float(border[..., 0].mean()) - float(interior[..., 0].mean())) < 12.0


def test_unmapped_neighbour_is_left_alone():
    """The cure must not be the repaint spilling across the boundary: a group nobody
    mapped keeps its own colour a couple of pixels out."""
    scene = _two_panel_scene()
    gm = scene[3]
    r = engine.Renderer(*scene)
    try:
        out = r.render({0: "#000000"}, RenderOptions(feather_px=1.5))
        base = r.render({}, RenderOptions(feather_px=1.5))
    finally:
        r.free()
    b = gm.shape[1] // 2
    far = np.abs(out[:, b + 4:].astype(np.int16) - base[:, b + 4:].astype(np.int16)).max()
    assert far <= 6, f"repaint bled {far}/255 onto the untouched neighbour"


# ------------------------------------------------------------------ helpers for the rule scenes

def _lin(hexcol: str) -> np.ndarray:
    return np.float32(imageio.srgb_to_linear(imageio.hex_to_rgb01(hexcol)))


def _groups_for(albedo: np.ndarray, gm: np.ndarray, n: int, locked=()) -> list[ColorGroup]:
    lab = imageio.linear_to_lab(albedo)
    return [_group(g, np.median(lab[gm == g], axis=0), int((gm == g).sum()), locked=g in locked) for g in range(n)]


def _lab_hue(lab: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) + 360.0) % 360.0


# ------------------------------------------------------------------ rule 8: reflections of the old paint
#
# A red tank shows in the chrome next to it; repainting only the tank left the chrome red.

REFL_NEAR = np.s_[8:24, 50:70]        # the paint reflected in a neutral group, 10-30 px from the paint
REFL_PROTECTED = np.s_[40:56, 50:70]  # the same colour, marked by the analysis as an object of its own
REFL_BY_LOCKED = np.s_[8:20, 112:128] # the same colour right next to a locked group
LOCKED = np.s_[4:20, 130:150]
GOLD = np.s_[44:60, 130:150]


def _reflection_scene(lock: bool = True):
    """Red paint (group 0) next to a neutral 'chrome' group (1) that reflects it in three
    places, a darker red-hued material (2, locked by default) and gold (3, an unmapped
    group with a colour of its own). Flat light, no residual."""
    h, w = 64, 160
    gm = np.full((h, w), 1, np.int32)
    gm[:, :40] = 0
    gm[LOCKED] = 2
    gm[GOLD] = 3
    alb = np.empty((h, w, 3), np.float32)
    for gid, hx in enumerate(("#c8140a", "#8a8a8a", "#7a2a20", "#c9a227")):
        alb[gm == gid] = _lin(hx)
    for sl in (REFL_NEAR, REFL_PROTECTED, REFL_BY_LOCKED):
        alb[sl] = _lin("#a04030")                                   # OK hue 2.5 deg on the locked group's side
    shading = np.full((h, w, 3), 0.8, np.float32)
    residual = np.zeros((h, w, 3), np.float32)
    protect = np.zeros((h, w), bool)
    protect[REFL_PROTECTED] = True
    return alb, shading, residual, gm, _groups_for(alb, gm, 4, locked=(2,) if lock else ()), protect


def test_reflections_of_the_old_paint_take_the_new_colour():
    alb, shd, res, gm, groups, protect = _reflection_scene()
    r = engine.Renderer(alb, shd, res, gm, groups, protect=protect)
    try:
        base = r.render({}, RenderOptions())
        out = r.render({0: "#123f9e"}, RenderOptions())
    finally:
        r.free()
    lab_b = imageio.rgb_to_lab(imageio.to_float(base[REFL_NEAR]))
    lab_o = imageio.rgb_to_lab(imageio.to_float(out[REFL_NEAR]))
    want = float(_lab_hue(np.array(imageio.hex_to_lab("#123f9e"), np.float32)))
    dev = np.abs((_lab_hue(lab_o) - want + 180.0) % 360.0 - 180.0)
    assert np.median(_hue_deg(lab_b)) < 40.0                       # it was red
    assert np.median(dev) < 20.0, f"reflection hue {np.median(_lab_hue(lab_o)):.0f}, target {want:.0f}"
    assert np.median(np.hypot(lab_o[..., 1], lab_o[..., 2])) > 20.0  # a coloured reflection, not grey


def test_reflection_stage_never_touches_locked_protected_or_coloured_groups():
    alb, shd, res, gm, groups, protect = _reflection_scene()
    r = engine.Renderer(alb, shd, res, gm, groups, protect=protect)
    try:
        base = r.render({}, RenderOptions())
        out = r.render({0: "#123f9e"}, RenderOptions())
    finally:
        r.free()
    diff = np.abs(out.astype(np.int16) - base.astype(np.int16)).max(axis=2)
    for name, sl in (("locked group", LOCKED), ("protected pixels", REFL_PROTECTED), ("gold group", GOLD)):
        assert diff[sl].max() == 0, f"{name} changed by {diff[sl].max()}"
    assert diff[REFL_NEAR].max() > 40                               # while the reflection moved


def test_locked_groups_protect_their_neighbourhood_without_closing_the_window():
    """A locked group in the paint's own hue (a gold caliper next to yellow paint) guards the
    pixels around it, but it no longer cuts the paint's hue window: as an unlocked coloured
    group 1 deg (OK hue) from the paint it closes that side of the window, and a reflection
    on that side stays red. That is what switched rule 8 off on the yellow BMW."""
    alb, shd, res, gm, groups, protect = _reflection_scene(lock=True)
    r = engine.Renderer(alb, shd, res, gm, groups, protect=protect)
    try:
        base = r.render({}, RenderOptions())
        out = r.render({0: "#123f9e"}, RenderOptions())
    finally:
        r.free()
    diff = np.abs(out.astype(np.int16) - base.astype(np.int16)).max(axis=2)
    assert diff[8:20, 126:128].max() <= 8                           # 2-4 px from the locked group
    assert diff[REFL_NEAR].mean() > 40                              # 10-30 px from the paint
    alb, shd, res, gm, groups, protect = _reflection_scene(lock=False)
    r = engine.Renderer(alb, shd, res, gm, groups, protect=protect)
    try:
        open_ = np.abs(r.render({0: "#123f9e"}, RenderOptions()).astype(np.int16)
                       - r.render({}, RenderOptions()).astype(np.int16)).max(axis=2)
    finally:
        r.free()
    assert open_[REFL_NEAR].mean() < 0.25 * diff[REFL_NEAR].mean()


def test_identity_is_exact_with_islands_and_protect():
    alb, shd, res, gm, groups, protect = _reflection_scene()
    islands = np.zeros(gm.shape, bool)
    islands[30:36, 10:20] = True
    ref = engine.recompose(alb, shd, res)
    r = engine.Renderer(alb, shd, res, gm, groups, islands=islands, protect=protect)
    try:
        for opts in (RenderOptions(), RenderOptions(feather_px=4.0), RenderOptions(residual_tint=1.0)):
            assert np.array_equal(r.render({}, opts), ref)
            assert np.array_equal(r.render({0: None}, opts), ref)
    finally:
        r.free()


# ------------------------------------------------------------------ rule 1: decal islands

def _decal_scene():
    """A white decal on red paint whose outer 2 px are the anti-aliased mix of both."""
    h, w = 64, 96
    gm = np.zeros((h, w), np.int32)
    gm[24:40, 30:66] = 1
    alb = np.empty((h, w, 3), np.float32)
    alb[gm == 0] = _lin("#c8140a")
    alb[gm == 1] = _lin("#f4f4f4")
    rim = (gm == 1) & ~np.pad(np.ones((12, 32), bool), ((26, 26), (32, 32)))
    alb[rim] = 0.5 * (_lin("#c8140a") + _lin("#f4f4f4"))
    shading = np.full((h, w, 3), 0.8, np.float32)
    return alb, shading, np.zeros((h, w, 3), np.float32), gm, _groups_for(alb, gm, 2)


def test_coverage_ramp_never_enters_a_decal_island():
    alb, shd, res, gm, groups = _decal_scene()
    decal = gm == 1
    changed = {}
    for with_islands in (False, True):
        r = engine.Renderer(alb, shd, res, gm, groups, islands=decal if with_islands else None)
        try:
            # rule 1's coverage, seen through the repainted albedo (rule 8 is not in it)
            d = np.abs(r.recolor_albedo({0: "#123f9e"}, RenderOptions()) - alb).max(axis=2)
        finally:
            r.free()
        changed[with_islands] = int((d[decal] > 1e-6).sum())
    assert changed[False] > 60                                      # the ramp repaints the letter's mixed rim
    assert changed[True] == 0                                       # it never enters an island
    # an island whose own group is mapped is painted like anything else
    r = engine.Renderer(alb, shd, res, gm, groups, islands=decal)
    try:
        d = np.abs(r.render({0: "#123f9e", 1: "#123f9e"}, RenderOptions()).astype(np.int16)
                   - r.render({}, RenderOptions()).astype(np.int16)).max(axis=2)
    finally:
        r.free()
    assert d[decal].min() > 20


# ------------------------------------------------------------------ rule 7a: the black floor

def test_black_keeps_the_form_of_the_part():
    """#000000 renders as the darkest real paint lit by the photo's own shading: dark, but
    the shading gradient stays visible (without the floor it is a flat void)."""
    h, w = 64, 96
    gm = np.zeros((h, w), np.int32)
    gm[:, 64:] = 1
    alb = np.empty((h, w, 3), np.float32)
    alb[gm == 0] = _lin("#c8140a")
    alb[gm == 1] = _lin("#8a8a8a")
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    shd = np.repeat((0.15 + 1.1 * yy / h)[..., None], 3, axis=2).astype(np.float32)
    r = engine.Renderer(alb, shd, np.zeros((h, w, 3), np.float32), gm, _groups_for(alb, gm, 2))
    try:
        out = imageio.to_float(r.render({0: "#000000"}, RenderOptions()))[:, 4:56]
    finally:
        r.free()
    rows = out.mean(axis=(1, 2))
    assert rows[52:60].mean() > 2.0 * rows[4:12].mean() > 0.0       # lit rows brighter than shadowed ones
    assert float(imageio.rgb_to_lab(out)[..., 0].mean()) < 8.0      # and still black
    assert float(np.hypot(*imageio.rgb_to_lab(out)[..., 1:].transpose(2, 0, 1)).mean()) < 2.0


# ------------------------------------------------------------------ rules 2, 5, 6, 7b: highlights

SPECULAR = np.s_[10:18, 50:58]
SPECK = np.s_[40:42, 52:54]


def _specular_scene(magenta_albedo: bool = False):
    """Red paint lit from dark (left) to blown (right, the red channel clipped), with a
    64 px white specular in the bright part and a 4 px white speck, next to a grey panel.
    With ``magenta_albedo`` the albedo under the specular (and under a control patch in the
    dark part) is 35 deg magenta-shifted, the decomposition's error at clipped highlights."""
    h, w = 64, 96
    gm = np.zeros((h, w), np.int32)
    gm[:, 64:] = 1
    alb = np.empty((h, w, 3), np.float32)
    alb[gm == 0] = _lin("#c8140a")
    alb[gm == 1] = _lin("#8a8a8a")
    if magenta_albedo:
        L, a, b = imageio.hex_to_lab("#c8140a")
        t = np.radians(-35.0)
        mag = imageio.lab_to_linear(np.float32([[L, a * np.cos(t) - b * np.sin(t), a * np.sin(t) + b * np.cos(t)]]))[0]
        alb[SPECULAR] = mag
        alb[40:48, 20:28] = mag
    xx = np.mgrid[0:h, 0:w][1].astype(np.float32)
    shd = np.repeat(np.clip(0.4 + 1.6 * xx / 64, 0, 2.0)[..., None], 3, axis=2).astype(np.float32)
    shd[gm == 1] = 0.8
    res = np.zeros((h, w, 3), np.float32)
    res[SPECULAR] = 0.6
    res[SPECK] = 0.6
    return alb, shd, res, gm, _groups_for(alb, gm, 2)


def test_white_estimate_and_the_highlight_mask():
    """W is the photo's white specular split off against how white the paint looks at that
    brightness: the blown white specular, not the chroma-clipped red paint around it; the
    highlight mask keeps white clipped components of at least HL_MIN_AREA px."""
    r = engine.Renderer(*_specular_scene())
    try:
        W = r._gloss(r._base).cpu().numpy()
        hl = r._highlight(r._base)[..., 0].cpu().numpy()
    finally:
        r.free()
    assert abs(float(W[12:16, 52:56].mean()) - 0.6) < 0.03
    assert W[20:30, 50:60].max() == 0.0                             # clipped red paint is the paint's own
    assert W[20:40, 70:90].max() == 0.0                             # neutral groups have no gloss floor
    assert hl[11:17, 51:57].min() == 1.0
    assert hl[SPECK].max() == 0.0 and hl[20:30, 50:60].max() == 0.0


def test_white_specular_survives_a_repaint():
    r = engine.Renderer(*_specular_scene())
    try:
        W = r._gloss(r._base)[12:16, 52:56].cpu().numpy()
        for target in ("#000000", "#123f9e"):
            out = imageio.to_float(r.render({0: target}, RenderOptions())[12:16, 52:56])
            assert out.min() >= float(imageio.linear_to_srgb(np.float32(W.min()))) - 2 / 255, target
    finally:
        r.free()


def test_clipped_highlight_takes_the_target_hue():
    """Under a clipped white highlight the albedo's hue offset is glint contamination: only
    its radial part survives the repaint, so the highlight lands on the target's hue instead
    of rotating a magenta offset into cyan. The same albedo elsewhere is texture and keeps it."""
    r = engine.Renderer(*_specular_scene(magenta_albedo=True))
    try:
        alb = r.recolor_albedo({0: "#123f9e"}, RenderOptions())
    finally:
        r.free()
    ok = engine.linear_to_oklab_t(torch.from_numpy(alb)).numpy()
    t = engine.hex_to_oklab("#123f9e")
    dev = (np.degrees(np.arctan2(ok[..., 2], ok[..., 1]) - np.arctan2(t[2], t[1])) + 180.0) % 360.0 - 180.0
    assert abs(float(np.median(dev[11:17, 51:57]))) < 1.0
    assert abs(float(np.median(dev[41:47, 21:27]))) > 10.0


# ------------------------------------------------------------------ rule 6: no hairline at a boundary

def test_boundary_residual_floor_draws_no_hairline_on_a_dark_repaint():
    """A neutral residual floor on the two columns at a group boundary is decomposition
    error on mixed pixels; kept as a glint it drew a light line along every silhouette of
    a black repaint. The same floor inside the part is a real sheen and stays."""
    h, w = 64, 96
    gm = np.zeros((h, w), np.int32)
    gm[:, 48:] = 1
    alb = np.empty((h, w, 3), np.float32)
    alb[gm == 0] = _lin("#c8140a")
    alb[gm == 1] = _lin("#202020")
    res = np.zeros((h, w, 3), np.float32)
    res[:, 46:48] = 0.08
    res[:, 20:22] = 0.08
    r = engine.Renderer(alb, np.full((h, w, 3), 0.8, np.float32), res, gm, _groups_for(alb, gm, 2))
    try:
        out = imageio.to_float(r.render({0: "#000000"}, RenderOptions())).mean(axis=2)
    finally:
        r.free()
    assert abs(out[:, 46:48].mean() - out[:, 30:40].mean()) < 0.01
    assert out[:, 20:22].mean() > out[:, 30:40].mean() + 0.1


# ------------------------------------------------------------------ rule 7d: pastel shadows

def test_light_target_shadows_keep_their_chroma():
    """Where the photo's paint is duller in shadow, a pastel repaint went slate there. A
    repainted pixel darker than a light target keeps ENV_SHADOW_CHROMA * C_T * L / L_T."""
    h, w = 64, 96
    gm = np.zeros((h, w), np.int32)
    gm[:, 64:] = 1
    L, a, b = imageio.hex_to_lab("#c8140a")
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    dull = (0.3 + 0.7 * yy / h)[..., None]
    lab = np.empty((h, w, 3), np.float32)
    lab[..., 0] = L
    lab[..., 1:2] = a * dull
    lab[..., 2:3] = b * dull
    lab[gm == 1] = (58.0, 0.0, 0.0)
    alb = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    shd = np.repeat((0.12 + 1.1 * yy / h)[..., None], 3, axis=2).astype(np.float32)
    r = engine.Renderer(alb, shd, np.zeros((h, w, 3), np.float32), gm, _groups_for(alb, gm, 2))
    t = engine.hex_to_oklab("#7fb2ff")
    ct = float(np.hypot(t[1], t[2]))
    try:
        out = r.render({0: "#7fb2ff"}, RenderOptions())
    finally:
        r.free()
    ok = engine.linear_to_oklab_t(torch.from_numpy(imageio.srgb_to_linear(imageio.to_float(out[2:14, 6:56])))).numpy()
    ratio = np.hypot(ok[..., 1], ok[..., 2]) / (ct * np.clip(ok[..., 0] / t[0], 1e-3, 1.0))
    assert float(np.median(ratio)) > 0.8, f"shadow chroma ratio {np.median(ratio):.2f} (0.63 without the floor)"


# ------------------------------------------------------------------ bookkeeping

def test_pixel_distances_follow_the_reference_resolution(scene):
    """A full-resolution export passes the working resolution, so the reflection falloff
    and the edge bands cover the same part of the photo as in the preview."""
    r = engine.Renderer(*scene, reference_long_side=W // 2)
    try:
        assert r._px(r._base) == pytest.approx(2.0)
        assert r._px(r._level_for(W // 2)) == pytest.approx(1.0)
    finally:
        r.free()
    r = engine.Renderer(*scene)
    try:
        assert r._px(r._base) == pytest.approx(1.0)
    finally:
        r.free()


def test_per_mapping_caches_are_bounded(scene):
    r = engine.Renderer(*scene)
    try:
        for i in range(3 * engine._MAPPING_CACHE_SIZE):
            r.render({i % 3: "#1f5fd6", (i + 1) % 3: None} if i % 2 else {0: "#1f5fd6", 1: "#f2b705", 2: None}, RenderOptions())
            r.render_at(48, {i % 3: "#1f5fd6"}, RenderOptions())
        assert len(r._dist) <= engine._MAPPING_CACHE_SIZE
        assert len(r._paint_light) <= engine._MAPPING_CACHE_SIZE
    finally:
        r.free()


def test_masks_must_match_the_group_map(scene):
    albedo, shading, residual, group_map, groups = scene
    with pytest.raises(ValueError, match="islands"):
        engine.Renderer(albedo, shading, residual, group_map, groups, islands=np.zeros((3, 3), bool))
    with pytest.raises(ValueError, match="protect"):
        engine.Renderer(albedo, shading, residual, group_map, groups, protect=np.zeros((3, 3), bool))


# ------------------------------------------------------------------ rule 1: the ramp gate

def _backdrop_scene(mixed_rim: bool = False):
    """Red paint (group 0, x < 48) in front of a white backdrop (group 1). With
    ``mixed_rim`` the backdrop's first 2 columns hold half the paint (an anti-aliased edge
    the label stopped short of)."""
    h, w = 48, 96
    gm = np.zeros((h, w), np.int32)
    gm[:, 48:] = 1
    alb = np.empty((h, w, 3), np.float32)
    alb[gm == 0] = _lin("#c8140a")
    alb[gm == 1] = _lin("#f0f0f0")
    if mixed_rim:
        alb[:, 48:50] = 0.5 * (_lin("#c8140a") + _lin("#f0f0f0"))
    shading = np.full((h, w, 3), 0.9, np.float32)
    return alb, shading, np.zeros((h, w, 3), np.float32), gm, _groups_for(alb, gm, 2)


def test_ramp_gate_keeps_the_ramp_off_a_neighbour_that_is_its_own_colour():
    """The outward ramp covered the first 2-3 px of every neighbour: a grey contour around a
    white lens, a blue haze along every silhouette on a white backdrop. It now only enters
    pixels that still hold some of the old paint."""
    for mixed in (False, True):
        alb, shd, res, gm, groups = _backdrop_scene(mixed)
        r = engine.Renderer(alb, shd, res, gm, groups)
        try:
            d = np.abs(r.render({0: "#123f9e"}, RenderOptions()).astype(np.int16)
                       - r.render({}, RenderOptions()).astype(np.int16)).max(axis=2)
        finally:
            r.free()
        if mixed:
            assert d[:, 48:50].mean() > 40                          # the old paint in the rim is repainted
            assert d[:, 52:].max() <= 2
        else:
            assert d[:, 48:].max() <= 2, f"backdrop darkened by {d[:, 48:].max()}"
    # the ungated ramp is what drew the contour
    alb, shd, res, gm, groups = _backdrop_scene(False)
    r = engine.Renderer(alb, shd, res, gm, groups)
    try:
        r._ramp_gate = lambda level, a_L, src_ab: None
        d = np.abs(r.render({0: "#123f9e"}, RenderOptions()).astype(np.int16)
                   - r.render({}, RenderOptions()).astype(np.int16)).max(axis=2)
    finally:
        r.free()
    assert d[:, 48:50].max() > 30


def test_ramp_gate_opens_where_the_photo_edge_is_soft():
    """The decomposition sharpens the albedo at a defocused edge, so the albedo gate shut the
    ramp there and a repainted part read as a hard cut-out on a blurred background (the
    Exia's feet). Where the photograph's own edge is soft the ramp is let through; at a sharp
    edge it stays shut (the contour on a white backdrop does not come back)."""
    alb, shd, _, gm, groups = _backdrop_scene(False)
    h, w = gm.shape
    sharp = alb * shd
    x = np.arange(w, dtype=np.float32)

    def ramp(x0, width):
        t = np.clip((x - x0) / width, 0.0, 1.0)[None, :, None]
        return np.broadcast_to((1.0 - t) * (_lin("#c8140a") * 0.9) + t * (_lin("#f0f0f0") * 0.9), sharp.shape)

    moved, gate = {}, {}
    for name, photo in (("sharp", sharp), ("soft", ramp(45.5, 5.0)), ("wide", ramp(36.0, 24.0))):
        res = (photo - sharp).astype(np.float32)                       # same albedo, the photo differs
        r = engine.Renderer(alb, shd, res, gm, groups)
        try:
            d = np.abs(r.recolor_albedo({0: "#123f9e"}, RenderOptions()) - alb).max(axis=2)
            gate[name] = r._soft_edges(r._base).cpu().numpy()[:, 46:51]
        finally:
            r.free()
        moved[name] = float(d[:, 48:51].mean())
    assert moved["sharp"] < 0.02                                       # gated: only the colour snap's sliver
    assert moved["soft"] > 0.1 and moved["soft"] > 5 * moved["sharp"]  # a 5 px defocus: the ramp follows it
    assert gate["soft"].mean() > 0.6 and gate["sharp"].max() < 0.05
    assert gate["wide"].max() < 0.1 and moved["wide"] < moved["soft"]   # a 24 px gradient is shading, not an edge


def test_soft_edge_gate_only_opens_at_a_group_boundary():
    """A soft gradient inside a neighbour (a specular falling off on a chrome part) is not a
    soft edge of the paint: the gate stays shut away from the group boundary, so the ramp
    cannot re-enter a differently coloured neighbour through it."""
    alb, shd, _, gm, groups = _backdrop_scene(False)
    h, w = gm.shape
    sharp = alb * shd
    x = np.arange(w, dtype=np.float32)
    t = np.clip((x - 58.0) / 4.0, 0.0, 1.0)[None, :, None]             # a 4 px soft transition at x = 58..62
    photo = np.broadcast_to((1.0 - t) * (_lin("#f0f0f0") * 0.9) + t * (_lin("#a0a0a0") * 0.9), sharp.shape)
    photo = np.where(np.arange(w)[None, :, None] < 48, sharp, photo)
    r = engine.Renderer(alb, shd, (photo - sharp).astype(np.float32), gm, groups)
    try:
        gate = r._soft_edges(r._base).cpu().numpy()
    finally:
        r.free()
    assert gate[:, 56:64].max() < 0.05                                 # soft, but 10 px from the boundary
    assert gate.shape == gm.shape


def test_update_groups_takes_new_flags_without_rebuilding(scene):
    albedo, shading, residual, gm, groups = scene
    r = engine.Renderer(albedo, shading, residual, gm, groups)
    try:
        base_albedo = r._base.albedo
        before = r.render({RED: "#123f9e"}, RenderOptions())
        locked = [ColorGroup.from_dict({**g.to_dict(), "locked": g.id == RED}) for g in groups]
        r.update_groups(locked)
        ident = r.render({}, RenderOptions())
        assert np.array_equal(r.render({RED: "#123f9e"}, RenderOptions()), ident)   # a locked group is skipped
        r.update_groups(groups)
        assert np.array_equal(r.render({RED: "#123f9e"}, RenderOptions()), before)
        assert r._base.albedo is base_albedo                        # the layers were not uploaded again
        recoloured = [ColorGroup.from_dict({**g.to_dict(), "albedo_lab": (50.0, 0.0, 0.0)}) for g in groups]
        with pytest.raises(ValueError):
            r.update_groups(recoloured)
    finally:
        r.free()


def test_update_groups_can_swap_the_protect_mask():
    alb, shd, res, gm, groups, protect = _reflection_scene()
    r = engine.Renderer(alb, shd, res, gm, groups)
    try:
        base = r.render({}, RenderOptions())
        assert np.abs(r.render({0: "#123f9e"}, RenderOptions()).astype(np.int16) - base).max(axis=2)[REFL_PROTECTED].max() > 20
        r.render_at(32, {0: "#123f9e"}, RenderOptions())            # a resized level exists too
        r.update_groups(groups, protect=protect)
        out = r.render({0: "#123f9e"}, RenderOptions())
        assert np.abs(out.astype(np.int16) - base).max(axis=2)[REFL_PROTECTED].max() == 0
    finally:
        r.free()


# ------------------------------------------------------------------ rule 8: objects of the paint's colour

def _own_colour_scene():
    """Red paint (0), a dusty red-brown group (1, CIELAB L 36, chroma 17: below the
    'coloured group' bar) that is an object of that colour, a dark red-brown group (2, L 18:
    the paint's bounce on a dark surface) and a neutral grey group (3)."""
    h, w = 48, 160
    gm = np.full((h, w), 3, np.int32)
    gm[:, :40] = 0
    gm[:, 60:90] = 1
    gm[:, 100:130] = 2
    alb = np.empty((h, w, 3), np.float32)
    for gid, hx in enumerate(("#e8321e", "#704c46", "#402420", "#8a8a8a")):
        alb[gm == gid] = _lin(hx)
    shading = np.full((h, w, 3), 0.8, np.float32)
    return alb, shading, np.zeros((h, w, 3), np.float32), gm, _groups_for(alb, gm, 4)


def test_rule8_leaves_alone_a_group_that_is_an_object_of_the_paints_colour(monkeypatch):
    alb, shd, res, gm, groups = _own_colour_scene()

    def diff():
        r = engine.Renderer(alb, shd, res, gm, groups)
        try:
            return np.abs(r.render({0: "#123f9e"}, RenderOptions()).astype(np.int16)
                          - r.render({}, RenderOptions()).astype(np.int16)).max(axis=2)
        finally:
            r.free()

    d = diff()
    assert d[gm == 1].max() == 0                                    # a red-brown object stays red-brown
    assert d[gm == 2].mean() > 5                                    # a dark bounce of the paint is recoloured
    monkeypatch.setattr(engine, "REFL_OWN_COLOUR_SHARE", 1.01)      # without the guard ...
    assert diff()[gm == 1].mean() > 5                               # ... rule 8 took it for a reflection


def test_a_darker_target_does_not_keep_the_old_paints_residual_energy(monkeypatch):
    """The coloured residual excess is diffuse energy of the old paint: kept whole, a bright
    yellow's residual lifted a #1b2a57 navy repaint to a medium royal blue (four times the
    target's lightness). On the repainted labels a darker target now keeps only the square
    root of the product's drop of it; a lighter target is unchanged."""
    h, w = 40, 80
    gm = np.zeros((h, w), np.int32)
    gm[:, 60:] = 1
    alb = np.empty((h, w, 3), np.float32)
    alb[gm == 0] = _lin("#d4b000")
    alb[gm == 1] = _lin("#808080")
    shd = np.full((h, w, 3), 0.6, np.float32)
    res = (0.25 * alb * shd * (gm == 0)[..., None]).astype(np.float32)   # yellow excess, no neutral floor
    groups = _groups_for(alb, gm, 2)

    def paint_lum(target, gamma):
        monkeypatch.setattr(engine, "EXCESS_FOLLOW_GAMMA", gamma)
        r = engine.Renderer(alb, shd, res, gm, groups)
        try:
            out = imageio.srgb_to_linear(imageio.to_float(r.render({0: target}, RenderOptions())))
        finally:
            r.free()
        inner = out[5:35, 5:50]
        return float(np.median(0.2126 * inner[..., 0] + 0.7152 * inner[..., 1] + 0.0722 * inner[..., 2]))

    expected = 0.6 * float(np.dot(_lin("#1b2a57"), [0.2126, 0.7152, 0.0722]))
    kept = paint_lum("#1b2a57", 0.0)                                   # the old rule
    now = paint_lum("#1b2a57", 0.5)
    assert kept > 3.0 * expected                                       # the lift the user saw
    assert now < 0.5 * kept and now < 2.5 * expected
    assert abs(paint_lum("#fff3a0", 0.5) - paint_lum("#fff3a0", 0.0)) < 1e-4   # lighter target: unchanged


# ------------------------------------------------------------------ rule 8 for a small repainted part

def _small_part_scene(twin: bool = False):
    """A big neutral image with a small yellow part (group 0, 30 x 30 px: 0.2 % of the image), the
    part's own reflection 5 px from it and a yellow-hued reflection 80 px away in the same neutral
    group (1); with ``twin`` an unmapped gold object of the part's hue (2, larger than the part)."""
    h, w = 400, 600
    gm = np.full((h, w), 1, np.int32)
    gm[100:130, 100:130] = 0
    alb = np.empty((h, w, 3), np.float32)
    alb[...] = _lin("#8a8a8a")
    alb[gm == 0] = _lin("#d8b020")
    near, far = np.s_[105:125, 135:140], np.s_[105:125, 210:215]
    for sl in (near, far):
        alb[sl] = _lin("#a09050")                                    # the part's hue, dull
    if twin:
        gm[250:300, 300:400] = 2
        alb[gm == 2] = _lin("#c8a830")
    shading = np.full((h, w, 3), 0.8, np.float32)
    n = 3 if twin else 2
    return alb, shading, np.zeros((h, w, 3), np.float32), gm, _groups_for(alb, gm, n), near, far


def _diff(alb, shd, res, gm, groups, mapping):
    r = engine.Renderer(alb, shd, res, gm, groups)
    try:
        return np.abs(r.render(mapping, RenderOptions()).astype(np.int16) - r.render({}, RenderOptions()).astype(np.int16)).max(axis=2)
    finally:
        r.free()


def test_rule8_reach_of_a_small_part_is_its_own_size():
    """Painting a small part alone recolours its reflections next to it, not pixels of its hue
    80 px away (the Ducati's spring tinted the reflections of its gold frame 86 px away)."""
    alb, shd, res, gm, groups, near, far = _small_part_scene()
    d = _diff(alb, shd, res, gm, groups, {0: "#e63946"})
    assert d[near].mean() > 5 and d[far].max() == 0
    # a large group keeps the full reach
    assert engine.Renderer.__dict__["_reach"]                        # (the rule lives in Renderer._reach)
    r = engine.Renderer(alb, shd, res, gm, groups)
    try:
        assert r._reach((1,)) == engine.REFL_FALLOFF_PX and r._reach((0,)) == pytest.approx(30.0, abs=0.5)
    finally:
        r.free()


def test_rule8_leaves_the_reflections_alone_when_an_unmapped_twin_has_the_parts_colour():
    """An unmapped coloured object of the source's hue at least half its size closes the source's
    hue window: every pixel of that hue may be its reflection."""
    alb, shd, res, gm, groups, near, far = _small_part_scene(twin=True)
    d = _diff(alb, shd, res, gm, groups, {0: "#e63946"})
    assert d[near].max() == 0

    def window(gm_, alb_):
        r = engine.Renderer(alb_, shd, res, gm_, _groups_for(alb_, gm_, 3))
        try:
            rp = r._group_params({0: "#e63946"}, RenderOptions()).refl[0]
            return rp.z_pos, rp.z_neg
        finally:
            r.free()

    assert window(gm, alb) == (0.0, 0.0)                             # closed on both sides
    # a crumb of the hue (9 px) only stops the window short of its own hue, on its side
    gm2 = gm.copy()
    gm2[gm == 2] = 1
    gm2[250:253, 300:303] = 2
    alb2 = alb.copy()
    alb2[(gm == 2) & (gm2 == 1)] = _lin("#8a8a8a")
    assert max(window(gm2, alb2)) > 0.0


def test_instances_of_a_split_part_render_like_the_part():
    """Split by instance, both instances keep the part's albedo as their reference (ColorGroup.
    ref_lab) and the same target: the render is the one of the part before the split."""
    h, w = 80, 160
    gm = np.full((h, w), 2, np.int32)
    gm[10:70, 10:60] = 0
    gm[20:70, 100:150] = 1
    rng = np.random.default_rng(3)
    alb = np.empty((h, w, 3), np.float32)
    alb[gm == 2] = _lin("#808080")
    alb[gm == 0] = _lin("#b02010") * rng.uniform(0.8, 1.1, (int((gm == 0).sum()), 1)).astype(np.float32)
    alb[gm == 1] = _lin("#801508") * rng.uniform(0.8, 1.1, (int((gm == 1).sum()), 1)).astype(np.float32)   # in shadow
    shd = np.full((h, w, 3), 0.8, np.float32)
    res = np.zeros((h, w, 3), np.float32)
    part = gm.copy()
    part[gm == 1] = 0                                                # before the split: one group
    lab = imageio.linear_to_lab(alb)
    whole = _group(0, np.median(lab[part == 0], axis=0), int((part == 0).sum()))
    before = engine.render_once(alb, shd, res, part, [whole, _group(2, (53.6, 0, 0), int((gm == 2).sum()))],
                                {0: "#123f9e"})
    inst = [_group(g, np.median(lab[gm == g], axis=0), int((gm == g).sum())) for g in (0, 1)]
    grey = _group(2, (53.6, 0, 0), int((gm == 2).sum()))
    own = engine.render_once(alb, shd, res, gm, inst + [grey], {0: "#123f9e", 1: "#123f9e"})
    for g in inst:
        g.ref_lab = whole.albedo_lab
    kept = engine.render_once(alb, shd, res, gm, inst + [grey], {0: "#123f9e", 1: "#123f9e"})
    assert np.abs(kept.astype(int) - before.astype(int)).max() <= 1
    assert np.abs(own.astype(int) - before.astype(int))[gm == 1].mean() > 3   # its own albedo: another shade


def _glossy_instances():
    """A part in two instances: a saturated red one with a white highlight and a duller one of
    the same part out of the light (another whiteness at every brightness)."""
    h, w = 90, 180
    gm = np.full((h, w), 2, np.int32)
    gm[10:80, 10:80] = 0
    gm[10:80, 100:170] = 1
    rng = np.random.default_rng(5)
    alb = np.empty((h, w, 3), np.float32)
    alb[gm == 2] = _lin("#808080")
    alb[gm == 0] = _lin("#c02010") * rng.uniform(0.6, 1.1, (int((gm == 0).sum()), 1)).astype(np.float32)
    alb[gm == 1] = _lin("#a05a50") * rng.uniform(0.6, 1.1, (int((gm == 1).sum()), 1)).astype(np.float32)
    shd = np.full((h, w, 3), 0.8, np.float32)
    res = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    spot = np.exp(-(((yy - 40) / 9.0) ** 2 + ((xx - 40) / 12.0) ** 2))
    res[..., :] = (0.55 * spot)[..., None].astype(np.float32)          # a white highlight on the first
    res[gm != 0] = 0.0
    part = gm.copy()
    part[gm == 1] = 0
    lab = imageio.linear_to_lab(alb)
    whole = _group(0, np.median(lab[part == 0], axis=0), int((part == 0).sum()))
    grey = _group(2, (53.6, 0, 0), int((gm == 2).sum()))
    inst = [_group(g, np.median(lab[gm == g], axis=0), int((gm == g).sum())) for g in (0, 1)]
    for g in inst:
        g.part, g.ref_lab = "brake_caliper", whole.albedo_lab
    whole.part = "brake_caliper"
    return alb, shd, res, gm, part, whole, grey, inst


def test_instances_of_a_split_part_keep_the_parts_gloss_statistics():
    """The gloss split measures how white each group's paint is from its own pixels: split by
    instance, both instances keep the part's (the BMW's rear caliper, out of the light, changed
    157 px of the render around it when it measured its own)."""
    alb, shd, res, gm, part, whole, grey, inst = _glossy_instances()
    r_part = engine.Renderer(alb, shd, res, part, [whole, grey])
    r_inst = engine.Renderer(alb, shd, res, gm, inst + [grey])
    try:
        med_p, bins_p = r_part._ratios()
        med_i, bins_i = r_inst._ratios()
        for gid in (0, 1):
            assert float(med_i[gid]) == pytest.approx(float(med_p[0]), abs=1e-6)
            assert torch.allclose(bins_i[gid], bins_p[0], atol=1e-6)
        assert list(r_inst._pools()[:3]) == [0, 0, 2]
        assert float(med_p[0]) > 0.1                                   # the dull instance's whiteness counts
        before = r_part.render({0: "#123f9e"}, RenderOptions())
        after = r_inst.render({0: "#123f9e", 1: "#123f9e"}, RenderOptions())
        assert np.abs(after.astype(int) - before.astype(int)).max() <= 1
    finally:
        r_part.free()
        r_inst.free()
    # without the part's reference each group is measured on its own pixels
    own = [dataclasses.replace(g, ref_lab=None) for g in inst]
    r_own = engine.Renderer(alb, shd, res, gm, own + [grey])
    try:
        assert list(r_own._pools()[:3]) == [0, 1, 2]
    finally:
        r_own.free()


def test_touching_instances_of_a_split_part_have_no_seam_in_the_edge_band():
    """Two instances that touch (the robot's feet, joined by their rim): the seam between them is
    a group boundary only after the split, and the edge band's limits at it changed 43 px of the
    render. The band reads the instances as one part: the split renders as the part did."""
    alb, shd, res, gm, part, whole, grey, inst = _glossy_instances()
    gm = gm.copy()
    gm[10:80, 80:100] = 1                                            # the second instance reaches the first
    part = gm.copy()
    part[gm == 1] = 0
    alb = alb.copy()
    alb[:, 80:100][gm[:, 80:100] == 1] = _lin("#a05a50")
    res = res.copy()
    res[..., :] = 0.0
    yy, xx = np.mgrid[0:gm.shape[0], 0:gm.shape[1]]
    res[..., :] = (0.4 * np.exp(-(((yy - 40) / 20.0) ** 2 + ((xx - 80) / 14.0) ** 2)))[..., None].astype(np.float32)
    res[gm == 2] = 0.0                                               # a highlight across the seam
    lab = imageio.linear_to_lab(alb)
    whole = _group(0, np.median(lab[part == 0], axis=0), int((part == 0).sum()))
    whole.part = "brake_caliper"
    inst = [_group(g, np.median(lab[gm == g], axis=0), int((gm == g).sum())) for g in (0, 1)]
    for g in inst:
        g.part, g.ref_lab = "brake_caliper", whole.albedo_lab
    before = engine.render_once(alb, shd, res, part, [whole, grey], {0: "#123f9e"})
    after = engine.render_once(alb, shd, res, gm, inst + [grey], {0: "#123f9e", 1: "#123f9e"})
    assert np.abs(after.astype(int) - before.astype(int)).max() <= 1
    r = engine.Renderer(alb, shd, res, gm, inst + [grey])
    try:
        band, _ = r._edge_band(r._base)
        assert not bool(band[40, 75:85].any())                         # no band at the seam ...
        assert bool(band[40, 5:15].any())                              # ... one at the grey backdrop
    finally:
        r.free()


def test_pools_ignore_groups_without_a_reference_and_other_parts():
    """Only groups with the same part kind and the same reference are pooled."""
    alb, shd, res, gm, part, whole, grey, inst = _glossy_instances()
    inst[1].part = "shock_spring"
    r = engine.Renderer(alb, shd, res, gm, inst + [grey])
    try:
        assert list(r._pools()[:3]) == [0, 1, 2]
    finally:
        r.free()
