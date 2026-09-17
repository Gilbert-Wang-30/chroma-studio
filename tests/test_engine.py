"""Unit tests for the recoloring engine (recolor/engine.py). No models, no network.

A small synthetic scene stands in for an analyzed job: three groups (a red panel, a
grey panel and a green stripe), smooth colored shading, a white specular blob and a
clipped (negative residual) patch.
"""
from __future__ import annotations

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

def test_torch_lab_matches_imageio():
    rng = np.random.default_rng(1)
    rgb = rng.random((500, 3)).astype(np.float32)
    t = torch.from_numpy(rgb)
    assert np.abs(engine.srgb_to_lab_t(t).numpy() - imageio.rgb_to_lab(rgb)).max() < 1e-3
    lab = imageio.rgb_to_lab(rgb)
    assert np.abs(engine.lab_to_srgb_t(torch.from_numpy(lab)).numpy() - imageio.lab_to_rgb(lab)).max() < 1e-4
    lin = rng.random((500, 3)).astype(np.float32)
    assert np.abs(engine.linear_to_lab_t(torch.from_numpy(lin)).numpy() - imageio.linear_to_lab(lin)).max() < 1e-3
    assert np.abs(engine.lab_to_linear_t(torch.from_numpy(lab)).numpy() - imageio.lab_to_linear(lab)).max() < 1e-4


def test_gamut_compression_keeps_lightness_and_hue():
    lab = torch.tensor([[95.0, 80.0, 80.0], [20.0, -90.0, -90.0], [50.0, 0.0, 0.0], [60.0, 30.0, -40.0]])
    lin = engine.lab_to_linear_gamut_t(lab)
    assert torch.isfinite(lin).all() and float(lin.min()) >= 0.0 and float(lin.max()) <= 1.0
    back = engine.linear_to_lab_t(lin)
    assert torch.allclose(back[:, 0], lab[:, 0], atol=0.5)                   # L preserved
    hue_in = torch.atan2(lab[:2, 2], lab[:2, 1])
    hue_out = torch.atan2(back[:2, 2], back[:2, 1])
    assert torch.allclose(hue_in, hue_out, atol=0.05)                         # hue preserved
    assert torch.allclose(back[2:], lab[2:], atol=0.05)                       # in-gamut untouched


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
    red = torch.tensor([45.0, 60.0, 45.0])
    navy = torch.tensor([30.0, 24.0, -56.0])
    grey = torch.tensor([50.0, 0.4, -0.3])
    want = np.arctan2(-56.0, 24.0) - np.arctan2(45.0, 60.0)          # navy hue minus red hue
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
    scene = _leaky_scene()
    r = engine.Renderer(*scene)
    try:
        out = r.render({0: "#f2f0eb"}, RenderOptions())
    finally:
        r.free()
    inside = _interior(scene[3], 0)
    lab = imageio.rgb_to_lab(imageio.to_float(out[inside][None]))[0]
    assert float(np.hypot(lab[..., 1], lab[..., 2]).mean()) < 9.0
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
        shading = r._base.shading
        hw = shading.shape[:2]
        m = torch.ones(hw, device=r.device)
        bounce = torch.zeros(hw + (4,), device=r.device)
        bounce[..., 0] = bounce[..., 3] = 1.0
        out = r._retint_shading(shading, m, bounce, torch.zeros(hw + (2,), device=r.device))
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
