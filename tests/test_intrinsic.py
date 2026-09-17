"""Tests for recolor.intrinsic. No model is loaded: the Careaga path is exercised by
monkeypatching ``careaga.load_models`` / ``careaga.run_model`` with fakes."""
from __future__ import annotations

import numpy as np
import pytest

from recolor import imageio, intrinsic
from recolor.intrinsic import careaga, heuristic


# ------------------------------------------------------------------ fixtures

def _gradient_scene(h: int = 96, w: int = 128, seed: int = 0) -> np.ndarray:
    """A textured scene with a soft illumination gradient and a highlight."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shading = 0.25 + 0.75 * (xx / w) * (0.6 + 0.4 * yy / h)
    albedo = np.stack([0.7 + 0.2 * np.sin(yy / 7), 0.3 + 0.1 * rng.random((h, w)), 0.2 * np.ones((h, w))], -1)
    lin = np.clip(albedo * shading[..., None], 0, 1).astype(np.float32)
    lin[42:47, 62:67] = 1.0  # blown highlight (< 0.5 % of pixels) -> forces albedo clipping
    return imageio.to_uint8(imageio.linear_to_srgb(lin))


def _flat_panel(h: int = 160, w: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """A flat red panel on a mid-grey background, crossed by a soft vertical shadow band
    (a local illumination feature: sigma 8 px, 50 % deep). Returns (image, panel_mask)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shadow = 1.0 - 0.5 * np.exp(-0.5 * ((xx - 120.0) / 8.0) ** 2)
    albedo = np.full((h, w, 3), 0.45, np.float32)
    mask = np.zeros((h, w), bool)
    mask[40:120, 60:180] = True
    albedo[mask] = (0.75, 0.08, 0.06)
    lin = np.clip(albedo * shadow[..., None], 0, 1).astype(np.float32)
    return imageio.to_uint8(imageio.linear_to_srgb(lin)), mask


def _lin(image: np.ndarray) -> np.ndarray:
    return imageio.srgb_to_linear(imageio.to_float(image))


def _assert_result_contract(image: np.ndarray, res: intrinsic.IntrinsicResult) -> None:
    h, w = image.shape[:2]
    for arr in (res.albedo, res.shading, res.residual):
        assert arr.shape == (h, w, 3)
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
    assert res.albedo.min() >= 0.0 and res.albedo.max() <= 1.0
    assert res.shading.min() >= 0.0
    assert intrinsic.reconstruction_error(image, res) < 1e-4


# ------------------------------------------------------------------ heuristic

def test_heuristic_identity_and_ranges():
    image = _gradient_scene()
    res = intrinsic.decompose(image, method="heuristic")
    assert res.method == "heuristic"
    _assert_result_contract(image, res)
    # Shading is grayscale: three identical channels, strictly positive.
    assert np.array_equal(res.shading[..., 0], res.shading[..., 1])
    assert np.array_equal(res.shading[..., 0], res.shading[..., 2])
    assert res.shading.min() > 0.0
    # Residual only appears where the albedo clipped (the blown highlight).
    clipped = res.albedo.max(-1) >= 1.0
    assert np.abs(res.residual[~clipped]).max() < 1e-6
    assert np.abs(res.residual[clipped]).max() > 1e-3


def test_heuristic_flat_panel_gives_flat_albedo_and_smooth_shading():
    image, mask = _flat_panel()
    res = intrinsic.decompose(image, method="heuristic")
    alb = res.albedo
    inner = np.zeros_like(mask)
    inner[48:112, 68:172] = True      # panel interior, away from the boundary
    # Flat albedo inside the panel although the shadow band crosses it: tiny spread per
    # channel relative to the mean (the input's own relative spread there is ~0.25).
    spread = alb[inner].std(0) / np.maximum(alb[inner].mean(0), 1e-3)
    assert spread.max() < 0.04, spread
    # The panel keeps its red chromaticity.
    m = alb[inner].mean(0)
    assert m[0] > 4 * m[1] and m[0] > 4 * m[2]
    # Background albedo is flat and neutral too, on both sides of the band.
    bg = np.zeros_like(mask)
    bg[5:30, 5:235] = True
    bspread = alb[bg].std(0) / np.maximum(alb[bg].mean(0), 1e-3)
    assert bspread.max() < 0.04, bspread
    assert np.abs(alb[bg].mean(0) - alb[bg].mean()).max() < 0.02
    # Shading: the band lands in it (a smooth ~50 % dip) ...
    shd = res.shading[..., 0]
    row = shd[80]
    assert row[120] < 0.65 * row[90] and row[120] < 0.65 * row[150]
    assert np.abs(np.diff(row, 2)).max() < 0.02 * row.mean()          # no ringing
    # ... and it is continuous across the panel edge: no halo, no step.
    assert abs(shd[80, 57] - shd[80, 63]) < 0.03 * shd[80, 60]
    assert abs(shd[37, 100] - shd[43, 100]) < 0.03 * shd[40, 100]


def test_heuristic_output_size_is_arbitrary():
    image = (np.random.default_rng(1).random((37, 53, 3)) * 255).astype(np.uint8)
    res = intrinsic.decompose(image, method="heuristic")
    _assert_result_contract(image, res)


def test_heuristic_rejects_bad_shapes():
    with pytest.raises(ValueError):
        heuristic.decompose_heuristic(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        intrinsic.decompose(np.zeros((10, 10, 3), np.float32), method="heuristic")


def test_recompose_matches_input():
    image = _gradient_scene()
    res = intrinsic.decompose(image, method="heuristic")
    back = intrinsic.recompose(res)
    assert back.dtype == np.uint8 and back.shape == image.shape
    assert np.abs(back.astype(int) - image.astype(int)).max() <= 1


# ------------------------------------------------------------------ careaga wrapper

def test_pad_to_multiple_and_crop_back():
    img = np.random.default_rng(2).random((45, 70, 3)).astype(np.float32)
    padded, size = careaga.pad_to_multiple(img)
    assert size == (45, 70)
    assert padded.shape == (64, 96, 3)
    assert padded.shape[0] % 32 == 0 and padded.shape[1] % 32 == 0
    assert np.array_equal(padded[:45, :70], img)                # content unchanged
    assert np.array_equal(padded[45, :70], img[43, :])          # reflect padding
    assert np.array_equal(careaga.crop_to(padded, size), img)
    aligned = np.zeros((64, 32, 3), np.float32)
    same, size2 = careaga.pad_to_multiple(aligned)
    assert same is aligned and size2 == (64, 32)
    tiny, _ = careaga.pad_to_multiple(np.ones((3, 5, 3), np.float32))
    assert tiny.shape == (32, 32, 3)                            # edge mode for tiny inputs


def _fake_pipeline(monkeypatch, calls: list):
    """Replace the model with a deterministic fake that behaves like run_pipeline:
    demands a 32-multiple size and returns hr_alb / dif_shd / residual / lin_img."""
    def fake_load():
        return {"fake": True}

    def fake_run(models, img01):
        assert models == {"fake": True}
        assert img01.shape[0] % 32 == 0 and img01.shape[1] % 32 == 0
        calls.append(img01.shape)
        lin = img01.astype(np.float64) ** 2.2
        shd = np.full_like(lin, 1.6)
        alb = np.clip(lin / shd, 0, 1)
        return {"hr_alb": alb.astype(np.float32), "dif_shd": shd.astype(np.float32),
                "residual": (lin - alb * shd), "lin_img": lin}

    monkeypatch.setattr(careaga, "is_available", lambda: True)
    monkeypatch.setattr(careaga, "load_models", fake_load)
    monkeypatch.setattr(careaga, "run_model", fake_run)


def test_careaga_wrapper_crops_back_and_keeps_identity(monkeypatch):
    calls: list = []
    _fake_pipeline(monkeypatch, calls)
    image = _gradient_scene(h=45, w=70)
    msgs: list[str] = []
    res = intrinsic.decompose(image, method="careaga", progress=lambda p, m: msgs.append(m))
    assert res.method == "careaga"
    assert calls == [(64, 96, 3)]
    _assert_result_contract(image, res)
    assert np.allclose(res.shading, 1.6)
    assert msgs and "ready" in msgs[-1].lower()


def test_careaga_oom_falls_back_to_heuristic(monkeypatch, caplog):
    import torch

    def fake_load():
        return {"fake": True}

    def boom(models, img01):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    monkeypatch.setattr(careaga, "is_available", lambda: True)
    monkeypatch.setattr(careaga, "load_models", fake_load)
    monkeypatch.setattr(careaga, "run_model", boom)
    image = _gradient_scene()
    with caplog.at_level("WARNING", logger="recolor.intrinsic"):
        res = intrinsic.decompose(image, method="careaga")
    assert res.method == "heuristic"
    _assert_result_contract(image, res)
    assert any("falling back" in r.message for r in caplog.records)


def test_careaga_nonfinite_output_falls_back_to_heuristic(monkeypatch, caplog):
    """The library divides by the 99th percentile of its rough albedo, which is 0/0 on
    an almost entirely black input (a logo on black): every layer comes back NaN. That
    must be detected and the heuristic used, never NaN layers with method='careaga'."""
    calls: list = []

    def fake_load():
        return {"fake": True}

    def nan_run(models, img01):
        calls.append(img01.shape)
        nan = np.full(img01.shape, np.nan, np.float32)
        return {"hr_alb": nan, "dif_shd": nan.copy(), "residual": nan.copy(), "lin_img": nan.copy()}

    monkeypatch.setattr(careaga, "is_available", lambda: True)
    monkeypatch.setattr(careaga, "load_models", fake_load)
    monkeypatch.setattr(careaga, "run_model", nan_run)
    image = np.zeros((70, 100, 3), np.uint8)
    image[30:36, 40:48] = 255                                   # 0.7 % non-black logo
    with caplog.at_level("WARNING", logger="recolor.intrinsic"):
        res = intrinsic.decompose(image, method="careaga")
    assert calls == [(96, 128, 3)]                              # the model was attempted
    assert res.method == "heuristic"
    _assert_result_contract(image, res)                         # finite, in range, identity
    assert any("non-finite" in r.message for r in caplog.records)
    # Display layers and the recomposition are usable, not black NaN casts.
    layers = intrinsic.layers_for_display(res)
    assert layers["albedo"][33, 44].max() > 200
    assert np.array_equal(intrinsic.recompose(res), image)
    # The wrapper itself raises (that is what decompose() catches).
    with pytest.raises(RuntimeError, match="non-finite"):
        careaga.decompose_careaga(image)


def test_careaga_explicit_request_uses_heuristic_when_unavailable(monkeypatch, caplog):
    def never():
        raise AssertionError("model load must not be attempted when unavailable")

    monkeypatch.setattr(careaga, "is_available", lambda: False)
    monkeypatch.setattr(careaga, "load_models", never)
    image = _gradient_scene()
    with caplog.at_level("WARNING", logger="recolor.intrinsic"):
        res = intrinsic.decompose(image, method="careaga")
    assert res.method == "heuristic"
    _assert_result_contract(image, res)
    assert any("not available" in r.message for r in caplog.records)


def test_empty_image_raises_value_error(monkeypatch):
    monkeypatch.setattr(careaga, "is_available", lambda: True)
    monkeypatch.setattr(careaga, "load_models", lambda: pytest.fail("must not load"))
    for shape in ((0, 10, 3), (10, 0, 3), (0, 0, 3)):
        empty = np.zeros(shape, np.uint8)
        for method in intrinsic.METHODS:
            with pytest.raises(ValueError, match="empty"):
                intrinsic.decompose(empty, method=method)
        with pytest.raises(ValueError, match="empty"):
            heuristic.decompose_heuristic(empty)
        with pytest.raises(ValueError, match="empty"):
            careaga.decompose_careaga(empty)


def test_heuristic_is_deterministic_above_quantile_sample(monkeypatch):
    """The albedo white point is measured on a strided subsample when the image has
    more pixels than QUANTILE_SAMPLE; two runs must be bit-identical (a random
    subsample was not)."""
    image = _gradient_scene(h=96, w=128)
    monkeypatch.setattr(heuristic, "QUANTILE_SAMPLE", 1000)   # 12288 px -> subsampled
    a = intrinsic.decompose(image, method="heuristic")
    b = intrinsic.decompose(image, method="heuristic")
    assert np.array_equal(a.albedo, b.albedo)
    assert np.array_equal(a.shading, b.shading)
    assert np.array_equal(a.residual, b.residual)
    _assert_result_contract(image, a)
    # The subsampled white point is close to the exact one.
    monkeypatch.setattr(heuristic, "QUANTILE_SAMPLE", 8_000_000)
    exact = intrinsic.decompose(image, method="heuristic")
    assert np.abs(a.albedo - exact.albedo).max() < 0.02


def test_auto_uses_heuristic_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(careaga, "is_available", lambda: False)
    image = _gradient_scene()
    res = intrinsic.decompose(image, method="auto")
    assert res.method == "heuristic"


def test_auto_respects_pixel_budget(monkeypatch):
    from recolor import config
    calls: list = []
    _fake_pipeline(monkeypatch, calls)
    monkeypatch.setattr(careaga, "is_available", lambda: True)
    monkeypatch.setattr(config, "FULLRES_INTRINSIC_MAX_PIXELS", 96 * 128 - 1)
    image = _gradient_scene(h=96, w=128)
    assert intrinsic.decompose(image, method="auto").method == "heuristic"
    assert calls == []
    monkeypatch.setattr(config, "FULLRES_INTRINSIC_MAX_PIXELS", 96 * 128)
    assert intrinsic.decompose(image, method="auto").method == "careaga"
    assert calls == [(96, 128, 3)]


def test_warmup_and_is_loaded_do_not_load_for_heuristic(monkeypatch):
    def never():
        raise AssertionError("model load must not happen")

    monkeypatch.setattr(careaga, "load_models", never)
    intrinsic.warmup("heuristic")
    assert intrinsic.is_loaded("heuristic") is True
    monkeypatch.setattr(careaga, "is_available", lambda: False)
    intrinsic.warmup("auto")                       # unavailable -> no-op
    assert intrinsic.is_loaded("careaga") is False
    with pytest.raises(ValueError):
        intrinsic.decompose(_gradient_scene(), method="magic")


# ------------------------------------------------------------------ display layers

def test_layers_for_display():
    image = _gradient_scene()
    res = intrinsic.decompose(image, method="heuristic")
    res.residual[10:20, 10:20] = -0.1                 # a negative residual patch
    layers = intrinsic.layers_for_display(res)
    assert set(layers) == {"albedo", "shading", "residual"}
    for arr in layers.values():
        assert arr.dtype == np.uint8 and arr.shape == image.shape
    # Shading normalized by its 99.5th percentile: brightest lit areas reach white.
    assert layers["shading"].max() >= 250
    # Residual is shown as |residual| x 4: the -0.1 patch reads as 0.4 linear.
    expected = imageio.to_uint8(imageio.linear_to_srgb(np.full((1, 1, 3), 0.4, np.float32)))[0, 0]
    assert np.array_equal(layers["residual"][15, 15], expected)
    assert np.array_equal(layers["albedo"], imageio.to_uint8(imageio.linear_to_srgb(res.albedo)))
