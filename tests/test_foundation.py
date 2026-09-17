import numpy as np
import pytest

from recolor import colornames, filters, imageio, types


def test_gamma_roundtrip():
    x = np.random.rand(64, 3).astype(np.float32)
    assert np.allclose(imageio.linear_to_srgb(imageio.srgb_to_linear(x)), x, atol=1e-5)


def test_hex_lab_roundtrip():
    for h in ("#ff0000", "#123456", "#fafafa", "#000000"):
        assert imageio.lab_to_hex(imageio.hex_to_lab(h)) == h


def test_fit_size_never_upscales():
    assert imageio.fit_size(3000, 2000, 1536) == (1536, 1024)
    assert imageio.fit_size(800, 600, 1536) == (800, 600)


def test_encode_roundtrip():
    import cv2
    img = (np.random.rand(20, 30, 3) * 255).astype(np.uint8)
    png = imageio.encode_png(img)
    back = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert np.array_equal(img, back)


def test_color_names_and_families():
    assert colornames.nearest_name(imageio.hex_to_lab("#dc143c")) == "Crimson"
    assert colornames.hue_family(imageio.hex_to_lab("#1f5fd6")) == "blue"
    assert colornames.hue_family(imageio.hex_to_lab("#808080")) == "neutral"
    assert colornames.hue_family(imageio.hex_to_lab("#ff8c1a")) == "orange"


def test_parse_color_words():
    got = colornames.parse_color_words("Navy and gold with racing green accents, plus #ff00aa")
    words = [w for w, _ in got]
    assert words == ["navy", "gold", "racing green", "#ff00aa"]


def test_guided_filter_shapes():
    g = np.random.rand(40, 50, 3).astype(np.float32)
    s = np.random.rand(40, 50).astype(np.float32)
    out = filters.guided_filter(g, s, radius=3)
    assert out.shape == (40, 50) and out.dtype == np.float32
    out3 = filters.guided_filter(g, np.random.rand(40, 50, 3).astype(np.float32), radius=3)
    assert out3.shape == (40, 50, 3)


def test_soft_weights_sum_to_one():
    gm = np.zeros((32, 32), np.int32)
    gm[:, 16:] = 1
    w = filters.soft_group_weights(gm, 2, feather_px=2.0)
    assert w.shape == (2, 32, 32)
    assert np.allclose(w.sum(0).cpu().numpy(), 1.0, atol=1e-4)
    assert 0.3 < float(w[0, 5, 16]) < 0.7  # blended at the edge


def test_refine_labels_keeps_ids():
    labels = np.zeros((64, 64), np.int32)
    labels[:, 32:] = 7
    guide = np.zeros((64, 64, 3), np.float32)
    guide[:, 30:] = 1.0
    out = filters.refine_labels_with_guide(labels, guide, radius=3)
    assert set(np.unique(out).tolist()) <= {0, 7}
    assert out[10, 5] == 0 and out[10, 60] == 7


def test_types_roundtrip():
    g = types.ColorGroup(id=1, name="Red", albedo_lab=(50.0, 60.0, 40.0), albedo_hex="#d62828",
                         area=100, area_frac=0.1, region_ids=[1, 2], hue_family="red")
    assert types.ColorGroup.from_dict(g.to_dict()) == g
    m = types.mapping_from_json({"1": "#00ff00", "2": None, "3": ""})
    assert m == {1: "#00ff00", 2: None, 3: None}
    assert types.RenderOptions.from_dict({"texture": 0.5, "bogus": 1}).texture == 0.5
