"""Palette tests: extraction on synthetic images, theme matching, parsed prompts,
caching and network-failure fallbacks.  No network: ``search_images`` is monkeypatched.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from recolor import config, imageio
from recolor import palette as pal
from recolor.palette import extract, sources, themes
from recolor.types import Palette


# ------------------------------------------------------------------ helpers

def flat_blocks(hexes: list[str], block: int = 40, rows: int = 1) -> np.ndarray:
    """uint8 RGB image made of equal flat blocks, one per hex."""
    cols = [np.round(imageio.hex_to_rgb01(h) * 255).astype(np.uint8) for h in hexes]
    strip = np.concatenate([np.full((block * rows, block, 3), c, np.uint8) for c in cols], axis=1)
    return strip


def synthetic_thumbs(images: list[np.ndarray]):
    """A ``fetch_thumbs`` stand-in that returns the given images (and writes them)."""
    def _fetch(items, cache_dir, width=640, max_workers=6):
        out = [(it, img) for it, img in zip(items, images)]
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            for i, (_, img) in enumerate(out):
                imageio.save_image(os.path.join(cache_dir, f"{i}.jpg"), img)
        return out
    return _fetch


def fake_items(k: int) -> list[dict]:
    return [{"url": f"https://example.org/{i}.jpg", "thumb_url": f"https://example.org/t{i}.jpg",
             "page_url": f"https://example.org/page/{i}", "title": f"img {i}", "license": "CC0",
             "width": 1200, "height": 800} for i in range(k)]


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = str(tmp_path / "palettes")
    monkeypatch.setattr(config, "PALETTE_CACHE_DIR", d)
    return d


@pytest.fixture
def offline(monkeypatch):
    """No source of images at all."""
    monkeypatch.setattr(sources, "search_images", lambda prompt, limit, thumb_width=640: [])
    monkeypatch.setattr(sources, "fetch_thumbs", synthetic_thumbs([]))


# ------------------------------------------------------------------ extraction

def test_extract_recovers_flat_colors():
    img = flat_blocks(["#d62828", "#1f5fd6", "#f7d51d", "#2e8b3d"], block=60)
    colors = extract.extract_palette([img], 4)
    assert len(colors) == 4
    got = {c.hex for c in colors}
    for h in ("#d62828", "#1f5fd6", "#f7d51d", "#2e8b3d"):
        lab = np.asarray(imageio.hex_to_lab(h), np.float32)
        d = min(float(imageio.delta_e(lab[None], np.asarray(c.lab, np.float32)[None])[0]) for c in colors)
        assert d < 3.0, (h, got)
    assert abs(sum(c.weight for c in colors) - 1.0) < 1e-5
    assert all(c.name for c in colors)
    assert [c.weight for c in colors] == sorted((c.weight for c in colors), reverse=True)


def test_extract_chroma_weighting_prefers_color_over_gray():
    # 85 % mid gray, 15 % vivid red: the red must survive and outrank the gray's neighbours.
    gray = np.full((100, 170, 3), 128, np.uint8)
    red = np.zeros((100, 30, 3), np.uint8)
    red[..., 0] = 214; red[..., 1] = 40; red[..., 2] = 40
    img = np.concatenate([gray, red], axis=1)
    colors = extract.extract_palette([img], 2)
    names = {c.name for c in colors}
    assert "Red" in names or "Crimson" in names or "Cherry" in names, names
    red_c = [c for c in colors if c.name in ("Red", "Crimson", "Cherry")][0]
    assert red_c.weight > 0.25       # 15 % of pixels but ~2.8x the weight each


def test_extract_merges_near_duplicates_and_drops_specks():
    # Two reds within dE 9 (merge), a blue and a yellow (so the lightness span rule is
    # satisfied), plus a mid-gray speck of 0.4 % of the pixels (dropped as < 2 %).
    base = flat_blocks(["#d62828", "#d82a2a", "#1f5fd6", "#f7d51d"], block=60)
    speck = np.full((60, 1, 3), 128, np.uint8)
    img = np.concatenate([base, speck], axis=1)
    colors = extract.extract_palette([img], 6)
    labs = np.asarray([c.lab for c in colors], np.float32)
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            assert float(imageio.delta_e(labs[i][None], labs[j][None])[0]) >= extract.MERGE_DELTA_E
    assert len(colors) == 3
    assert "Gray" not in {c.name for c in colors}
    assert all(c.weight >= extract.MIN_CLUSTER_WEIGHT for c in colors)


def test_extract_lightness_span_rule():
    # Six mid-tones of similar L plus tiny dark and light patches: darkest + lightest are re-added.
    mids = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#d35400", "#16a085"]
    img = flat_blocks(mids, block=80)
    dark = np.full((80, 3, 3), 8, np.uint8)
    light = np.full((80, 3, 3), 250, np.uint8)
    img = np.concatenate([img, dark, light], axis=1)
    colors = extract.extract_palette([img], 6)
    assert len(colors) == 6
    assert extract.lightness_span(colors) >= extract.MIN_LIGHTNESS_SPAN


def test_extract_empty_and_limits():
    assert extract.extract_palette([], 5) == []
    assert extract.extract_palette([flat_blocks(["#ff0000"])], 0) == []
    one = extract.extract_palette([flat_blocks(["#ff0000"], block=20)], 5)
    assert len(one) == 1 and abs(one[0].weight - 1.0) < 1e-6


def test_pool_pixels_caps_per_image():
    big = np.random.default_rng(0).integers(0, 255, (600, 600, 3), np.uint8)
    lab = extract.pool_pixels([big, big[:10, :10]])
    assert lab.shape == (extract.MAX_PIXELS_PER_IMAGE + 100, 3)
    assert lab.dtype == np.float32


def test_pool_pixels_tolerates_float_gray_and_rgba():
    red_u8 = flat_blocks(["#ff0000"], block=8)
    red_f32 = red_u8.astype(np.float32) / 255.0                 # [0,1] float, not /255 again
    lab_u8, lab_f32 = extract.pool_pixels([red_u8]), extract.pool_pixels([red_f32])
    assert np.allclose(lab_u8, lab_f32, atol=0.5)
    assert lab_f32[0, 0] > 40                                    # not an all-black palette
    gray = np.full((8, 8), 128, np.uint8)                        # 2-D grayscale -> neutral
    lab_g = extract.pool_pixels([gray])
    assert lab_g.shape == (64, 3) and abs(lab_g[0, 1]) < 1 and abs(lab_g[0, 2]) < 1
    rgba = np.dstack([red_u8, np.full(red_u8.shape[:2], 255, np.uint8)])
    assert np.allclose(extract.pool_pixels([rgba]), lab_u8)
    assert extract.pool_pixels([np.zeros((0, 3), np.uint8), np.zeros((4, 4, 2), np.uint8)]).shape == (0, 3)
    pal_f = extract.extract_palette([red_f32], 3)
    assert pal_f and imageio.delta_e(np.array([pal_f[0].lab], np.float32),
                                     np.array([imageio.hex_to_lab("#ff0000")], np.float32))[0] < 3


def test_weighted_kmeans_deterministic():
    pts = np.random.default_rng(1).random((2000, 3)).astype(np.float32) * 100
    w = np.ones(2000, np.float32)
    c1, w1 = extract.weighted_kmeans(pts, w, 5)
    c2, w2 = extract.weighted_kmeans(pts, w, 5)
    assert np.allclose(c1, c2, atol=1e-4) and np.allclose(w1, w2, atol=1e-2)
    assert abs(float(w1.sum()) - 2000.0) < 1e-2


# ------------------------------------------------------------------ themes

def test_themes_are_curated():
    assert len(themes.THEMES) >= 40
    for name, hexes in themes.THEMES.items():
        assert 5 <= len(hexes) <= 8, name
        labs = np.asarray([imageio.hex_to_lab(h) for h in hexes], np.float32)
        assert float(labs[:, 0].max() - labs[:, 0].min()) >= 25.0, name   # has a value structure
        for i in range(len(hexes)):
            assert hexes[i] == imageio.rgb01_to_hex(imageio.hex_to_rgb01(hexes[i]))
            for j in range(i + 1, len(hexes)):
                assert float(imageio.delta_e(labs[i][None], labs[j][None])[0]) >= 5.0, (name, hexes[i], hexes[j])
    for name in themes.THEME_KEYWORDS:
        assert name in themes.THEMES, name


@pytest.mark.parametrize("prompt,expected", [
    ("hawaii sunset", "hawaii sunset"),
    ("A Hawaiian sunset!", "hawaii sunset"),
    ("sunset over the bay", "sunset"),
    ("Stealth Matte finish", "stealth matte"),
    ("gulf livery please", "gulf racing"),
    ("cyber-punk", "cyberpunk"),
    ("neon tokyo street", "neon tokyo"),
    ("desert camo", "desert camo"),
    ("SAKURA", "sakura"),
    ("Racing", "racing livery"),
])
def test_match_theme(prompt, expected):
    got = themes.match_theme(prompt)
    assert got is not None and got[0] == expected
    assert got[1] == themes.THEMES[expected]
    got[1].append("#000000")                       # a copy, not the shared list
    assert themes.THEMES[expected][-1] != "#000000"


def test_match_theme_degenerate_inputs():
    assert themes.match_theme(None) is None
    assert themes.match_theme("") is None
    assert themes.match_theme("   ") is None
    assert themes.match_theme(42) is None


def test_match_theme_none_and_whole_words():
    assert themes.match_theme("") is None
    assert themes.match_theme("a bright orange bicycle") is None
    assert themes.match_theme("firewood") is None            # "fire" only matches as a word
    assert themes.match_theme("desert camo") != themes.match_theme("desert")


# ------------------------------------------------------------------ generate_palette

def test_parsed_prompt_offline(cache_dir, offline):
    p = pal.generate_palette("navy and gold", n_colors=6)
    assert p.method in ("parsed", "mixed")
    assert p.colors[0].hex == "#1a2a5a" and p.colors[1].hex == "#d4af37"
    assert len(p.colors) == 6
    assert abs(sum(c.weight for c in p.colors) - 1.0) < 1e-6
    assert p.colors[0].weight > p.colors[1].weight > p.colors[-1].weight
    assert p.sources == []
    assert p.id == pal.palette_id("navy and gold", 6)


def test_parsed_prompt_mixed_with_images(cache_dir, monkeypatch):
    green = flat_blocks(["#2e8b3d"], block=64)
    monkeypatch.setattr(sources, "search_images", lambda prompt, limit, thumb_width=640: fake_items(1))
    monkeypatch.setattr(sources, "fetch_thumbs", synthetic_thumbs([green]))
    p = pal.generate_palette("crimson and gold bicycle", n_colors=3)
    assert p.method == "mixed"
    assert [c.hex for c in p.colors[:2]] == ["#dc143c", "#d4af37"]
    lab = np.asarray(imageio.hex_to_lab("#2e8b3d"), np.float32)
    assert float(imageio.delta_e(lab[None], np.asarray(p.colors[2].lab, np.float32)[None])[0]) < 3
    assert len(p.sources) == 1 and p.sources[0].thumb == f"/api/palettes/{p.id}/sources/0.jpg"
    assert os.path.exists(os.path.join(cache_dir, p.id, "sources", "0.jpg"))


def test_theme_prompt_offline(cache_dir, offline):
    p = pal.generate_palette("Hawaii Sunset", n_colors=6)
    assert p.method == "theme"
    assert [c.hex for c in p.colors] == themes.THEMES["hawaii sunset"][:6]
    big = pal.generate_palette("hawaii sunset", n_colors=8)
    assert big.method == "theme" and len(big.colors) == 8     # theme has 7: one padded


def test_images_method_and_exact_count(cache_dir, monkeypatch):
    imgs = [flat_blocks(["#d62828", "#1f5fd6"], block=64), flat_blocks(["#f7d51d"], block=64)]
    monkeypatch.setattr(sources, "search_images", lambda prompt, limit, thumb_width=640: fake_items(2))
    monkeypatch.setattr(sources, "fetch_thumbs", synthetic_thumbs(imgs))
    p = pal.generate_palette("xyzzy widget", n_colors=6)
    assert p.method == "images"
    assert len(p.colors) == 6                        # 3 real colors, padded from fallback
    assert abs(sum(c.weight for c in p.colors) - 1.0) < 1e-6
    assert len({c.hex for c in p.colors}) == 6
    assert len(p.sources) == 2
    # Measured colors come first and weights are monotonically descending: padding
    # from the fallback set must never outrank a color actually seen in the images.
    ws = [c.weight for c in p.colors]
    assert ws == sorted(ws, reverse=True), ws
    real = {c.hex for c in extract.extract_palette(imgs, 6)}
    assert {c.hex for c in p.colors[:3]} == real
    assert max(ws[3:]) < min(ws[:3])


def test_images_padding_weights_below_measured(cache_dir, monkeypatch):
    # Unequal areas (100/60/30 px wide): the reviewer's reproduction of padding outranking.
    strip = np.concatenate([np.full((40, 100, 3), (214, 40, 40), np.uint8),
                            np.full((40, 60, 3), (31, 95, 214), np.uint8),
                            np.full((40, 30, 3), (247, 213, 29), np.uint8)], axis=1)
    monkeypatch.setattr(sources, "search_images", lambda prompt, limit, thumb_width=640: fake_items(1))
    monkeypatch.setattr(sources, "fetch_thumbs", synthetic_thumbs([strip]))
    p = pal.generate_palette("xyzzy gadget", n_colors=6)
    assert p.method == "images"
    ws = [c.weight for c in p.colors]
    assert ws == sorted(ws, reverse=True), ws
    assert ws[3] < ws[2] and ws[5] > 0
    assert abs(sum(ws) - 1.0) < 1e-6
    assert p.colors[0].name.lower().find("red") >= 0 or p.colors[0].lab[1] > 30   # red heaviest


def test_fallback_when_everything_fails(cache_dir, monkeypatch):
    def boom(prompt, limit, thumb_width=640):
        raise OSError("network down")
    monkeypatch.setattr(sources, "search_images", boom)
    p = pal.generate_palette("xyzzy widget", n_colors=5)
    assert p.method == "fallback"
    assert [c.hex for c in p.colors] == pal.FALLBACK_COLORS[:5]
    assert abs(sum(c.weight for c in p.colors) - 1.0) < 1e-6


def test_large_n_still_exact(cache_dir, offline):
    p = pal.generate_palette("xyzzy", n_colors=14)
    assert len(p.colors) == 14 and abs(sum(c.weight for c in p.colors) - 1.0) < 1e-6
    assert len({c.hex for c in p.colors}) == 14
    # Well beyond the API cap (16): still exact, and the tail is variants rather than
    # a run of identical fallback duplicates.
    huge = pal.generate_palette("xyzzy", n_colors=60)
    assert len(huge.colors) == 60 and abs(sum(c.weight for c in huge.colors) - 1.0) < 1e-6
    assert len({c.hex for c in huge.colors}) >= 50
    assert pal.generate_palette("xyzzy", n_colors="3").id == pal.palette_id("xyzzy", 3)
    assert len(pal.generate_palette("xyzzy", n_colors=0).colors) == 1


def test_caching_roundtrip(cache_dir, monkeypatch):
    calls = []

    def search(prompt, limit, thumb_width=640):
        calls.append(prompt)
        return fake_items(1)
    monkeypatch.setattr(sources, "search_images", search)
    monkeypatch.setattr(sources, "fetch_thumbs", synthetic_thumbs([flat_blocks(["#2e8b3d"], block=64)]))
    a = pal.generate_palette("Koi Pond", n_colors=4)
    b = pal.generate_palette("koi   pond", n_colors=4)            # normalized prompt -> same id
    assert a.id == b.id and calls == ["Koi Pond"]
    assert b.to_dict() == a.to_dict()
    with open(os.path.join(cache_dir, a.id, "palette.json")) as f:
        on_disk = Palette.from_dict(json.load(f))
    assert on_disk.to_dict() == a.to_dict()
    assert pal.load_palette(a.id).to_dict() == a.to_dict()
    assert pal.load_palette("nope") is None
    c = pal.generate_palette("koi pond", n_colors=4, use_cache=False)
    assert calls == ["Koi Pond", "koi pond"] and c.id == a.id
    assert pal.generate_palette("koi pond", n_colors=5).id != a.id   # n is part of the id


def test_progress_and_max_images_zero(cache_dir, monkeypatch):
    def never(prompt, limit, thumb_width=640):
        raise AssertionError("must not search when max_images == 0")
    monkeypatch.setattr(sources, "search_images", never)
    seen = []
    p = pal.generate_palette("cyberpunk", n_colors=6, max_images=0, progress=lambda f, m: seen.append(f))
    assert p.method == "theme"
    assert seen and seen == sorted(seen) and seen[-1] == 1.0


# ------------------------------------------------------------------ sources (no network)

def test_search_images_survives_failures(monkeypatch):
    monkeypatch.setattr(sources, "_search_commons", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    monkeypatch.setattr(sources, "_search_ddgs", lambda *a, **k: [])
    assert sources.search_images("anything", 5) == []
    assert sources.search_images("", 5) == []


def test_search_images_prefers_large_and_dedupes(monkeypatch):
    items = [
        {"url": "u1", "thumb_url": "t1", "page_url": "p1", "title": "small", "license": "CC0", "width": 300, "height": 200},
        {"url": "u2", "thumb_url": "t2", "page_url": "p2", "title": "big", "license": "CC0", "width": 3000, "height": 2000},
        {"url": "u2", "thumb_url": "t2", "page_url": "p2", "title": "dup", "license": "CC0", "width": 3000, "height": 2000},
    ]
    monkeypatch.setattr(sources, "_search_commons", lambda *a, **k: list(items))
    monkeypatch.setattr(sources, "_search_ddgs", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")))
    got = sources.search_images("q", 5)
    assert [g["title"] for g in got] == ["big", "small"]


def test_fetch_thumbs_skips_failures(monkeypatch, tmp_path):
    img = flat_blocks(["#123456"], block=48)
    def fake_get(url, timeout, max_bytes=None):
        if "bad" in url:
            raise OSError("404")
        return imageio.encode_jpeg(img)
    monkeypatch.setattr(sources, "_http_get", fake_get)
    items = [{"thumb_url": "https://x/bad", "url": "https://x/bad2"}, {"thumb_url": "https://x/ok"}]
    out = sources.fetch_thumbs(items, str(tmp_path / "s"), width=640)
    assert len(out) == 1 and out[0][0] is items[1]
    assert out[0][1].shape == (48, 48, 3)
    assert os.path.exists(tmp_path / "s" / "0.jpg")


def test_fetch_caps_bytes_and_skips_huge_originals(monkeypatch):
    img = flat_blocks(["#123456"], block=48)
    calls: list[tuple[str, int | None]] = []
    def fake_get(url, timeout, max_bytes=None):
        calls.append((url, max_bytes))
        if "thumb" in url:
            raise OSError("thumb 404")
        return imageio.encode_jpeg(img)
    monkeypatch.setattr(sources, "_http_get", fake_get)
    # Original is tried (with the byte cap) when its pixel count is reasonable...
    ok = {"thumb_url": "https://x/thumb1", "url": "https://x/orig1", "width": 4000, "height": 3000}
    assert sources._fetch_one(ok, 640) is not None
    assert calls == [("https://x/thumb1", sources.MAX_FETCH_BYTES), ("https://x/orig1", sources.MAX_FETCH_BYTES)]
    # ...but not for a giant file, so a slow multi-MB download can never happen.
    calls.clear()
    huge = {"thumb_url": "https://x/thumb2", "url": "https://x/orig2", "width": 12000, "height": 9000}
    assert sources._fetch_one(huge, 640) is None
    assert calls == [("https://x/thumb2", sources.MAX_FETCH_BYTES)]


def test_http_get_enforces_cap(monkeypatch):
    import io
    class FakeResp(io.BytesIO):
        def __init__(self, body, length=None):
            super().__init__(body)
            self.headers = {"Content-Length": str(length)} if length is not None else {}
        def __enter__(self): return self
        def __exit__(self, *a): return False
    body = b"x" * 1000
    monkeypatch.setattr(sources.urllib.request, "urlopen", lambda req, timeout: FakeResp(body))
    assert sources._http_get("https://x/a", 1.0) == body
    assert sources._http_get("https://x/a", 1.0, max_bytes=1000) == body
    with pytest.raises(ValueError):
        sources._http_get("https://x/a", 1.0, max_bytes=999)
    monkeypatch.setattr(sources.urllib.request, "urlopen", lambda req, timeout: FakeResp(body, length=50_000_000))
    with pytest.raises(ValueError):
        sources._http_get("https://x/a", 1.0, max_bytes=1000)
