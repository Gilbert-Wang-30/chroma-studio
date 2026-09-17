"""HTTP API tests with every model-backed sibling replaced by a tiny fake.

The fakes are injected through `sys.modules` under the sibling's real module names,
so `recolor.pipeline`'s lazy imports resolve to them whether or not the real modules
exist yet, and the real `analyze` code path (artifact writing, events, id encoders)
runs end to end on a small synthetic image. No models, no network, no GPU needed.
"""
from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recolor import config, imageio, jobs, pipeline
from recolor.types import ColorGroup, Palette, PaletteColor, PaletteSource, Region


# ----------------------------------------------------------------------------- fakes

@dataclass
class _FakeIntrinsic:
    albedo: np.ndarray
    shading: np.ndarray
    residual: np.ndarray
    method: str


def _fake_intrinsic_module() -> types.ModuleType:
    m = types.ModuleType("recolor.intrinsic")

    def decompose(image_rgb_u8, method="auto", progress=None):
        lin = imageio.srgb_to_linear(imageio.to_float(image_rgb_u8))
        shading = np.full_like(lin, 0.8)
        albedo = np.clip(lin / shading, 0, 1)
        residual = lin - albedo * shading
        if progress:
            progress(0.5, "fake intrinsic halfway")
            progress(1.0)
        return _FakeIntrinsic(albedo.astype(np.float32), shading.astype(np.float32),
                              residual.astype(np.float32), "heuristic" if method != "careaga" else "careaga")

    def layers_for_display(res):
        u8 = imageio.to_uint8
        return {"albedo": u8(imageio.linear_to_srgb(res.albedo)),
                "shading": u8(imageio.linear_to_srgb(np.clip(res.shading, 0, 1))),
                "residual": u8(np.abs(res.residual) * 4)}

    m.decompose = decompose
    m.layers_for_display = layers_for_display
    m.warmup = lambda method="careaga": None
    m.is_loaded = lambda method="careaga": False
    m.IntrinsicResult = _FakeIntrinsic
    return m


def _fake_sam_module() -> types.ModuleType:
    m = types.ModuleType("recolor.segmentation.sam_masks")

    class SamMasker:
        def generate(self, image_rgb_u8, detail="balanced", progress=None):
            h, w = image_rgb_u8.shape[:2]
            left = np.zeros((h, w), bool)
            left[:, : w // 2] = True
            if progress:
                progress(0.5, "fake sam · 8 points/side")
            return [{"segmentation": left, "area": int(left.sum()), "bbox": [0, 0, w // 2, h],
                     "predicted_iou": 0.9, "stability_score": 0.9},
                    {"segmentation": ~left, "area": int((~left).sum()), "bbox": [w // 2, 0, w - w // 2, h],
                     "predicted_iou": 0.9, "stability_score": 0.9}]

    m.SamMasker = SamMasker
    m.DETAIL_PRESETS = {"fast": {}, "balanced": {}, "max": {}}
    return m


def _fake_hierarchy_module() -> types.ModuleType:
    m = types.ModuleType("recolor.segmentation.hierarchy")

    def build_regions(image_rgb_u8, albedo_lin, masks, detail="balanced", progress=None):
        h, w = image_rgb_u8.shape[:2]
        labels = np.zeros((h, w), np.int32)
        labels[:, w // 2:] = 1
        labels[h // 2:, w // 2:] = 2
        info = [{"source": "sam", "confidence": 0.9}, {"source": "sam", "confidence": 0.9},
                {"source": "split", "confidence": 0.0}]
        if progress:
            progress(1.0, "fake regions")
        return labels, info

    m.build_regions = build_regions
    return m


def _region(rid: int, labels: np.ndarray, albedo: np.ndarray, gid: int) -> Region:
    mask = labels == rid
    ys, xs = np.nonzero(mask)
    lab = imageio.linear_to_lab(np.median(albedo[mask], axis=0)[None, :])[0]
    lab_t = (float(lab[0]), float(lab[1]), float(lab[2]))
    h, w = labels.shape
    touches = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return Region(id=rid, area=int(mask.sum()), bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                  albedo_lab=lab_t, albedo_hex=imageio.lab_to_hex(lab_t), group_id=gid,
                  touches_border=touches, source="sam", confidence=0.9)


def _make_groups(regions: list[Region], labels: np.ndarray, assignment: dict[int, int]) -> tuple[list[ColorGroup], np.ndarray]:
    total = labels.size
    groups: list[ColorGroup] = []
    group_map = np.zeros_like(labels)
    for gid in sorted(set(assignment.values())):
        rids = [r.id for r in regions if assignment[r.id] == gid]
        area = sum(r.area for r in regions if r.id in rids)
        first = next(r for r in regions if r.id == rids[0])
        for rid in rids:
            group_map[labels == rid] = gid
        groups.append(ColorGroup(id=gid, name=f"Group {gid}", albedo_lab=first.albedo_lab, albedo_hex=first.albedo_hex,
                                 area=area, area_frac=area / total, region_ids=rids, hue_family="neutral"))
    for r in regions:
        r.group_id = assignment[r.id]
    return groups, group_map


def _fake_grouping_module() -> types.ModuleType:
    m = types.ModuleType("recolor.segmentation.grouping")

    def group_regions(labels, albedo_lin, region_info, max_groups=None, delta_e=10.0):
        n = int(labels.max()) + 1
        regions = [_region(i, labels, albedo_lin, i) for i in range(n)]
        assignment = {i: i for i in range(n)}
        if max_groups is not None:
            assignment = {i: min(i, max_groups - 1) for i in range(n)}
        groups, gm = _make_groups(regions, labels, assignment)
        return regions, groups, gm

    def regroup(regions, labels, albedo_lin, max_groups, delta_e):
        assignment = {r.id: (r.id if max_groups is None else min(r.id, max_groups - 1)) for r in regions}
        groups, gm = _make_groups(regions, labels, assignment)
        return regions, groups, gm

    def merge_groups(groups, regions, group_map, labels, ids):
        target = min(ids)
        assignment = {r.id: (target if r.group_id in ids else r.group_id) for r in regions}
        # renumber to 0..G-1 keeping order
        order = sorted(set(assignment.values()))
        remap = {g: i for i, g in enumerate(order)}
        assignment = {k: remap[v] for k, v in assignment.items()}
        new_groups, gm = _make_groups(regions, labels, assignment)
        for g in new_groups:
            old = next((o for o in groups if o.id == order[g.id]), None)
            if old is not None:
                g.locked, g.is_background, g.name = old.locked, old.is_background, old.name
        return regions, new_groups, gm

    def split_group(groups, regions, group_map, labels, albedo_lin, gid, k=2):
        assignment = {r.id: r.group_id for r in regions}
        members = [r for r in regions if r.group_id == gid]
        new_gid = max(assignment.values()) + 1
        for r in members[1:]:
            assignment[r.id] = new_gid
        new_groups, gm = _make_groups(regions, labels, assignment)
        return regions, new_groups, gm

    def move_regions(groups, regions, group_map, labels, region_ids, gid):
        assignment = {r.id: (gid if r.id in region_ids else r.group_id) for r in regions}
        new_groups, gm = _make_groups(regions, labels, assignment)
        return regions, new_groups, gm

    m.group_regions, m.regroup, m.merge_groups = group_regions, regroup, merge_groups
    m.split_group, m.move_regions = split_group, move_regions
    return m


def _fake_engine_module() -> types.ModuleType:
    m = types.ModuleType("recolor.engine")

    def _paint(albedo, shading, residual, group_map, groups, mapping, options):
        lin = albedo * shading + residual
        out = imageio.linear_to_srgb(lin)
        for gid, hexv in (mapping or {}).items():
            if hexv:
                out[group_map == gid] = imageio.hex_to_rgb01(hexv)
        return imageio.to_uint8(out)

    class Renderer:
        calls = 0

        def __init__(self, albedo_lin, shading_lin, residual, group_map, groups):
            self.layers = (albedo_lin, shading_lin, residual, group_map, groups)

        def render(self, mapping, options):
            Renderer.calls += 1
            return _paint(*self.layers, mapping, options)

        def render_at(self, long_side, mapping, options):
            return imageio.resize_long_side(self.render(mapping, options), long_side)

    m.Renderer = Renderer
    m.render_once = _paint
    return m


def _fake_palette_module(cache_root: str) -> types.ModuleType:
    m = types.ModuleType("recolor.palette")
    sources = types.ModuleType("recolor.palette.sources")
    sources.search_images = lambda prompt, limit: [{"url": "x", "page_url": "y", "title": "t", "license": "CC0",
                                                    "width": 800, "height": 600}]
    m.sources = sources

    def generate_palette(prompt, n_colors=6, max_images=6, progress=None):
        items = sources.search_images(prompt, max_images)   # stubbed in tests
        pid = "deadbeef" + str(n_colors)
        d = os.path.join(cache_root, pid, "sources")
        os.makedirs(d, exist_ok=True)
        imageio.save_image(os.path.join(d, "0.jpg"), np.full((8, 8, 3), 200, np.uint8))
        hexes = ["#ff6600", "#0044aa", "#ffffff", "#222222", "#33aa55", "#cc2255", "#ffcc00", "#8800ff"][:n_colors]
        colors = [PaletteColor(h, imageio.hex_to_lab(h), 1.0 / len(hexes), "x") for h in hexes]
        pal = Palette(id=pid, prompt=prompt, colors=colors, method="images" if items else "fallback", created=1.0,
                      sources=[PaletteSource(url="y", title="t", license="CC0", thumb=f"/api/palettes/{pid}/sources/0.jpg")])
        with open(os.path.join(cache_root, pid, "palette.json"), "w") as f:
            json.dump(pal.to_dict(), f)
        return pal

    m.generate_palette = generate_palette
    return m


def _fake_mapping_module() -> types.ModuleType:
    m = types.ModuleType("recolor.mapping")
    m.STRATEGIES = ["balanced", "area", "luminance", "hue", "contrast"]

    def suggest_mapping(groups, colors, strategy="balanced", keep_background=True, keep_locked=True):
        hexes = [c if isinstance(c, str) else c.hex for c in colors]
        out = {}
        for i, g in enumerate(groups):
            if (keep_locked and g.locked) or (keep_background and g.is_background):
                out[g.id] = None
            else:
                out[g.id] = hexes[i % len(hexes)]
        return out

    m.suggest_mapping = suggest_mapping
    return m


# ----------------------------------------------------------------------------- fixtures

def _sample_image() -> bytes:
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, :32] = (200, 40, 40)
    img[:24, 32:] = (40, 60, 210)
    img[24:, 32:] = (230, 230, 230)
    return imageio.encode_png(img)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated data dirs, fake siblings, synchronous analysis, a fresh registry."""
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", str(data))
    monkeypatch.setattr(config, "JOBS_DIR", str(data / "jobs"))
    monkeypatch.setattr(config, "CACHE_DIR", str(data / "cache"))
    monkeypatch.setattr(config, "PALETTE_CACHE_DIR", str(data / "cache" / "palettes"))
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "tiny.png").write_bytes(_sample_image())
    (samples / "MANIFEST.json").write_text(json.dumps({"tiny.png": {"title": "File:Tiny", "license": "CC0",
                                                                     "size": [64, 48], "artist": "me"}}))
    monkeypatch.setattr(config, "SAMPLES_DIR", str(samples))
    config.ensure_dirs()

    for name, mod in {
        "recolor.intrinsic": _fake_intrinsic_module(),
        "recolor.segmentation.sam_masks": _fake_sam_module(),
        "recolor.segmentation.hierarchy": _fake_hierarchy_module(),
        "recolor.segmentation.grouping": _fake_grouping_module(),
        "recolor.engine": _fake_engine_module(),
        "recolor.palette": _fake_palette_module(str(data / "cache" / "palettes")),
        "recolor.mapping": _fake_mapping_module(),
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    registry = jobs.JobRegistry(str(data / "jobs"))
    monkeypatch.setattr(jobs, "registry", registry)
    monkeypatch.setattr(pipeline, "registry", registry)
    monkeypatch.setattr(pipeline, "enqueue", lambda job: pipeline.analyze(job))
    monkeypatch.setattr(pipeline, "start_worker", lambda: None)
    pipeline._layers_cache.clear()
    pipeline._renderer_cache.clear()
    return {"data": data, "registry": registry}


@pytest.fixture
def client(env):
    from recolor.server.app import create_app
    with TestClient(create_app()) as c:
        yield c


def _create(client, **fields):
    r = client.post("/api/jobs", json={"sample": "tiny.png", **fields})
    assert r.status_code == 201, r.text
    return r.json()


def _events(client, jid):
    with client.stream("GET", f"/api/jobs/{jid}/events?once=1") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        return [json.loads(ln[5:]) for ln in r.iter_lines() if ln.startswith("data:")]


# ----------------------------------------------------------------------------- tests

def test_health_and_samples(client):
    h = client.get("/api/health").json()
    assert h["ok"] is True and set(h["models"]) == {"sam2", "intrinsic"}
    assert h["models"]["sam2"] in ("cold", "loading", "ready") and "vram_total_mb" in h and "jobs" in h
    s = client.get("/api/samples").json()
    assert s == [{"name": "tiny.png", "url": "/api/samples/tiny.png", "thumb": "/api/samples/tiny.png?w=320",
                  "width": 64, "height": 48, "title": "Tiny", "source_title": "Tiny", "license": "CC0", "artist": "me"}]
    full = client.get("/api/samples/tiny.png")
    assert full.status_code == 200 and full.headers["content-type"] == "image/png"
    thumb = client.get("/api/samples/tiny.png?w=32")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    arr = imageio.load_image(thumb.content)
    assert arr.shape == (24, 32, 3)
    cached = os.listdir(os.path.join(config.CACHE_DIR, "samples"))
    assert cached == ["tiny_w32.jpg"]
    assert client.get("/api/samples/../MANIFEST.json").status_code == 404
    missing = client.get("/api/samples/nope.jpg")
    assert missing.status_code == 404 and missing.json() == {"error": "sample_not_found", "detail": "no sample 'nope.jpg'"}


def test_create_from_sample_runs_pipeline(client, env):
    job = _create(client, detail="fast", max_groups=None, delta_e=12)
    assert job["status"] == "ready" and job["error"] is None
    assert job["options"] == {"detail": "fast", "intrinsic": "auto", "max_groups": None, "delta_e": 12.0}
    assert job["image"] == {"width": 64, "height": 48, "work_width": 64, "work_height": 48,
                            "preview_width": 64, "preview_height": 48}
    assert all(job["stages"][s]["state"] == "done" for s in job["stages"])
    assert all(job["stages"][s]["progress"] == 1 for s in job["stages"])
    assert job["stages"]["segment"]["message"] == "2 part proposals"
    assert job["timings"]["total_s"] >= 0
    assert job["intrinsic_method"] == "heuristic"
    assert len(job["groups"]) == 3 and job["regions_count"] == 3
    assert [g["id"] for g in job["groups"]] == [0, 1, 2]
    d = os.path.join(config.JOBS_DIR, job["id"])
    for f in ("job.json", "original.png", "work.png", "preview.jpg", "albedo.npy", "shading.npy", "residual.npy",
              "labels.npy", "group_map.npy", "regions.json", "layers/albedo.jpg", "layers/shading.jpg",
              "layers/residual.jpg", "layers/regions.png", "layers/groups.png", "layers/edges.png",
              "ids/regions.png", "ids/groups.png"):
        assert os.path.isfile(os.path.join(d, f)), f
    with open(os.path.join(d, "job.json")) as f:
        assert json.load(f) == client.get(f"/api/jobs/{job['id']}").json()

    listing = client.get("/api/jobs").json()
    assert listing[0] == {"id": job["id"], "name": "tiny.png", "created": job["created"], "status": "ready",
                          "thumb": f"/api/jobs/{job['id']}/layers/preview", "width": 64, "height": 48, "n_groups": 3}


def test_layers_and_id_encoders(client):
    job = _create(client)
    jid = job["id"]
    for layer in ("original", "work", "preview", "albedo", "shading", "residual", "regions", "groups", "edges"):
        r = client.get(f"/api/jobs/{jid}/layers/{layer}")
        assert r.status_code == 200, layer
        assert r.headers["content-type"].startswith("image/")
    assert client.get(f"/api/jobs/{jid}/layers/depth").status_code == 404

    ids = imageio.load_image(client.get(f"/api/jobs/{jid}/ids/regions").content)
    labels = np.load(os.path.join(config.JOBS_DIR, jid, "labels.npy"))
    assert np.array_equal(pipeline.decode_region_ids(ids), labels)
    gids = imageio.load_image(client.get(f"/api/jobs/{jid}/ids/groups").content)
    group_map = np.load(os.path.join(config.JOBS_DIR, jid, "group_map.npy"))
    assert np.array_equal(gids[..., 0].astype(np.int32), group_map)
    assert not gids[..., 1:].any()

    big = np.array([[0, 255, 256], [65536, 70000, 16777215]], np.int32)
    assert np.array_equal(pipeline.decode_region_ids(pipeline.encode_region_ids(big)), big)


def test_events_replay_and_live(client, env):
    job = _create(client)
    evs = _events(client, job["id"])
    assert evs[0] == {"type": "status", "status": "ready"}
    stages = [e for e in evs if e["type"] == "stage"]
    assert [e["stage"] for e in stages] == ["ingest", "intrinsic", "segment", "regions", "groups"]
    assert all(e["state"] == "done" and e["progress"] == 1 for e in stages)
    assert [e["type"] for e in evs[-2:]] == ["groups", "done"]
    assert len(evs[-2]["groups"]) == 3

    # live delivery: a subscriber receives what the job publishes, bounded queue never blocks
    j = env["registry"].get(job["id"])
    q = j.subscribe()
    j.set_stage("groups", "running", 0.5, "again")
    assert q.get_nowait() == {"type": "stage", "stage": "groups", "state": "running", "progress": 0.5, "message": "again"}
    for i in range(2000):
        j.publish({"type": "status", "status": f"spam{i}"})
    assert q.qsize() <= 512
    while not q.empty():
        q.get_nowait()
    j.unsubscribe(q)
    j.set_stage("groups", "done")
    assert q.empty()

    # live: an event published while the stream is open is delivered; the stream ends
    # by itself after the idle timeout (the heartbeat path is the same loop).
    import threading
    threading.Timer(0.2, lambda: j.set_stage("groups", "running", 0.7, "live tick")).start()
    with client.stream("GET", f"/api/jobs/{job['id']}/events?timeout=0.6") as r:
        lines = list(r.iter_lines())
    assert lines[0] == ": connected"
    assert "id: 1" in lines[:3]
    data = [json.loads(ln[5:]) for ln in lines if ln.startswith("data:")]
    assert data[-1] == {"type": "stage", "stage": "groups", "state": "running", "progress": 0.7, "message": "live tick"}
    assert client.get("/api/jobs/000000000000/events").status_code == 404


def test_error_in_stage_is_reported(client, env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("CUDA fell over")
    monkeypatch.setattr(sys.modules["recolor.segmentation.sam_masks"].SamMasker, "generate", boom)
    r = client.post("/api/jobs", json={"sample": "tiny.png"})
    job = r.json()
    assert job["status"] == "error"
    assert job["error"] == "segment: RuntimeError: CUDA fell over"
    assert job["stages"]["intrinsic"]["state"] == "done"
    assert job["stages"]["segment"] == {"state": "error", "progress": job["stages"]["segment"]["progress"],
                                        "message": "RuntimeError: CUDA fell over", "seconds": job["stages"]["segment"]["seconds"]}
    assert job["stages"]["regions"]["state"] == "skipped" and job["stages"]["groups"]["state"] == "skipped"
    evs = _events(client, job["id"])
    assert evs[-1] == {"type": "error", "message": "segment: RuntimeError: CUDA fell over"}
    for path in ("render", "export", "regroup", "groups/merge"):
        resp = client.post(f"/api/jobs/{job['id']}/{path}", json={"mapping": {}, "group_ids": [0, 1]})
        assert resp.status_code == 409, path
        assert resp.json()["error"] == "conflict"


def test_upload_multipart_and_limits(client, env, monkeypatch):
    png = _sample_image()
    r = client.post("/api/jobs", files={"file": ("shot.png", png, "image/png")},
                    data={"detail": "max", "intrinsic": "heuristic", "max_groups": "2", "delta_e": "8"})
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["name"] == "shot.png" and job["status"] == "ready"
    assert job["options"] == {"detail": "max", "intrinsic": "heuristic", "max_groups": 2, "delta_e": 8.0}
    assert len(job["groups"]) == 2
    assert os.path.isfile(os.path.join(config.JOBS_DIR, job["id"], "original.png"))

    jpg = imageio.encode_jpeg(imageio.load_image(png))
    r = client.post("/api/jobs", files={"file": ("../evil/../shot.jpg", jpg, "image/jpeg")})
    assert r.status_code == 201 and r.json()["name"] == "shot.jpg"
    assert os.path.isfile(os.path.join(config.JOBS_DIR, r.json()["id"], "original.jpg"))

    r = client.post("/api/jobs", files={"file": ("junk.jpg", b"not an image at all", "image/jpeg")})
    assert r.status_code == 415 and r.json()["error"] == "unsupported_image"
    r = client.post("/api/jobs", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert r.status_code == 400 and r.json()["error"] == "empty_file"
    r = client.post("/api/jobs", data={"detail": "fast"})
    assert r.status_code == 400 and r.json()["error"] == "no_file"
    r = client.post("/api/jobs", files={"file": ("shot.png", png, "image/png")}, data={"detail": "ultra"})
    assert r.status_code == 400 and "detail must be one of" in r.json()["detail"]
    r = client.post("/api/jobs", json={"sample": "tiny.png", "max_groups": 0})
    assert r.status_code == 400 and "max_groups" in r.json()["detail"]
    r = client.post("/api/jobs", json={})
    assert r.status_code == 400 and r.json()["error"] == "no_image"
    r = client.post("/api/jobs", content=b"{bad json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"] == "bad_json"

    monkeypatch.setattr("recolor.server.app.MAX_UPLOAD_BYTES", 1000)
    r = client.post("/api/jobs", files={"file": ("big.png", png * 20, "image/png")})
    assert r.status_code == 413 and r.json()["error"] == "too_large"
    r = client.post("/api/jobs", files={"file": ("big.png", png, "image/png")}, headers={"content-length": "999999999"})
    assert r.status_code == 413


def test_group_edits(client, env):
    job = _create(client)
    jid = job["id"]
    r = client.patch(f"/api/jobs/{jid}/groups/1", json={"name": "Trim", "locked": True})
    assert r.status_code == 200
    g1 = next(g for g in r.json()["groups"] if g["id"] == 1)
    assert g1["name"] == "Trim" and g1["locked"] is True and g1["is_background"] is False
    assert client.patch(f"/api/jobs/{jid}/groups/9", json={"name": "x"}).status_code == 400
    assert client.patch(f"/api/jobs/{jid}/groups/1", json={}).status_code == 400
    assert client.patch(f"/api/jobs/{jid}/groups/1", json={"name": "   "}).status_code == 400

    r = client.post(f"/api/jobs/{jid}/groups/merge", json={"group_ids": [1, 2]})
    assert r.status_code == 200 and [g["id"] for g in r.json()["groups"]] == [0, 1]
    assert r.json()["groups"][1]["region_ids"] == [1, 2] and r.json()["groups"][1]["name"] == "Trim"
    gm = np.load(os.path.join(config.JOBS_DIR, jid, "group_map.npy"))
    assert gm.max() == 1
    ids = imageio.load_image(client.get(f"/api/jobs/{jid}/ids/groups").content)
    assert ids[..., 0].max() == 1
    assert client.post(f"/api/jobs/{jid}/groups/merge", json={"group_ids": [1]}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/groups/merge", json={"group_ids": [1, 7]}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/groups/merge", json={"group_ids": "1,2"}).status_code == 400

    r = client.post(f"/api/jobs/{jid}/groups/split", json={"group_id": 1, "k": 2})
    assert r.status_code == 200 and [g["id"] for g in r.json()["groups"]] == [0, 1, 2]
    assert client.post(f"/api/jobs/{jid}/groups/split", json={"group_id": 1, "k": 1}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/groups/split", json={"k": 2}).status_code == 400

    r = client.post(f"/api/jobs/{jid}/groups/move", json={"region_ids": [2], "group_id": 0})
    assert r.status_code == 200
    assert next(g for g in r.json()["groups"] if g["id"] == 0)["region_ids"] == [0, 2]
    assert client.post(f"/api/jobs/{jid}/groups/move", json={"region_ids": [42], "group_id": 0}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/groups/move", json={"region_ids": [], "group_id": 0}).status_code == 400

    # a saved mapping follows the *regions* across edits (ids are renumbered by area),
    # not the ids: group 0 = region 0 (left half, 768 px) + region 2 (384 px), group 1 = region 1
    assert client.put(f"/api/jobs/{jid}/state", json={"mapping": {"0": "#ff0000", "1": "#00ff00", "7": None}}).status_code == 400
    assert client.put(f"/api/jobs/{jid}/state", json={"mapping": {"0": "#ff0000", "1": "#00ff00"}}).status_code == 200
    r = client.post(f"/api/jobs/{jid}/regroup", json={"max_groups": 3})
    assert r.status_code == 200 and [g["id"] for g in r.json()["groups"]] == [0, 1, 2]
    # old group 0's area-majority (region 0) is new group 0; old group 1 (region 1) is new group 1
    assert r.json()["mapping"] == {"0": "#ff0000", "1": "#00ff00"}
    assert client.put(f"/api/jobs/{jid}/state", json={"mapping": {"0": "#ff0000", "1": "#00ff00", "2": "#0000ff"}}).status_code == 200
    r = client.post(f"/api/jobs/{jid}/regroup", json={"max_groups": 2})
    assert r.status_code == 200
    body = r.json()
    assert [g["id"] for g in body["groups"]] == [0, 1] and body["options"]["max_groups"] == 2
    # regions 1 and 2 both land in new group 1: the first-listed (larger) old group's color wins
    assert body["mapping"] == {"0": "#ff0000", "1": "#00ff00"}
    # split 50/50 by area: neither half holds a majority, so the color is dropped rather than guessed
    r = client.post(f"/api/jobs/{jid}/groups/split", json={"group_id": 1, "k": 2})
    assert r.status_code == 200 and [g["id"] for g in r.json()["groups"]] == [0, 1, 2]
    assert r.json()["mapping"] == {"0": "#ff0000"}
    # move region 2 (384 px) into group 1 (region 1, 384 px): the color of the group whose regions
    # form the majority of the *new* group is what carries; group 0's color follows region 0
    assert client.put(f"/api/jobs/{jid}/state", json={"mapping": {"0": "#ff0000", "1": "#00ff00", "2": "#0000ff"}}).status_code == 200
    r = client.post(f"/api/jobs/{jid}/groups/move", json={"region_ids": [2], "group_id": 1})
    assert r.status_code == 200
    assert next(g for g in r.json()["groups"] if g["id"] == 1)["region_ids"] == [1, 2]
    assert r.json()["mapping"] == {"0": "#ff0000", "1": "#00ff00"}
    # a rename/lock ('update') never touches the mapping
    r = client.patch(f"/api/jobs/{jid}/groups/1", json={"name": "Both"})
    assert r.status_code == 200 and r.json()["mapping"] == {"0": "#ff0000", "1": "#00ff00"}
    assert client.post(f"/api/jobs/{jid}/regroup", json={"delta_e": "hot"}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/regroup", json={"max_groups": 0}).status_code == 400


def test_palette_endpoints(client, env, monkeypatch):
    calls = []
    fake_sources = sys.modules["recolor.palette"].sources
    monkeypatch.setattr(fake_sources, "search_images", lambda prompt, limit: calls.append(prompt) or [])
    r = client.post("/api/palettes", json={"prompt": "hawaii sunset", "n_colors": 4})
    assert r.status_code == 200, r.text
    pal = r.json()
    assert calls == ["hawaii sunset"]
    assert pal["prompt"] == "hawaii sunset" and len(pal["colors"]) == 4 and pal["method"] == "fallback"
    assert pal["sources"][0]["thumb"] == f"/api/palettes/{pal['id']}/sources/0.jpg"
    assert client.get(f"/api/palettes/{pal['id']}").json() == pal
    src = client.get(f"/api/palettes/{pal['id']}/sources/0.jpg")
    assert src.status_code == 200 and src.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/palettes/{pal['id']}/sources/5.jpg").status_code == 404
    assert client.get("/api/palettes/nope").status_code == 404
    assert client.get("/api/palettes/..%2F..").status_code == 404
    assert client.post("/api/palettes", json={"prompt": "   "}).status_code == 400
    assert client.post("/api/palettes", json={"prompt": "x", "n_colors": 99}).status_code == 400


def test_mapping_suggest_render_export_state(client, env):
    job = _create(client)
    jid = job["id"]
    client.patch(f"/api/jobs/{jid}/groups/2", json={"locked": True})
    r = client.post(f"/api/jobs/{jid}/mapping/suggest", json={"colors": ["#ff0000", "#00ff00"], "strategy": "hue"})
    assert r.status_code == 200
    assert r.json() == {"mapping": {"0": "#ff0000", "1": "#00ff00", "2": None}, "strategy": "hue"}
    assert client.post(f"/api/jobs/{jid}/mapping/suggest", json={"colors": ["#ff0000"], "strategy": "vibes"}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/mapping/suggest", json={"colors": ["red"]}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/mapping/suggest", json={"colors": []}).status_code == 400

    r = client.post(f"/api/jobs/{jid}/render", json={"mapping": {"0": "#00ff00"}, "options": {"feather_px": 2}})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/jpeg" and r.headers["x-render-ms"].isdigit()
    img = imageio.load_image(r.content)
    assert img.shape == (48, 64, 3)
    assert img[10, 10, 1] > 200 and img[10, 10, 0] < 60          # group 0 painted green
    assert img[10, 50, 2] > 150                                  # group 1 untouched (blue)
    r = client.post(f"/api/jobs/{jid}/render", json={"mapping": {"0": "#00ff00"}, "options": {"mode": "wild"}})
    assert r.status_code == 400 and "options.mode" in r.json()["detail"]
    assert client.post(f"/api/jobs/{jid}/render", json={"mapping": {"7": "#00ff00"}}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/render", json={"mapping": {"0": "green"}}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/render", json={"mapping": {"0": "#00ff00"}, "options": {"saturation": 9}}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/render", json=[1, 2]).status_code == 400

    r = client.post(f"/api/jobs/{jid}/export", json={"mapping": {"1": "#ffff00"}, "options": {}, "quality": "work", "format": "png"})
    assert r.status_code == 200, r.text
    exp = r.json()
    assert exp["width"] == 64 and exp["height"] == 48 and exp["ms"] >= 0
    assert exp["url"] == f"/api/jobs/{jid}/exports/{exp['file']}" and exp["file"].endswith(".png")
    dl = client.get(exp["url"])
    assert dl.status_code == 200 and dl.headers["content-type"] == "image/png"
    assert dl.headers["content-disposition"].startswith("attachment;") and exp["file"] in dl.headers["content-disposition"]
    out = imageio.load_image(dl.content)
    assert out[5, 50].tolist() == [255, 255, 0]
    r = client.post(f"/api/jobs/{jid}/export", json={"mapping": {}, "quality": "full", "format": "png"})
    assert r.status_code == 200 and r.json()["file"].endswith("_full_" + r.json()["file"].split("_full_")[1])
    full = imageio.load_image(client.get(r.json()["url"]).content)
    assert full.shape == (48, 64, 3)
    original = imageio.load_image(client.get(f"/api/jobs/{jid}/layers/original").content)
    assert np.abs(full.astype(int) - original.astype(int)).max() <= 2     # identity mapping reproduces the original
    r = client.post(f"/api/jobs/{jid}/export", json={"mapping": {}, "quality": "full", "format": "jpeg"})
    assert r.status_code == 200 and r.json()["file"].endswith(".jpg")
    assert client.get(r.json()["url"]).headers["content-type"] == "image/jpeg"
    assert client.post(f"/api/jobs/{jid}/export", json={"quality": "huge"}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/export", json={"format": "gif"}).status_code == 400
    assert client.get(f"/api/jobs/{jid}/exports/../job.json").status_code == 404
    assert client.get(f"/api/jobs/{jid}/exports/missing.png").status_code == 404

    r = client.put(f"/api/jobs/{jid}/state", json={"mapping": {"0": "#abcdef", "1": ""},
                                                   "render_options": {"feather_px": 3, "bogus": 1}, "palette_id": "deadbeef4"})
    assert r.status_code == 200
    body = r.json()
    assert body["mapping"] == {"0": "#abcdef", "1": None}
    assert body["render_options"]["feather_px"] == 3 and "bogus" not in body["render_options"]
    assert body["palette_id"] == "deadbeef4"
    assert client.get(f"/api/jobs/{jid}").json() == body
    with open(os.path.join(config.JOBS_DIR, jid, "job.json")) as f:
        assert json.load(f)["palette_id"] == "deadbeef4"
    assert client.put(f"/api/jobs/{jid}/state", json={}).status_code == 400
    assert client.put(f"/api/jobs/{jid}/state", json={"palette_id": "../x"}).status_code == 400


def test_delete_and_errors(client, env):
    job = _create(client)
    jid = job["id"]
    assert client.get(f"/api/jobs/{jid}/layers/preview").status_code == 200
    r = client.delete(f"/api/jobs/{jid}")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert not os.path.exists(os.path.join(config.JOBS_DIR, jid))
    assert client.get("/api/jobs").json() == []
    r = client.get(f"/api/jobs/{jid}")
    assert r.status_code == 404 and r.json() == {"error": "job_not_found", "detail": f"no job '{jid}'"}
    assert client.delete(f"/api/jobs/{jid}").status_code == 404
    r = client.get("/api/jobs/../../etc/passwd")
    assert r.status_code == 404 and "error" in r.json()
    r = client.get("/api/nothing-here")
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    r = client.patch(f"/api/jobs/{'a' * 12}/groups/notanumber", json={"name": "x"})
    assert r.status_code == 422 and r.json()["error"] == "invalid" and "gid" in r.json()["detail"]


def test_registry_reload_and_resume(env, monkeypatch):
    registry = env["registry"]
    img = imageio.load_image(_sample_image())
    from recolor.types import AnalysisOptions
    job = registry.create(img, "keep.jpg", AnalysisOptions(detail="max"))
    job.set_stage("ingest", "running", 0.3, "half")
    job.set_status("analyzing")
    fresh = jobs.JobRegistry(registry.root)
    loaded = fresh.get(job.id)
    assert loaded is not None and loaded.meta == job.meta
    assert fresh.list()[0]["status"] == "analyzing"
    monkeypatch.setattr(pipeline, "registry", fresh)
    queued = []
    monkeypatch.setattr(pipeline, "enqueue", lambda j: queued.append(j.id))
    assert pipeline.resume_pending() == 1
    assert queued == [job.id] and fresh.get(job.id).status == "queued"
    assert fresh.get(job.id).meta["stages"]["ingest"]["state"] == "idle"


def test_index_served_or_placeholder(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert r.headers["cache-control"] == "no-store"


def test_cuda_oom_is_retried_once_then_humanized(client, env, monkeypatch):
    class OutOfMemoryError(RuntimeError):
        pass

    oom = OutOfMemoryError("CUDA out of memory. Tried to allocate 1024.00 MiB. GPU 0 has a total capacity of "
                           "31.35 GiB of which 459.12 MiB is free. Process 1 has 20.00 GiB memory in use.")
    monkeypatch.setattr(pipeline, "_OOM_RETRY_DELAY_S", 0.0)
    sam = sys.modules["recolor.segmentation.sam_masks"].SamMasker
    real_generate, calls = sam.generate, []

    def flaky(self, image, detail="balanced", progress=None):
        calls.append(1)
        if len(calls) == 1:
            raise oom
        return real_generate(self, image, detail, progress)
    monkeypatch.setattr(sam, "generate", flaky)
    job = _create(client)
    assert job["status"] == "ready" and len(calls) == 2

    calls.clear()
    monkeypatch.setattr(sam, "generate", lambda self, image, detail="balanced", progress=None: (_ for _ in ()).throw(oom))
    job = client.post("/api/jobs", json={"sample": "tiny.png"}).json()
    assert job["status"] == "error"
    assert job["stages"]["segment"]["message"] == ("GPU out of memory (needed 1024.00 MiB, 459.12 MiB free); another "
                                                   "process is using the GPU - try again in a moment or lower the detail")
    assert job["error"].startswith("segment: GPU out of memory")


def test_remap_mapping_by_region_membership():
    def grp(gid, rids):
        return ColorGroup(id=gid, name="g", albedo_lab=(50.0, 0.0, 0.0), albedo_hex="#777777", area=1,
                          area_frac=0.1, region_ids=list(rids), hue_family="neutral")

    def reg(rid, area):
        return Region(id=rid, area=area, bbox=(0, 0, 1, 1), albedo_lab=(50.0, 0.0, 0.0), albedo_hex="#777777",
                      group_id=0, touches_border=False, source="sam", confidence=1.0)

    regions = [reg(0, 100), reg(1, 30), reg(2, 30), reg(3, 5)]
    old = [grp(0, [0, 3]), grp(1, [1]), grp(2, [2])]
    mapping = {"0": "#ff0000", "1": "#00ff00", "2": None}
    # regroup renumbers: old 1+2 become new 0 (60 px), old 0 becomes new 1 (105 px)... ids swap
    new = [grp(0, [1, 2]), grp(1, [0, 3])]
    assert pipeline.remap_mapping(mapping, old, new, regions) == {"0": "#00ff00", "1": "#ff0000"}
    # a group whose regions scatter without an area majority loses its color (region 0 goes
    # to new 0 = 100/105 -> majority, so it keeps it); region 3 alone in new 2 gets nothing
    new = [grp(0, [0]), grp(1, [1, 2]), grp(2, [3])]
    assert pipeline.remap_mapping(mapping, old, new, regions) == {"0": "#ff0000", "1": "#00ff00"}
    # 50/50 by area -> dropped; unknown old ids ignored; empty mapping stays empty
    old2 = [grp(0, [1, 2])]
    assert pipeline.remap_mapping({"0": "#ff0000", "9": "#000000"}, old2, [grp(0, [1]), grp(1, [2])], regions) == {}
    assert pipeline.remap_mapping({}, old, new, regions) == {}


def test_concurrent_saves_never_500(client, env):
    """PATCH /groups, PUT /state and the worker's set_stage all persist job.json; they
    used to collide on one shared temp file (FileNotFoundError -> 500)."""
    import threading
    job = _create(client)
    jid = job["id"]
    j = env["registry"].get(jid)
    codes: list[int] = []
    lock = threading.Lock()

    def worker(i: int):
        out = []
        for k in range(4):
            if (i + k) % 3 == 0:
                r = client.patch(f"/api/jobs/{jid}/groups/{k % 3}", json={"name": f"n{i}-{k}"})
            elif (i + k) % 3 == 1:
                r = client.put(f"/api/jobs/{jid}/state", json={"mapping": {"0": "#ff0000"}, "palette_id": f"p{i}"})
            else:
                j.set_stage("groups", "running", 0.5, f"tick {i}")   # the worker racing the API
                j.set_stage("groups", "done")
                r = client.get(f"/api/jobs/{jid}")
            out.append(r.status_code)
        with lock:
            codes.extend(out)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(codes) == 64 and all(c == 200 for c in codes), codes
    with open(os.path.join(config.JOBS_DIR, jid, "job.json")) as f:
        assert json.load(f)["id"] == jid          # never truncated / half-written
    assert not [f for f in os.listdir(os.path.join(config.JOBS_DIR, jid)) if f.endswith(".tmp")]


def test_delete_during_analysis_leaves_nothing_behind(client, env, monkeypatch):
    """DELETE while a stage runs: the worker must not write job.json back into the
    removed directory (it came back as a broken 'analyzing' job on restart)."""
    registry = env["registry"]
    sam = sys.modules["recolor.segmentation.sam_masks"].SamMasker
    real_generate = sam.generate
    deleted: list[str] = []

    def generate_and_delete(self, image, detail="balanced", progress=None):
        jid = next(iter(registry._jobs))
        pipeline.invalidate(registry.get(jid))        # exactly what DELETE /api/jobs/{id} does
        assert registry.delete(jid) is True
        deleted.append(jid)
        return real_generate(self, image, detail, progress)   # the stage keeps going after the delete

    monkeypatch.setattr(sam, "generate", generate_and_delete)
    r = client.post("/api/jobs", json={"sample": "tiny.png"})
    assert r.status_code == 201
    jid = deleted[0]
    assert r.json()["id"] == jid
    assert not os.path.exists(os.path.join(config.JOBS_DIR, jid)), os.listdir(os.path.join(config.JOBS_DIR, jid))
    assert registry.get(jid) is None and client.get(f"/api/jobs/{jid}").status_code == 404
    assert client.get("/api/jobs").json() == []
    # a fresh registry over the same directory sees nothing to resume
    fresh = jobs.JobRegistry(registry.root)
    assert fresh.list() == []
    # the deleted job object itself is inert: saving is a no-op and never recreates the directory
    j = registry._jobs.get(jid)
    assert j is None
    stale = jobs.Job(jid, os.path.join(config.JOBS_DIR, jid), jobs.Job.new_meta(jid, "x.jpg",
                     __import__("recolor.types", fromlist=["ImageInfo"]).ImageInfo(1, 1, 1, 1, 1, 1),
                     __import__("recolor.types", fromlist=["AnalysisOptions"]).AnalysisOptions()))
    stale.deleted = True
    stale.set_status("ready")
    stale.set_stage("ingest", "done")
    assert not os.path.exists(os.path.join(config.JOBS_DIR, jid))


def test_error_releases_pending_image(client, env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("nope")
    monkeypatch.setattr(sys.modules["recolor.segmentation.sam_masks"].SamMasker, "generate", boom)
    job = client.post("/api/jobs", json={"sample": "tiny.png"}).json()
    assert job["status"] == "error"
    assert env["registry"].get(job["id"]).pending_image is None


def test_full_export_matches_preview_intrinsic(client, env, monkeypatch):
    """A full-res export only uses a fresh decomposition when it ran with the same
    method as the preview; a fallback (OOM -> heuristic) upsamples the preview's layers."""
    job = _create(client)
    jid = job["id"]
    assert job["intrinsic_method"] == "heuristic"
    r = client.post(f"/api/jobs/{jid}/export", json={"mapping": {}, "quality": "full", "format": "png"})
    assert r.status_code == 200 and r.json()["intrinsic"] == "heuristic"
    r = client.post(f"/api/jobs/{jid}/export", json={"mapping": {}, "quality": "work", "format": "png"})
    assert r.status_code == 200 and r.json()["intrinsic"] == "work"

    # pretend the preview was Careaga; the fake model "falls back" to the heuristic
    j = env["registry"].get(jid)
    j.update(intrinsic_method="careaga")
    monkeypatch.setattr(pipeline, "_fullres_intrinsic_fits", lambda n: True)
    intr = sys.modules["recolor.intrinsic"]
    real = intr.decompose
    seen = []

    def fallback(image, method="auto", progress=None):
        seen.append(method)
        res = real(image, method="heuristic", progress=progress)
        return res
    monkeypatch.setattr(intr, "decompose", fallback)
    r = client.post(f"/api/jobs/{jid}/export", json={"mapping": {"0": "#00ff00"}, "quality": "full", "format": "png"})
    assert r.status_code == 200, r.text
    assert seen == ["careaga"] and r.json()["intrinsic"] == "upsampled"
    out = imageio.load_image(client.get(r.json()["url"]).content)
    assert out[10, 10].tolist() == [0, 255, 0]
    original = imageio.load_image(client.get(f"/api/jobs/{jid}/layers/original").content)
    # unmapped half intact (column 32 is the region boundary the guided label refinement may snap)
    assert np.abs(out[:, 34:].astype(int) - original[:, 34:].astype(int)).max() <= 2
    r = client.post(f"/api/jobs/{jid}/export", json={"mapping": {}, "quality": "full", "format": "png"})
    assert r.json()["intrinsic"] == "upsampled"
    ident = imageio.load_image(client.get(r.json()["url"]).content)
    assert np.abs(ident.astype(int) - original.astype(int)).max() <= 2   # residual recomputed: identity holds

    # consistent method -> the full-res decomposition is used
    monkeypatch.setattr(intr, "decompose", real)
    r = client.post(f"/api/jobs/{jid}/export", json={"mapping": {}, "quality": "full", "format": "png"})
    assert r.status_code == 200 and r.json()["intrinsic"] == "careaga"
    # no VRAM for the model pass -> not even attempted
    monkeypatch.setattr(pipeline, "_fullres_intrinsic_fits", lambda n: False)
    monkeypatch.setattr(intr, "decompose", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    r = client.post(f"/api/jobs/{jid}/export", json={"mapping": {}, "quality": "full", "format": "png"})
    assert r.status_code == 200 and r.json()["intrinsic"] == "upsampled"


def test_dot_segments_are_404_not_index(client):
    # (plain `/a/../b` is normalized by the HTTP client before it is sent; the encoded
    # spellings reach the server verbatim and are the ones StaticFiles turned into index.html)
    for path in ("/api/palettes/..%2F..", "/api/palettes/%2e%2e", "/api/jobs/..%2F..%2Fetc%2Fpasswd",
                 "/api/samples/..%2FMANIFEST.json", "/%2e%2e/", "/%2e"):
        r = client.get(path)
        assert r.status_code == 404, (path, r.status_code)
        assert r.json()["error"] == "not_found"
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/samples/tiny.png").status_code == 200
