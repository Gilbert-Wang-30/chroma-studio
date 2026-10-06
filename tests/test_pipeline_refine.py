"""The pipeline's groups stage with the real grouping and refinement code (no models:
ViTMatte is reported unavailable, so the colour guided-filter fallback runs), and the
group edits that must keep the engine masks consistent."""
from __future__ import annotations

import numpy as np
import pytest

from recolor import config, imageio, jobs, pipeline
from recolor.segmentation import matting
from recolor.types import AnalysisOptions

H, W = 80, 120


def _scene():
    """Red paint in two close tones with white lettering and a dull dark red part (another
    material in the paint's hue) on a grey backdrop."""
    labels = np.full((H, W), 1, np.int32)
    labels[5:75, 5:100] = 0
    labels[58:72, 76:92] = 2
    lab = np.zeros((H, W, 3), np.float32)
    lab[...] = (60.0, 0.0, 0.0)
    lab[labels == 0] = (45.0, 60.0, 45.0)
    lab[5:75, 70:100] = (48.0, 52.0, 50.0)                         # a second, close tone: one group
    lab[20:32, 20:50] = (90.0, 0.0, 0.0)
    lab[labels == 2] = (30.0, 20.0, 15.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    shading = np.full((H, W, 3), 0.9, np.float32)
    work = imageio.to_uint8(imageio.linear_to_srgb(albedo * shading))
    info = [{"id": i, "source": "sam", "confidence": 0.9} for i in range(3)]
    return work, albedo, shading, labels, info


@pytest.fixture
def job(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", str(data))
    monkeypatch.setattr(config, "JOBS_DIR", str(data / "jobs"))
    monkeypatch.setattr(config, "CACHE_DIR", str(data / "cache"))
    config.ensure_dirs()
    registry = jobs.JobRegistry(str(data / "jobs"))
    monkeypatch.setattr(jobs, "registry", registry)
    monkeypatch.setattr(pipeline, "registry", registry)
    monkeypatch.setattr(matting, "_runner", None)
    monkeypatch.setattr(matting, "_state", "unavailable")          # no transformers: guided fallback
    pipeline._layers_cache.clear()
    pipeline._renderer_cache.clear()
    work, albedo, shading, labels, info = _scene()
    j = registry.create(work, "scene.png", AnalysisOptions())
    imageio.save_image(j.path("work.png"), work)
    imageio.save_f16(j.path("albedo.npy"), albedo)
    imageio.save_f16(j.path("shading.npy"), shading)
    imageio.save_f16(j.path("residual.npy"), albedo * shading * 0.0)
    ctx = {"work": work, "albedo": albedo, "labels": labels, "region_info": info}
    j.set_stage("groups", "running", 0.0)
    pipeline._stage_groups(j, ctx)
    j.set_status("ready")
    yield j
    pipeline._layers_cache.clear()
    pipeline._renderer_cache.clear()


def test_groups_stage_writes_a_refined_partition_and_the_masks(job):
    labels = np.load(job.path("labels.npy"))
    gm = np.load(job.path("group_map.npy"))
    islands = np.load(job.path("islands.npy"))
    protect = np.load(job.path("protect.npy"))
    assert labels.dtype == np.int32 and labels.min() == 0 and gm.dtype == np.int32 and gm.min() == 0
    assert islands.dtype == bool and islands[20:32, 20:50].all() and islands.sum() == 360
    assert protect.dtype == bool and protect.shape == labels.shape
    ids = pipeline.decode_region_ids(imageio.load_image(job.path("ids", "regions.png")))
    assert np.array_equal(ids, labels)                               # the UI's id map matches the new labels
    groups = job.groups()
    assert sum(g.locked for g in groups) == 1                        # the dull dark red part
    assert "guided filter" in job.meta["stages"]["groups"]["message"]
    assert len(job.meta["groups"]) == len(groups) and job.meta["regions_count"] == int(labels.max()) + 1
    layers = pipeline.load_layers(job)
    assert np.array_equal(layers["islands"], islands) and np.array_equal(layers["protect"], protect)


def test_regroup_reapplies_the_rules_and_keeps_the_islands(job):
    islands = np.load(job.path("islands.npy"))
    pipeline.apply_group_edit(job, "regroup", {"delta_e": 4.0})      # finer groups, fresh flags
    pipeline.apply_group_edit(job, "regroup", {"delta_e": 10.0})
    assert sum(g.locked for g in job.groups()) == 1                  # the analysis rules ran again
    assert np.array_equal(np.load(job.path("islands.npy")), islands)
    gm = np.load(job.path("group_map.npy"))
    assert np.load(job.path("protect.npy")).shape == gm.shape


def test_regroup_keeps_the_users_own_lock_and_background_choices(job):
    """A regroup builds new groups with the analysis's automatic flags; the user's own
    choices come back by region membership, like the mapping (a bracket locked by hand came
    back unlocked while its mapping was carried over), and the protect mask follows them."""
    from recolor.segmentation import refine
    auto = next(g for g in job.groups() if g.locked)                 # the material the analysis locked
    free = next(g for g in job.groups() if not g.locked and not g.is_background)
    pipeline.apply_group_edit(job, "update", {"group_id": auto.id, "locked": False})
    pipeline.apply_group_edit(job, "update", {"group_id": free.id, "locked": True})
    pipeline.apply_group_edit(job, "regroup", {"delta_e": job.options.delta_e})
    by_regions = {tuple(g.region_ids): g for g in job.groups()}
    assert by_regions[tuple(auto.region_ids)].locked is False
    assert by_regions[tuple(free.region_ids)].locked is True
    layers = pipeline.load_layers(job)
    expect = refine.protect_mask(imageio.load_image(job.path("work.png")), layers["labels"], layers["group_map"],
                                 job.groups(), layers["islands"])
    assert np.array_equal(np.load(job.path("protect.npy")), expect)
    # a flag the user never touched is the analysis's: background stays where it was
    assert [g.is_background for g in job.groups()] == [by_regions[tuple(g.region_ids)].is_background
                                                        for g in job.groups()]


def test_apply_user_flags_needs_a_majority_of_the_group():
    from recolor.types import ColorGroup, Region

    def reg(rid, area):
        return Region(id=rid, area=area, bbox=(0, 0, 1, 1), albedo_lab=(50.0, 0.0, 0.0), albedo_hex="#777777",
                      group_id=0, touches_border=False, source="sam", confidence=1.0)

    def grp(gid, rids, locked=False):
        return ColorGroup(id=gid, name="g", albedo_lab=(50.0, 0.0, 0.0), albedo_hex="#777777", area=1,
                          area_frac=0.1, region_ids=list(rids), hue_family="neutral", locked=locked)

    regions = [reg(0, 100), reg(1, 40), reg(2, 70), reg(3, 10)]
    groups = [grp(0, [0, 1]), grp(1, [2, 3], locked=True)]
    flags = {"locked": {"1": True, "2": False}, "is_background": {}}
    assert pipeline.apply_user_flags(groups, regions, flags) is True
    assert groups[0].locked is False                                 # 40 of 140 px: not the user's group
    assert groups[1].locked is False                                 # 70 of 80 px were unlocked by hand
    assert pipeline.apply_user_flags(groups, regions, flags) is False


def test_pixel_level_split_persists_the_new_label_map(job):
    """Splitting a one-region group cuts the region at the pixel level; the new label map
    must reach labels.npy and the UI's id map (it used to live only in the dropped cache)."""
    paint = max(job.groups(), key=lambda g: g.area if not g.is_background else -1)
    before = np.load(job.path("labels.npy"))
    pipeline.apply_group_edit(job, "split", {"group_id": paint.id, "k": 2})
    after = np.load(job.path("labels.npy"))
    regions = pipeline._load_regions(job)
    assert int(after.max()) == int(before.max()) + 1                 # a region was cut in two
    assert int(after.max()) + 1 == len(regions)                      # labels.npy matches regions.json
    ids = pipeline.decode_region_ids(imageio.load_image(job.path("ids", "regions.png")))
    assert np.array_equal(ids, after)
    assert np.array_equal(pipeline.load_layers(job)["labels"], after)


# ------------------------------------------------------------------ a regroup reproduces the analysis

def _greedy_snap(image, labels, group_map, groups, protect=None, progress=None):
    """A boundary snap that lets the dull dark red part (region 2) grow deep into the grey
    backdrop, so that its snapped median is grey: clustering the snapped regions again
    would regroup it (the tan undertray the BMW lost to the backdrop)."""
    out = labels.copy()
    out[40:78, 100:120] = labels[65, 84]
    return out, "vitmatte"


@pytest.fixture
def greedy_job(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", str(data))
    monkeypatch.setattr(config, "JOBS_DIR", str(data / "jobs"))
    monkeypatch.setattr(config, "CACHE_DIR", str(data / "cache"))
    config.ensure_dirs()
    registry = jobs.JobRegistry(str(data / "jobs"))
    monkeypatch.setattr(jobs, "registry", registry)
    monkeypatch.setattr(pipeline, "registry", registry)
    monkeypatch.setattr(matting, "snap_labels", _greedy_snap)
    pipeline._layers_cache.clear()
    pipeline._renderer_cache.clear()
    work, albedo, shading, labels, info = _scene()
    j = registry.create(work, "scene.png", AnalysisOptions())
    imageio.save_image(j.path("work.png"), work)
    imageio.save_f16(j.path("albedo.npy"), albedo)
    imageio.save_f16(j.path("shading.npy"), shading)
    imageio.save_f16(j.path("residual.npy"), albedo * shading * 0.0)
    ctx = {"work": work, "albedo": imageio.load_f16(j.path("albedo.npy")), "labels": labels, "region_info": info}
    j.set_stage("groups", "running", 0.0)
    pipeline._stage_groups(j, ctx)
    j.set_status("ready")
    yield j
    pipeline._layers_cache.clear()
    pipeline._renderer_cache.clear()


def _state(job):
    groups = [(g.id, g.name, g.area, tuple(g.region_ids), g.locked, g.is_background) for g in job.groups()]
    return groups, np.load(job.path("group_map.npy")), np.load(job.path("protect.npy"))


def test_regroup_with_the_jobs_own_options_gives_back_the_analysis(greedy_job):
    job = greedy_job
    seed = np.load(job.path(pipeline.SEED_FILE))
    assert seed["labels"].shape == (H, W) and len(seed["origin"]) == int(np.load(job.path("labels.npy")).max()) + 1
    groups, gm, protect = _state(job)
    # clustering the snapped regions again (the old regroup) would group differently here
    from recolor.segmentation import grouping
    layers = pipeline.load_layers(job)
    _, plain, _ = grouping.regroup(pipeline._load_regions(job), layers["labels"], layers["albedo"], None, 10.0)
    assert [tuple(g.region_ids) for g in plain] != [g[3] for g in groups]
    opts = job.options
    pipeline.apply_group_edit(job, "regroup", {"max_groups": opts.max_groups, "delta_e": opts.delta_e})
    g2, gm2, prot2 = _state(job)
    assert g2 == groups and np.array_equal(gm2, gm) and np.array_equal(prot2, protect)
    pipeline.apply_group_edit(job, "regroup", {"max_groups": 2})     # other options still work
    assert len(job.groups()) <= 3
    pipeline.apply_group_edit(job, "regroup", {"max_groups": None, "delta_e": opts.delta_e})
    assert _state(job)[0] == groups                                  # and the default comes back


def test_regroup_after_a_pixel_level_split_keeps_the_cut(greedy_job):
    job = greedy_job
    paint = max((g for g in job.groups() if not g.is_background and not g.locked), key=lambda g: g.area)
    before = int(np.load(job.path("labels.npy")).max()) + 1
    pipeline.apply_group_edit(job, "split", {"group_id": paint.id, "k": 2})
    labels = np.load(job.path("labels.npy"))
    assert int(labels.max()) + 1 > before
    pipeline.apply_group_edit(job, "regroup", {"delta_e": job.options.delta_e})
    regions = pipeline._load_regions(job)
    assert len(regions) == int(labels.max()) + 1                     # the new pieces are placed, by colour
    gm = np.load(job.path("group_map.npy"))
    assert gm.shape == labels.shape and gm.min() >= 0


def test_lock_toggle_does_not_wait_for_the_gpu(job):
    """The protect mask a lock toggle recomputes is CPU work: it must not queue behind an
    analysis stage or a full-resolution export holding the GPU lock."""
    import threading
    g = next(g for g in job.groups() if not g.is_background)
    done = threading.Event()
    with pipeline.gpu_lock:                                          # "an analysis stage is running"
        t = threading.Thread(target=lambda: (pipeline.apply_group_edit(job, "update", {"group_id": g.id, "locked": True}),
                                             done.set()), daemon=True)
        # RLock: another thread cannot enter while this one holds it
        t.start()
        assert done.wait(10.0), "the lock toggle waited for gpu_lock"
    t.join(5.0)
    assert next(x for x in job.groups() if x.id == g.id).locked


def test_lock_toggle_keeps_the_renderer(job):
    renderer = pipeline.get_renderer(job)
    g = next(g for g in job.groups() if not g.is_background)
    pipeline.apply_group_edit(job, "update", {"group_id": g.id, "locked": True})
    assert pipeline.get_renderer(job) is renderer                    # flags updated in place, no rebuild
    assert next(x for x in renderer.groups if x.id == g.id).locked
    assert np.array_equal(pipeline.load_layers(job)["protect"], np.load(job.path("protect.npy")))
    pipeline.apply_group_edit(job, "merge", {"group_ids": [0, 1]})
    assert pipeline.get_renderer(job) is not renderer                # a merge rebuilds it


def test_part_recovery_failure_keeps_the_partition(monkeypatch):
    class Boom:
        @staticmethod
        def recover_parts(*a, **k):
            raise RuntimeError("prompting failed")

    class Masker:
        def prompt_parts(self, image, points):
            return []

    monkeypatch.setattr(pipeline, "_hierarchy", lambda: Boom)
    monkeypatch.setattr(pipeline, "_sam_masks", lambda: type("M", (), {"SamMasker": Masker}))
    labels = np.zeros((8, 8), np.int32)
    labels[:, 4:] = 1
    info = [{"id": 0}, {"id": 1}]
    job = type("J", (), {"id": "x", "set_stage": lambda self, *a, **k: None})()
    out, out_info, n = pipeline._recover_parts(job, {"work": np.zeros((8, 8, 3), np.uint8), "albedo": np.zeros((8, 8, 3), np.float32)},
                                               labels, info)
    assert n == 0 and out is labels and out_info is info


# ------------------------------------------------------------------ concurrent edits

def _slow_refine_after_edit(monkeypatch, delay=0.15):
    """refine_after_edit (the ~100 ms protect recompute of a lock toggle) made slower, so
    two edits overlap for sure."""
    from recolor.segmentation import refine
    import time as _time
    real = refine.refine_after_edit

    def slow(*a, **k):
        _time.sleep(delay)
        return real(*a, **k)

    monkeypatch.setattr(refine, "refine_after_edit", slow)


def test_two_concurrent_lock_toggles_are_both_kept(job, monkeypatch):
    import threading
    _slow_refine_after_edit(monkeypatch)
    a, b = [g for g in job.groups() if not g.locked][:2]
    errors = []

    def toggle(g):
        try:
            pipeline.apply_group_edit(job, "update", {"group_id": g.id, "locked": True})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=toggle, args=(g,)) for g in (a, b)]
    threads[0].start()
    import time
    time.sleep(0.02)                                                 # the second click, 20 ms later
    threads[1].start()
    for t in threads:
        t.join(10.0)
    assert not errors
    locked = {g.id for g in job.groups() if g.locked}
    assert {a.id, b.id} <= locked                                    # neither toggle was lost


def _consistent(job):
    """job.json, group_map.npy, regions.json and protect.npy describe one grouping."""
    from recolor.segmentation import refine
    groups = job.groups()
    gm = np.load(job.path("group_map.npy"))
    regions = pipeline._load_regions(job)
    labels = np.load(job.path("labels.npy"))
    assert int(gm.max()) + 1 == len(groups)
    assert {r.group_id for r in regions} == {g.id for g in groups}
    for g in groups:
        assert set(g.region_ids) == {r.id for r in regions if r.group_id == g.id}
        assert np.array_equal(gm[np.isin(labels, g.region_ids)], np.full(int(np.isin(labels, g.region_ids).sum()), g.id))
    expect = refine.protect_mask(imageio.load_image(job.path("work.png")), labels, gm, groups,
                                 np.load(job.path("islands.npy")))
    assert np.array_equal(np.load(job.path("protect.npy")), expect)


def test_a_lock_toggle_racing_a_merge_leaves_one_consistent_grouping(job, monkeypatch):
    import threading, time
    _slow_refine_after_edit(monkeypatch)
    n0 = len(job.groups())
    target = next(g for g in job.groups() if not g.locked and not g.is_background)
    t1 = threading.Thread(target=pipeline.apply_group_edit, args=(job, "update", {"group_id": target.id, "locked": True}))
    t2 = threading.Thread(target=pipeline.apply_group_edit, args=(job, "merge", {"group_ids": [0, 1]}))
    t1.start()
    time.sleep(0.02)                                                 # click lock, then press M
    t2.start()
    t1.join(10.0)
    t2.join(10.0)
    assert len(job.groups()) == n0 - 1                               # the merge was not written over
    _consistent(job)
    merged_regions = set(target.region_ids)
    assert any(g.locked and merged_regions <= set(g.region_ids) for g in job.groups())   # nor the lock


def test_a_renderer_built_during_an_edit_is_not_cached_with_the_old_groups(job, monkeypatch):
    """A preview that starts building its renderer just before a merge writes the new
    grouping must not keep a renderer of the old groups in the cache."""
    import threading
    from recolor import engine
    real = engine.Renderer
    built = []

    class Racing(real):
        def __init__(self, albedo, shading, residual, group_map, groups, **kw):
            if not built:                                            # the first build: a merge lands meanwhile
                t = threading.Thread(target=pipeline.apply_group_edit, args=(job, "merge", {"group_ids": [0, 1]}))
                t.start()
                t.join(10.0)
            built.append(len(groups))
            super().__init__(albedo, shading, residual, group_map, groups, **kw)

    monkeypatch.setattr(engine, "Renderer", Racing)
    n0 = len(job.groups())
    r = pipeline.get_renderer(job)
    assert len(job.groups()) == n0 - 1
    cached = pipeline.get_renderer(job)
    assert len(cached.groups) == n0 - 1                              # the cache holds the merged grouping
    assert built[0] == n0 and built[-1] == n0 - 1                    # the stale build was redone
    assert len(r.groups) == n0 - 1


# ------------------------------------------------------------------ refinement is an enhancement

def test_a_refinement_failure_keeps_the_plain_grouping(tmp_path, monkeypatch):
    from recolor.segmentation import refine
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", str(data))
    monkeypatch.setattr(config, "JOBS_DIR", str(data / "jobs"))
    monkeypatch.setattr(config, "CACHE_DIR", str(data / "cache"))
    config.ensure_dirs()
    registry = jobs.JobRegistry(str(data / "jobs"))
    monkeypatch.setattr(pipeline, "registry", registry)
    work, albedo, shading, labels, info = _scene()
    j = registry.create(work, "scene.png", AnalysisOptions())
    imageio.save_image(j.path("work.png"), work)
    for name in ("islands.npy", "protect.npy"):                      # stale masks of an earlier analysis
        np.save(j.path(name), np.zeros((H, W), bool))

    def boom(*a, **k):
        raise ValueError("an edge case in a heuristic")

    monkeypatch.setattr(refine, "refine_groups", boom)
    ctx = {"work": work, "albedo": albedo, "labels": labels, "region_info": info}
    j.set_stage("groups", "running", 0.0)
    pipeline._stage_groups(j, ctx)
    assert len(j.groups()) >= 2 and np.load(j.path("group_map.npy")).shape == (H, W)
    assert not pipeline._is_refined(j) and not (data / "jobs" / j.id / pipeline.SEED_FILE).exists()
    assert "refinement skipped" in j.meta["stages"]["groups"]["message"]

    class OutOfMemoryError(RuntimeError):
        pass

    def oom(*a, **k):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")

    monkeypatch.setattr(refine, "refine_groups", oom)                # the stage-level retry handles an OOM
    with pytest.raises(OutOfMemoryError):
        pipeline._stage_groups(j, ctx)


def test_the_groups_message_adds_up_with_the_regions_stage():
    msg = pipeline.groups_message(10, 73, 72, {"decal_regions": 1, "snap": "vitmatte", "decal_px": 222,
                                               "locked": [5, 6, 7]})
    assert msg.startswith("10 color groups from 73 regions (72 + 1 carved out)")
    assert "3 other materials locked" in msg
    assert pipeline.groups_message(4, 20, 21, {}) == "4 color groups from 20 regions (1 merged away at the edges)"
    assert pipeline.groups_message(4, 20, 20, {}) == "4 color groups from 20 regions"


# ------------------------------------------------------------------ this round's carry-overs

def test_apply_user_flags_tie_keeps_the_automatic_flag():
    from recolor.types import ColorGroup, Region

    def reg(rid, area):
        return Region(id=rid, area=area, bbox=(0, 0, 1, 1), albedo_lab=(50.0, 0.0, 0.0), albedo_hex="#777777",
                      group_id=0, touches_border=False, source="sam", confidence=1.0)

    regions = [reg(0, 50), reg(1, 50)]
    g = ColorGroup(id=0, name="g", albedo_lab=(50.0, 0.0, 0.0), albedo_hex="#777777", area=100, area_frac=0.1,
                   region_ids=[0, 1], hue_family="neutral", locked=False, is_background=True)
    flags = {"locked": {"0": True, "1": False}, "is_background": {"0": False, "1": True}}
    assert pipeline.apply_user_flags([g], regions, flags) is False       # 50 : 50 on both flags: nothing changes
    assert g.locked is False and g.is_background is True


def test_snapshot_hands_out_a_copy_of_the_layers(job):
    """A later lock toggle replaces the cached protect mask in place; a snapshot taken
    before it must keep the mask it was taken with."""
    (layers, groups), _ = pipeline._snapshot(job, lambda layers, groups: (layers, groups))
    before = layers["protect"]
    g = next(g for g in job.groups() if not g.is_background and not g.locked)
    pipeline.apply_group_edit(job, "update", {"group_id": g.id, "locked": True})
    assert layers["protect"] is before
    cached = pipeline.load_layers(job)
    assert cached is not layers                                         # the snapshot is not the cache's own dict


def test_a_users_background_choice_survives_a_regroup(job):
    free = next(g for g in job.groups() if not g.is_background and not g.locked)
    pipeline.apply_group_edit(job, "update", {"group_id": free.id, "is_background": True})
    pipeline.apply_group_edit(job, "regroup", {"delta_e": job.options.delta_e})
    by_regions = {tuple(g.region_ids): g for g in job.groups()}
    assert by_regions[tuple(free.region_ids)].is_background is True
    pipeline.apply_group_edit(job, "update", {"group_id": by_regions[tuple(free.region_ids)].id, "is_background": False})
    pipeline.apply_group_edit(job, "regroup", {"delta_e": job.options.delta_e})
    by_regions = {tuple(g.region_ids): g for g in job.groups()}
    assert by_regions[tuple(free.region_ids)].is_background is False


@pytest.fixture
def matte_job(tmp_path, monkeypatch):
    """The groups stage on a scene whose regions stage decided the backdrop with the matte
    rule: the grey surround is backdrop (kind 2), the paint and the dark part are object."""
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", str(data))
    monkeypatch.setattr(config, "JOBS_DIR", str(data / "jobs"))
    monkeypatch.setattr(config, "CACHE_DIR", str(data / "cache"))
    config.ensure_dirs()
    registry = jobs.JobRegistry(str(data / "jobs"))
    monkeypatch.setattr(jobs, "registry", registry)
    monkeypatch.setattr(pipeline, "registry", registry)
    monkeypatch.setattr(matting, "_runner", None)
    monkeypatch.setattr(matting, "_state", "unavailable")
    pipeline._layers_cache.clear()
    pipeline._renderer_cache.clear()
    work, albedo, shading, labels, info = _scene()
    info[1]["bg"] = 2
    j = registry.create(work, "scene.png", AnalysisOptions())
    imageio.save_image(j.path("work.png"), work)
    imageio.save_f16(j.path("albedo.npy"), albedo)
    imageio.save_f16(j.path("shading.npy"), shading)
    imageio.save_f16(j.path("residual.npy"), albedo * shading * 0.0)
    ctx = {"work": work, "albedo": albedo, "residual": albedo * 0.0, "labels": labels, "region_info": info}
    j.set_stage("groups", "running", 0.0)
    pipeline._stage_groups(j, ctx)
    j.set_status("ready")
    yield j
    pipeline._layers_cache.clear()
    pipeline._renderer_cache.clear()


def test_matte_backdrop_flags_reach_the_job_and_survive_a_regroup(matte_job):
    job = matte_job
    groups = job.groups()
    bg = [g for g in groups if g.is_background]
    assert len(bg) == 1 and 1 in bg[0].region_ids and all(rid == 1 for rid in bg[0].region_ids)
    seed = np.load(job.path(pipeline.SEED_FILE))
    assert "bg" in seed.files and seed["bg"].tolist() == [0, 2, 0]
    regions = pipeline._load_regions(job)
    assert [r.backdrop for r in sorted(regions, key=lambda r: r.id)][:3] == [False, True, False]
    assert "1 background group" in job.meta["stages"]["groups"]["message"]
    before = [(g.region_ids, g.is_background, g.locked) for g in job.groups()]
    pipeline.apply_group_edit(job, "regroup", {"delta_e": job.options.delta_e})
    assert [(g.region_ids, g.is_background, g.locked) for g in job.groups()] == before
    pipeline.apply_group_edit(job, "regroup", {"delta_e": 40.0})        # even one coarse cluster keeps the backdrop apart
    groups = job.groups()
    assert any(g.is_background for g in groups) and all(1 not in g.region_ids for g in groups if not g.is_background)


def test_effective_groups_lock_the_background_while_it_is_ignored(matte_job):
    job = matte_job
    assert pipeline.ignores_background(job)
    eff = pipeline.effective_groups(job)
    assert all(g.locked for g in eff if g.is_background) and not any(g.locked for g in job.groups() if g.is_background)
    renderer = pipeline.get_renderer(job)
    assert all(g.locked for g in renderer.groups if g.is_background)
    pipeline.save_state(job, {"ignore_background": False})
    assert not any(g.locked for g in pipeline.get_renderer(job).groups if g.is_background)
    assert pipeline.get_renderer(job) is renderer                       # a flag change keeps the renderer


# ------------------------------------------------------------------ the seed's sources, the ignore default, PUT /state

def test_the_seed_records_the_input_sources_and_the_regroup_hands_them_on(job, monkeypatch):
    from recolor.segmentation import refine
    seed = np.load(job.path(pipeline.SEED_FILE))
    assert "sources" in seed.files and seed["sources"].tolist() == ["sam", "sam", "sam"]
    loaded = pipeline._load_seed(job, np.load(job.path("labels.npy")).shape)
    assert loaded is not None and loaded[4] == ["sam", "sam", "sam"]
    real, seen = refine.regroup_refined, []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("sources"))
        return real(*args, **kwargs)

    monkeypatch.setattr(refine, "regroup_refined", spy)
    pipeline.apply_group_edit(job, "regroup", {"delta_e": job.options.delta_e})
    assert seen == [["sam", "sam", "sam"]]
    # a seed written before the sources were recorded still loads (sources None)
    pipeline._write_seed(job, loaded[0], loaded[1], loaded[2], loaded[3])
    assert pipeline._load_seed(job, loaded[0].shape)[4] is None


def test_ignore_background_starts_off_for_a_scene_that_is_mostly_background(matte_job, monkeypatch):
    from recolor.types import ColorGroup

    def grp(gid, frac, bg):
        return ColorGroup(id=gid, name=f"g{gid}", albedo_lab=(50.0, 0.0, 0.0), albedo_hex="#777777", area=int(frac * 1000),
                          area_frac=frac, region_ids=[gid], hue_family="neutral", is_background=bg)

    on, share = pipeline.default_ignore_background([grp(0, 0.6, True), grp(1, 0.3, False), grp(2, 0.1, False)])
    assert on and share == 0.6                                            # a product shot: the backdrop is ignored
    assert pipeline.default_ignore_background([grp(0, 0.4, False)]) == (True, 0.0)
    on, share = pipeline.default_ignore_background([grp(0, 0.5, True), grp(1, 0.28, True)] + [grp(i, 0.044, False) for i in range(2, 7)])
    assert on and abs(share - 0.78) < 1e-9                                # sneakers on a white backdrop: 78 % in two groups, still ignored
    on, share = pipeline.default_ignore_background([grp(i, 0.13, True) for i in range(6)] + [grp(i, 0.055, False) for i in range(6, 10)])
    assert not on and abs(share - 0.78) < 1e-9                            # 78 % of the image in six groups: a scene, left paintable
    on, _ = pipeline.default_ignore_background([grp(i, 0.1, i < 8) for i in range(10)])
    assert not on                                                         # 8 of 10 groups: a street scene
    on, _ = pipeline.default_ignore_background([grp(i, 0.1, i < 5) for i in range(10)])
    assert on                                                             # half the groups is still a product shot
    # the groups stage writes the default into the job and says so in its message
    job = matte_job
    assert job.meta["ignore_background"] is True
    monkeypatch.setattr(pipeline, "BG_IGNORE_MAX_GROUPS", 0.0)
    note = pipeline._apply_ignore_default(job, job.groups())
    assert job.meta["ignore_background"] is False and note["ignore_background"] is False
    msg = pipeline.groups_message(3, 3, 3, {"background": [1], **note})
    assert "background kept paintable" in msg and "% of the image" in msg


def test_save_state_applies_the_ignore_setting_as_an_edit(matte_job):
    job = matte_job
    gen = pipeline._generation(job.id)
    renderer = pipeline.get_renderer(job)
    assert all(g.locked for g in renderer.groups if g.is_background)
    pipeline.save_state(job, {"ignore_background": False, "palette_id": None})
    assert job.meta["ignore_background"] is False and job.meta["palette_id"] is None
    assert pipeline._generation(job.id) == gen + 2                       # an edit generation moved
    assert pipeline.get_renderer(job) is renderer                        # the renderer took the flags in place
    assert not any(g.locked for g in renderer.groups if g.is_background)
    gen = pipeline._generation(job.id)
    pipeline.save_state(job, {"mapping": {}})                            # other fields are a plain write
    assert pipeline._generation(job.id) == gen


def test_a_renderer_built_while_the_setting_changes_is_not_cached_with_the_old_locks(matte_job, monkeypatch):
    """A preview that starts building its renderer just before PUT /state switches the
    background off must not leave a renderer with the background locked in the cache."""
    import threading
    from recolor import engine
    job = matte_job
    real = engine.Renderer
    built = []

    class Racing(real):
        def __init__(self, albedo, shading, residual, group_map, groups, **kw):
            if not built:                                            # the first build: the switch lands meanwhile
                t = threading.Thread(target=pipeline.save_state, args=(job, {"ignore_background": False}))
                t.start()
                t.join(10.0)
            built.append([g.locked for g in groups if g.is_background])
            super().__init__(albedo, shading, residual, group_map, groups, **kw)

    monkeypatch.setattr(engine, "Renderer", Racing)
    r = pipeline.get_renderer(job)
    assert built[0] and all(built[0])                                    # the stale build saw the background locked
    assert not any(built[-1])                                            # and was redone with the new flags
    assert not any(g.locked for g in r.groups if g.is_background)
    assert pipeline.get_renderer(job) is r


# ------------------------------------------------------------------ detected parts and the junk pruning

def _part_scene():
    """A yellow bike frame on a grey backdrop with two yellow springs (detected parts, the
    frame's colour) and a colour-shifted shadow sliver under the frame's top tube: its albedo
    kept part of the shadow as a duller, darker yellow, so the material lock locks it."""
    h, w = 120, 200
    labels = np.zeros((h, w), np.int32)
    labels[15:105, 15:185] = 1
    labels[30:60, 30:45] = 2                                         # spring (left)
    labels[30:60, 150:165] = 3                                       # spring (right)
    labels[80:83, 90:110] = 4                                        # a 60 px shadow sliver
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (60.0, 0.0, 0.0)
    lab[labels == 1] = (80.0, 5.0, 75.0)
    lab[labels == 2] = (80.0, 5.0, 75.0)
    lab[labels == 3] = (80.0, 5.0, 75.0)
    lab[labels == 4] = (50.0, 4.0, 40.0)                             # the shadow leaked into the albedo
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    shading = np.full((h, w, 3), 0.9, np.float32)
    shading[labels == 4] = 0.3
    work = imageio.to_uint8(imageio.linear_to_srgb(np.clip(albedo * shading, 0, 1)))
    spring = {"source": "kind", "part_kind": "shock_spring", "part_label": "Shock spring", "part_plural": "Shock springs"}
    info = [{"id": 0, "source": "sam", "bg": 2}, {"id": 1, "source": "sam"}, dict(spring, id=2, part_instance=0),
            dict(spring, id=3, part_instance=1), {"id": 4, "source": "sam"}]
    fg = (labels > 0).astype(np.float32)
    masks = [labels == 2, labels == 3]
    return work, albedo, shading, labels, info, fg, masks


@pytest.fixture
def part_job(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", str(data))
    monkeypatch.setattr(config, "JOBS_DIR", str(data / "jobs"))
    monkeypatch.setattr(config, "CACHE_DIR", str(data / "cache"))
    config.ensure_dirs()
    registry = jobs.JobRegistry(str(data / "jobs"))
    monkeypatch.setattr(jobs, "registry", registry)
    monkeypatch.setattr(pipeline, "registry", registry)
    monkeypatch.setattr(matting, "_runner", None)
    monkeypatch.setattr(matting, "_state", "unavailable")
    pipeline._layers_cache.clear()
    pipeline._renderer_cache.clear()
    work, albedo, shading, labels, info, fg, masks = _part_scene()
    j = registry.create(work, "bike.png", AnalysisOptions())
    imageio.save_image(j.path("work.png"), work)
    imageio.save_f16(j.path("albedo.npy"), albedo)
    imageio.save_f16(j.path("shading.npy"), shading)
    imageio.save_f16(j.path("residual.npy"), albedo * 0.0)
    ctx = {"work": work, "albedo": imageio.load_f16(j.path("albedo.npy")), "shading": imageio.load_f16(j.path("shading.npy")),
           "residual": albedo * 0.0, "labels": labels, "region_info": info, "fg": fg, "part_masks": masks}
    j.set_stage("groups", "running", 0.0)
    pipeline._stage_groups(j, ctx)
    j.set_status("ready")
    yield j
    pipeline._layers_cache.clear()
    pipeline._renderer_cache.clear()


def test_the_groups_stage_names_the_parts_and_prunes_the_shadow(part_job):
    job = part_job
    groups = job.groups()
    springs = [g for g in groups if g.part]
    assert len(springs) == 1 and springs[0].name == "Shock springs" and springs[0].part_instances == 2
    assert springs[0].part_label == "Shock spring" and not springs[0].locked and not springs[0].is_background
    frame = next(g for g in groups if 1 in g.region_ids)
    assert 4 in frame.region_ids                                     # the shadow sliver joined the frame
    assert "1 part named (Shock springs)" in job.meta["stages"]["groups"]["message"]
    assert "sliver folded in" in job.meta["stages"]["groups"]["message"]
    regions = {r.id: r for r in pipeline._load_regions(job)}
    assert regions[2].part_kind == "shock_spring" and regions[3].part_instance == 1 and regions[1].part_kind == ""
    raw = job.meta["groups"][0]                                      # the API record carries the new fields
    assert {"part", "part_label", "part_plural", "part_instances", "minor", "parent"} <= set(raw)
    seed = pipeline._load_seed(job, np.load(job.path("labels.npy")).shape)
    assert seed.prune is not None and seed.prune.backdrop_crumbs is True        # a product shot
    assert seed.fg is not None and seed.fg.dtype == bool and len(seed.part_masks) == 2
    assert seed.part_tags[2]["kind"] == "shock_spring" and seed.part_tags[3]["instance"] == 1


def test_a_regroup_keeps_the_parts_and_the_pruning(part_job):
    job = part_job
    before = sorted((tuple(g.region_ids), g.name, g.part) for g in job.groups())
    pipeline.apply_group_edit(job, "regroup", {"delta_e": job.options.delta_e})
    assert sorted((tuple(g.region_ids), g.name, g.part) for g in job.groups()) == before
    pipeline.apply_group_edit(job, "regroup", {"max_groups": 2})     # a coarse regroup: the part still has its own
    assert any(g.part == "shock_spring" and set(g.region_ids) == {2, 3} for g in job.groups())


def test_split_by_instance_and_merge_back(part_job):
    job = part_job
    springs = next(g for g in job.groups() if g.part)
    frame = next(g for g in job.groups() if 1 in g.region_ids)
    with pytest.raises(pipeline.PipelineError) as e:
        pipeline.apply_group_edit(job, "split", {"group_id": frame.id, "mode": "instances"})
    assert e.value.status == 400
    with pytest.raises(pipeline.PipelineError):
        pipeline.apply_group_edit(job, "split", {"group_id": springs.id, "mode": "sideways"})
    pipeline.save_state(job, {"mapping": {str(springs.id): "#e63946"}})
    pipeline.apply_group_edit(job, "split", {"group_id": springs.id, "mode": "instances"})
    names = sorted(g.name for g in job.groups() if g.part)
    assert names == ["Shock spring (left)", "Shock spring (right)"]
    # both instances keep the part's paint
    assert sorted(job.meta["mapping"][str(g.id)] for g in job.groups() if g.part) == ["#e63946", "#e63946"]
    ids = [g.id for g in job.groups() if g.part]
    pipeline.apply_group_edit(job, "merge", {"group_ids": ids})
    assert [g.name for g in job.groups() if g.part] == ["Shock springs"]


def test_a_seed_from_before_the_pruning_regroups_as_that_analysis_did(part_job, monkeypatch):
    from recolor.segmentation import refine
    job = part_job
    seed = pipeline._load_seed(job, np.load(job.path("labels.npy")).shape)
    pipeline._write_seed(job, seed[0], seed[1], seed[2], seed[3], seed[4])        # the older seed shape
    old = pipeline._load_seed(job, seed[0].shape)
    assert old.prune is None and old.part_tags == {} and old.fg is None
    seen = []
    real = refine.regroup_refined

    def spy(*a, **kw):
        seen.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(refine, "regroup_refined", spy)
    pipeline.apply_group_edit(job, "regroup", {"delta_e": job.options.delta_e})
    assert "prune" not in seen[0] and "part_tags" not in seen[0]


def test_a_pruning_rule_added_after_the_analysis_stays_off_in_its_regroup(part_job):
    """The seed records the pruning's parameters; a field it does not have (a rule added since)
    reads as that rule's off value, so a regroup reruns the pruning the analysis ran."""
    import json as _json
    job = part_job
    seed = pipeline._load_seed(job, np.load(job.path("labels.npy")).shape)
    assert seed.prune.crumb_group > 0                                   # a fresh analysis runs the rule
    z = dict(np.load(job.path("regroup.npz")))
    d = _json.loads(str(z["prune"]))
    d.pop("crumb_group")
    z["prune"] = np.asarray(_json.dumps(d))
    np.savez_compressed(job.path("regroup.npz"), **z)
    assert pipeline._load_seed(job, seed[0].shape).prune.crumb_group == 0


def test_the_background_toggle_updates_the_minor_view(part_job):
    job = part_job
    frame = next(g for g in job.groups() if 1 in g.region_ids)
    backdrop = next(g for g in job.groups() if g.is_background)
    pipeline.apply_group_edit(job, "update", {"group_id": backdrop.id, "is_background": False})
    assert all(g.minor is False for g in job.groups())               # the backdrop is no minor row
    assert next(g for g in job.groups() if g.id == frame.id).parent == -1


def test_prune_params_merge_the_backdrop_crumbs_on_a_product_shot_only():
    from recolor.types import ColorGroup

    def grp(gid, frac, bg):
        return ColorGroup(id=gid, name=f"g{gid}", albedo_lab=(50.0, 0.0, 0.0), albedo_hex="#777777",
                          area=int(frac * 1000), area_frac=frac, region_ids=[gid], hue_family="neutral", is_background=bg)

    product = pipeline._prune_params([grp(0, 0.6, True), grp(1, 0.3, False), grp(2, 0.1, False)])
    scene = pipeline._prune_params([grp(i, 0.1, i < 8) for i in range(10)])
    assert product.backdrop_crumbs is True and scene.backdrop_crumbs is False


def test_the_regions_stage_detects_and_stamps_the_parts(tmp_path, monkeypatch):
    """`_detect_parts` with the models replaced: the caption picks the motorcycle vocabulary,
    the detector's box is prompted, the gated mask is stamped as a part region and its mask
    kept for the pruning; a detector that is unavailable leaves the partition alone."""
    import types as _types
    work, albedo, shading, labels, info, fg, masks = _part_scene()
    base = labels.copy()
    base[base >= 2] = 1                                              # the springs are not regions yet
    base_info = [{"id": 0, "source": "sam"}, {"id": 1, "source": "sam"}]
    job = _types.SimpleNamespace(id="deadbeef0000", options=AnalysisOptions(),
                                 set_stage=lambda *a, **k: None)
    spring = labels == 2

    def detect(image, phrases):
        ys, xs = np.nonzero(spring)
        return [{"box": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                 "phrase": "coil spring", "score": 0.6, "det": "owlv2"}]

    class Masker:
        def prompt_boxes(self, image, jobs):
            out = []
            for j in jobs:
                x0, y0, x1, y1 = j["crop"]
                out.append([{"mask": spring[y0:y1, x0:x1].copy(), "x0": x0, "y0": y0, "score": 0.95, "clipped": False}])
            return out

    monkeypatch.setattr(pipeline, "_partdetect", lambda: _types.SimpleNamespace(detect=detect))
    monkeypatch.setattr(pipeline, "_florence", lambda: _types.SimpleNamespace(caption=lambda img: "a yellow motorbike"))
    monkeypatch.setattr(pipeline, "_sam_masks", lambda: _types.SimpleNamespace(SamMasker=Masker))
    ctx = {"work": work, "albedo": albedo}
    fg = np.ones(labels.shape, np.float32)                           # the whole photo is the object
    out, out_info = pipeline._detect_parts(job, ctx, base, base_info, fg)
    tagged = [d for d in out_info if d.get("part_kind")]
    assert len(tagged) == 1 and tagged[0]["part_kind"] == "shock_spring"
    assert (out[spring] == tagged[0]["id"]).all() and len(ctx["part_masks"]) == 1
    kinds = pipeline._backdrop_flags(job, ctx, out, out_info, fg)
    assert kinds is not None and all(d["bg"] == 0 for d in out_info if d.get("part_kind"))
    monkeypatch.setattr(pipeline, "_partdetect", lambda: _types.SimpleNamespace(detect=lambda image, phrases: None))
    out2, info2 = pipeline._detect_parts(job, ctx, base, base_info, fg)
    assert np.array_equal(out2, base) and info2 == base_info


# ------------------------------------------------------------------ this round

def test_a_regroup_at_one_group_with_parts_keeps_the_colour_group_paintable(part_job):
    """max_groups 1: the one colour group holds the object and the backdrop; flagged background
    it would leave nothing paintable while the background is ignored."""
    job = part_job
    pipeline.apply_group_edit(job, "regroup", {"max_groups": 1})
    groups = job.groups()
    colour = [g for g in groups if not g.part]
    assert len(colour) == 1 and not colour[0].is_background
    assert any(g.part == "shock_spring" for g in groups) and not any(g.is_background for g in groups)
    assert [g.id for g in pipeline.effective_groups(job) if not g.locked]      # something is paintable


def test_a_regroup_never_prunes_the_region_the_user_locked(part_job):
    """The shadow sliver joined the frame at the analysis. Split off again and locked by the user,
    it keeps its own group through a regroup: the pruning leaves the regions the user flagged alone
    (it runs before the user's flags are re-applied, and dissolved the lock would have been lost)."""
    job = part_job
    frame = next(g for g in job.groups() if 1 in g.region_ids)
    pipeline.apply_group_edit(job, "split", {"group_id": frame.id, "k": 2})
    sliver = next(g for g in job.groups() if 4 in g.region_ids)
    assert 1 not in sliver.region_ids
    pipeline.apply_group_edit(job, "update", {"group_id": sliver.id, "locked": True})
    pipeline.apply_group_edit(job, "regroup", {"delta_e": job.options.delta_e})
    g = next(g for g in job.groups() if 4 in g.region_ids)
    assert 1 not in g.region_ids and g.locked


def test_a_lock_on_the_host_of_a_pruned_sliver_keeps_the_sliver_in_it(part_job):
    """Locking a group records a vote for every region of it, the sliver the analysis's pruning
    folded in too. A regroup must fold it in again: exempted as a region the user flagged, it came
    back as a locked junk group of its own after every lock and regroup (robot feet rim, BMW,
    Alpine), and so did the backdrop crumbs of a background group the user un-flagged."""
    job = part_job
    frame = next(g for g in job.groups() if 1 in g.region_ids)
    assert 4 in frame.region_ids
    before = sorted(tuple(g.region_ids) for g in job.groups())
    pipeline.apply_group_edit(job, "update", {"group_id": frame.id, "locked": True})
    flags = pipeline._load_user_flags(job)
    assert flags["locked"].get("4") is True                          # the sliver carries the frame's vote
    pipeline.apply_group_edit(job, "regroup", {"delta_e": job.options.delta_e})
    assert sorted(tuple(g.region_ids) for g in job.groups()) == before
    g = next(g for g in job.groups() if 1 in g.region_ids)
    assert 4 in g.region_ids and g.locked
    springs = next(g for g in job.groups() if g.part)
    pipeline.apply_group_edit(job, "update", {"group_id": springs.id, "locked": True})
    pipeline.apply_group_edit(job, "regroup", {"delta_e": job.options.delta_e})
    assert sorted(tuple(g.region_ids) for g in job.groups()) == before
    assert next(g for g in job.groups() if g.part).locked


def test_a_view_of_an_older_panel_rule_is_refreshed_in_memory_when_served(part_job):
    """A job analysed under an older Minor rule keeps stale minor flags in its record; serving
    it re-annotates the groups in memory and leaves job.json as it was."""
    import hashlib
    from recolor.segmentation import grouping
    job = part_job
    assert job.meta[pipeline.PANEL_RULE_KEY] == grouping.PANEL_RULE          # an analysis records the rule
    assert not pipeline.refresh_panel_view(job)                              # current: nothing to do
    with job.lock:
        job.meta.pop(pipeline.PANEL_RULE_KEY)
        for g in job.meta["groups"]:
            if 2 in g["region_ids"]:
                g["minor"], g["parent"] = True, 0                            # the old size-only rule's view
    job.save()
    md5 = hashlib.md5(open(job.path("job.json"), "rb").read()).hexdigest()
    assert pipeline.refresh_panel_view(job)
    assert not any(g.minor for g in job.groups())
    assert job.meta[pipeline.PANEL_RULE_KEY] == grouping.PANEL_RULE
    assert hashlib.md5(open(job.path("job.json"), "rb").read()).hexdigest() == md5   # not rewritten on open
    assert not pipeline.refresh_panel_view(job)


def test_the_minor_view_needs_a_body_colour_for_the_cast_test():
    """Two neutrals pass the lightness-normalised (a, b) test at any lightness: a white badge on
    black trim, or a silver strip beside charcoal, is not the trim's shadow."""
    from recolor.segmentation import grouping
    assert not grouping._lighting_variant((92.0, 0.5, 1.0), (12.0, 0.3, 0.5))           # white badge on black
    assert not grouping._lighting_variant((65.0, -1.0, -2.0), (35.0, 0.0, -1.0))        # silver beside charcoal
    assert grouping._lighting_variant((40.0, 0.5, 1.0), (47.0, 0.2, 0.4))                # a grey a shade darker
    assert grouping._lighting_variant((30.0, 42.0, 32.0), (48.0, 58.0, 44.0))            # the red paint in shadow


def test_a_regroup_keeps_the_names_the_user_gave(part_job):
    job = part_job
    springs = next(g for g in job.groups() if g.part)
    frame = next(g for g in job.groups() if 1 in g.region_ids)
    pipeline.apply_group_edit(job, "update", {"group_id": springs.id, "name": "Rear shocks"})
    pipeline.apply_group_edit(job, "update", {"group_id": frame.id, "name": "Frame"})
    pipeline.apply_group_edit(job, "regroup", {"delta_e": job.options.delta_e})
    names = {g.name for g in job.groups()}
    assert {"Rear shocks", "Frame"} <= names
    backdrop = next(g for g in job.groups() if g.is_background)
    pipeline.apply_group_edit(job, "update", {"group_id": backdrop.id, "name": "Wall"})
    pipeline.apply_group_edit(job, "regroup", {"max_groups": 1})          # one colour group: frame and backdrop
    names = {g.name for g in job.groups()}
    assert "Rear shocks" in names and "Frame" in names                  # the frame is most of it
    assert "Wall" not in names                                          # the backdrop is not


def test_the_fast_preset_skips_the_detected_parts(monkeypatch):
    import types as _types
    calls = []
    monkeypatch.setattr(pipeline, "_detect_parts", lambda *a: calls.append(a) or (a[2], a[3]))
    monkeypatch.setattr(pipeline, "_find_extras", lambda job, ctx: [])
    monkeypatch.setattr(pipeline, "_recover_parts", lambda job, ctx, labels, info: (labels, info, 0))
    monkeypatch.setattr(pipeline, "_matte_cut", lambda job, ctx, labels, info: (labels, info, 0, None))
    monkeypatch.setattr(pipeline, "_write_labels", lambda *a: None)
    work, albedo, shading, labels, info, fg, masks = _part_scene()
    hier = _types.SimpleNamespace(build_regions=lambda *a, **k: (labels.copy(), [dict(d) for d in info]))
    monkeypatch.setattr(pipeline, "_hierarchy", lambda: hier)
    for detail, n in (("fast", 0), ("balanced", 1)):
        job = _types.SimpleNamespace(id="f00", options=AnalysisOptions(detail=detail), set_stage=lambda *a, **k: None)
        ctx = {"work": work, "albedo": albedo, "masks": []}
        pipeline._stage_regions(job, ctx)
        assert len(calls) == n and ctx["part_masks"] == []


@pytest.mark.parametrize("broken", ["raise", "inconsistent"])
def test_detected_parts_keep_the_partition_when_the_stamping_fails(monkeypatch, broken):
    import types as _types
    work, albedo, shading, labels, info, fg, masks = _part_scene()
    base = labels.copy()
    base[base >= 2] = 1
    base_info = [{"id": 0, "source": "sam"}, {"id": 1, "source": "sam"}]
    job = _types.SimpleNamespace(id="deadbeef0001", options=AnalysisOptions(), set_stage=lambda *a, **k: None)

    def find(*a, **kw):
        return [smallparts_part(masks[0])], {"class": "motorcycle", "detections": 1, "jobs": 1}

    def stamp(labels, info, parts, lab):
        if broken == "raise":
            raise RuntimeError("stamping broke")
        return labels[:10], info, {}                                 # the wrong shape

    sp = _types.SimpleNamespace(find_kind_parts=find, stamp_parts=stamp)
    monkeypatch.setattr(pipeline, "_smallparts", lambda: sp)
    monkeypatch.setattr(pipeline, "_partdetect", lambda: _types.SimpleNamespace(detect=lambda i, p: []))
    monkeypatch.setattr(pipeline, "_florence", lambda: _types.SimpleNamespace(caption=lambda img: "a motorbike"))
    monkeypatch.setattr(pipeline, "_sam_masks", lambda: _types.SimpleNamespace(SamMasker=lambda: _types.SimpleNamespace(
        prompt_boxes=lambda image, jobs: [])))
    ctx = {"work": work, "albedo": albedo, "part_masks": [np.ones((2, 2), bool)]}
    out, out_info = pipeline._detect_parts(job, ctx, base, base_info, np.ones(base.shape, np.float32))
    assert np.array_equal(out, base) and out_info == base_info and ctx["part_masks"] == []


def smallparts_part(mask):
    from recolor.segmentation.smallparts import PartMask
    ys, xs = np.nonzero(mask)
    return PartMask("shock_spring", "Shock spring", "Shock springs", mask, 0.6, 0.95,
                    [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], "coil spring", "owlv2")


def test_detected_parts_hand_the_partition_to_the_wheel_look(monkeypatch):
    import types as _types
    work, albedo, shading, labels, info, fg, masks = _part_scene()
    seen = {}

    def find(image, albedo_, fg_, caption, detect, prompt, budget_s=None, **kw):
        seen.update(kw)
        return [], {"class": "motorcycle"}

    zoom = lambda image, crops, phrases: []                          # noqa: E731
    monkeypatch.setattr(pipeline, "_smallparts", lambda: _types.SimpleNamespace(find_kind_parts=find,
                                                                                stamp_parts=lambda *a: None))
    monkeypatch.setattr(pipeline, "_partdetect", lambda: _types.SimpleNamespace(detect=lambda i, p: [], detect_in=zoom))
    monkeypatch.setattr(pipeline, "_florence", lambda: _types.SimpleNamespace(caption=lambda img: "a motorbike"))
    monkeypatch.setattr(pipeline, "_sam_masks", lambda: _types.SimpleNamespace(SamMasker=lambda: _types.SimpleNamespace(
        prompt_boxes=lambda image, jobs: [])))
    job = _types.SimpleNamespace(id="deadbeef0002", options=AnalysisOptions(), set_stage=lambda *a, **k: None)
    ctx = {"work": work, "albedo": albedo}
    pipeline._detect_parts(job, ctx, labels, info, fg)
    assert seen["zoom"] is zoom and seen["labels"] is labels and seen["info"] is info
