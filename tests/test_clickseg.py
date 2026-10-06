"""Select part and Find part: SAM 2 prompted by clicks and boxes (a fake predictor stands in for
the model), the carve of a user part into a refined job (the real grouping, refinement and junk
pruning; ViTMatte unavailable, so the guided fallback snaps), its survival through Auto-regroup
and the pruning, its deletion through the merge endpoint, and old jobs. No models, no network."""
from __future__ import annotations

import json
import os
import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from recolor import config, imageio, jobs, pipeline
from recolor.segmentation import interactive, matting, sam_masks, userparts
from recolor.types import AnalysisOptions

H, W = 120, 200
CALIPER = (136, 73)          # a click on the blue caliper the automatic pass left in the frame's region
BOLT = (127, 77)             # its light-blue bolt
SLIVER = (64, 96)            # a sliver in the frame's colour


# ----------------------------------------------------------------------------- the fake SAM

class FakePredictor:
    """``SAM2ImagePredictor`` stand-in on flat-colour images: a positive point answers with the
    8-connected blob of its colour plus the other-coloured blobs inside it (its bolts; the blobs of
    the colour around it are holes), negative points cut their blob out; a box answers with the
    blob of the most frequent colour inside it that is not its border's; multimask gives (blob,
    object, object grown by 3 px) scored 0.8 / 0.95 / 0.5. Every call is logged."""
    log: list = []

    def set_image(self, image):
        self.image = np.ascontiguousarray(image)
        FakePredictor.log.append(("set_image", tuple(self.image.shape[:2])))
        im = self.image.astype(np.int64)
        self.flat = (im[..., 0] << 16) | (im[..., 1] << 8) | im[..., 2]

    def _blob(self, x, y):
        h, w = self.flat.shape
        xi, yi = int(np.clip(round(float(x)), 0, w - 1)), int(np.clip(round(float(y)), 0, h - 1))
        same = (self.flat == self.flat[yi, xi]).astype(np.uint8)
        _, cc = cv2.connectedComponents(same, connectivity=8)
        return cc == cc[yi, xi]

    def _object(self, blob):
        ring = cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~blob
        vals, cnt = np.unique(self.flat[ring], return_counts=True)
        surround = vals[np.argmax(cnt)] if len(vals) else None
        ys, xs = np.nonzero(blob)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        out = blob.copy()
        other = (self.flat != surround) & ~blob
        k, cc = cv2.connectedComponents(other.astype(np.uint8), connectivity=8)
        for i in range(1, k):
            m = cc == i
            yy, xx = np.nonzero(m)
            if yy.min() >= y0 and yy.max() < y1 and xx.min() >= x0 and xx.max() < x1:
                out |= m
        return out

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True):
        FakePredictor.log.append(("predict", None if point_coords is None else np.asarray(point_coords).round(2).tolist(),
                                  None if point_labels is None else np.asarray(point_labels).tolist(),
                                  None if box is None else np.asarray(box).round(2).tolist(),
                                  mask_input is not None, bool(multimask_output), tuple(self.flat.shape)))
        h, w = self.flat.shape
        pos, neg = [], []
        if point_coords is not None:
            for (x, y), lab in zip(np.asarray(point_coords).reshape(-1, 2), np.asarray(point_labels).ravel()):
                (pos if lab == 1 else neg).append((x, y))
        blob = np.zeros((h, w), bool)
        obj = np.zeros((h, w), bool)
        if pos:
            blob = self._blob(*pos[0])
            for p in pos:
                obj |= self._object(self._blob(*p))
        elif box is not None:
            x0, y0, x1, y1 = (int(round(float(v))) for v in np.asarray(box).ravel())
            win = self.flat[y0:y1, x0:x1]
            border = np.concatenate([win[0], win[-1], win[:, 0], win[:, -1]])
            bv, bc = np.unique(border, return_counts=True)
            vals, cnt = np.unique(win, return_counts=True)
            order = [v for _, v in sorted(zip(-cnt, vals)) if v != bv[np.argmax(bc)]]
            if order:
                ys, xs = np.nonzero(win == order[0])
                blob = self._blob(xs[0] + x0, ys[0] + y0)
                obj = self._object(blob)
        for p in neg:
            cut = self._blob(*p)
            blob &= ~cut
            obj &= ~cut
        grown = cv2.dilate(obj.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        if multimask_output:
            masks, scores = [blob, obj, grown], [0.8, 0.95, 0.5]
        else:
            masks, scores = [obj], [0.93]
        return (np.stack(masks).astype(np.float32), np.array(scores, np.float32),
                np.zeros((len(masks), 256, 256), np.float32))


# ----------------------------------------------------------------------------- the scene

def _scene(tank_part: bool = False, two_springs: bool = False):
    """A yellow bike frame on a grey backdrop with a detected yellow spring and a red tank; inside
    the frame's region a blue caliper the automatic pass missed (with a light-blue bolt, a 25 px
    hole through which the frame shows and a 4 px hole of noise) and a sliver of the frame's
    colour, 2 dE off it. With ``two_springs`` a second spring (instance 1, region 4) sits below
    the first."""
    labels = np.zeros((H, W), np.int32)
    labels[15:105, 15:185] = 1
    labels[30:60, 30:45] = 2
    labels[20:50, 100:170] = 3
    if two_springs:
        labels[64:92, 30:45] = 4
    lab = np.zeros((H, W, 3), np.float32)
    lab[...] = (60.0, 0.0, 0.0)
    lab[labels == 1] = (80.0, 5.0, 75.0)
    lab[labels == 2] = (80.0, 5.0, 75.0)
    lab[labels == 4] = (80.0, 5.0, 75.0)
    lab[labels == 3] = (50.0, 60.0, 45.0)
    lab[70:92, 120:152] = (45.0, 5.0, -45.0)
    lab[76:80, 126:130] = (72.0, -5.0, -25.0)
    lab[84:89, 140:145] = (80.0, 5.0, 75.0)
    lab[72:74, 146:148] = (80.0, 5.0, 75.0)
    lab[95:99, 60:70] = (78.0, 5.0, 72.0)
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    shading = np.full((H, W, 3), 0.9, np.float32)
    work = imageio.to_uint8(imageio.linear_to_srgb(np.clip(albedo * shading, 0, 1)))
    spring = {"source": "kind", "part_kind": "shock_spring", "part_label": "Shock spring", "part_plural": "Shock springs"}
    info = [{"id": 0, "source": "sam", "bg": 2}, {"id": 1, "source": "sam"}, dict(spring, id=2, part_instance=0),
            {"id": 3, "source": "sam"}]
    masks = [labels == 2]
    if tank_part:                                                    # the tank detected as a part too
        info[3] = {"id": 3, "source": "kind", "part_kind": "tank", "part_label": "Fuel tank",
                   "part_plural": "Fuel tanks", "part_instance": 0}
        masks.append(labels == 3)
    if two_springs:
        info.append(dict(spring, id=4, part_instance=1))
        masks.append(labels == 4)
    return work, albedo, shading, labels, info, (labels > 0).astype(np.float32), masks


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", str(data))
    monkeypatch.setattr(config, "JOBS_DIR", str(data / "jobs"))
    monkeypatch.setattr(config, "CACHE_DIR", str(data / "cache"))
    config.ensure_dirs()
    registry = jobs.JobRegistry(str(data / "jobs"))
    monkeypatch.setattr(jobs, "registry", registry)
    monkeypatch.setattr(pipeline, "registry", registry)
    monkeypatch.setattr(pipeline, "start_worker", lambda: None)
    monkeypatch.setattr(matting, "_runner", None)
    monkeypatch.setattr(matting, "_state", "unavailable")
    sentinel = object()

    def fake_load(self):
        if sam_masks.SamMasker._model is None:
            sam_masks.SamMasker._model = sentinel

    monkeypatch.setattr(sam_masks.SamMasker, "load", fake_load)
    monkeypatch.setattr(sam_masks.SamMasker, "_new_predictor", lambda self: FakePredictor())
    monkeypatch.setattr(sam_masks.SamMasker, "_model", sentinel)
    sam_masks.SamMasker._prompt_sessions.clear()
    FakePredictor.log.clear()
    pipeline._layers_cache.clear()
    pipeline._renderer_cache.clear()

    def make_job(name="bike.png", tank_part=False, two_springs=False):
        work, albedo, shading, labels, info, fg, masks = _scene(tank_part, two_springs)
        j = registry.create(work, name, AnalysisOptions())
        imageio.save_image(j.path("work.png"), work)
        imageio.save_f16(j.path("albedo.npy"), albedo)
        imageio.save_f16(j.path("shading.npy"), shading)
        imageio.save_f16(j.path("residual.npy"), albedo * 0.0)
        ctx = {"work": work, "albedo": imageio.load_f16(j.path("albedo.npy")),
               "shading": imageio.load_f16(j.path("shading.npy")), "residual": albedo * 0.0, "labels": labels,
               "region_info": info, "fg": fg, "part_masks": masks}
        j.set_stage("groups", "running", 0.0)
        pipeline._stage_groups(j, ctx)
        j.set_status("ready")
        return j

    yield {"make_job": make_job, "registry": registry}
    sam_masks.SamMasker._prompt_sessions.clear()
    pipeline._layers_cache.clear()
    pipeline._renderer_cache.clear()


@pytest.fixture
def client(env):
    from recolor.server.app import create_app
    with TestClient(create_app()) as c:
        yield c


def _set_images():
    return [e[1] for e in FakePredictor.log if e[0] == "set_image"]


def _decode(entry) -> np.ndarray:
    """A mask entry of the API as a full-size bool mask."""
    import base64
    import io
    from PIL import Image
    out = np.zeros((H, W), bool)
    x0, y0, x1, y1 = entry["bbox"]
    png = base64.b64decode(entry["png"].split(",", 1)[1])
    out[y0:y1, x0:x1] = np.asarray(Image.open(io.BytesIO(png)).convert("L")) > 127
    return out


def _caliper_expected(with_bolt=True):
    work = _scene()[0]
    blue = (work == work[CALIPER[1], CALIPER[0]]).all(-1)
    bolt = (work == work[BOLT[1], BOLT[0]]).all(-1)
    m = blue | bolt if with_bolt else blue
    m[72:74, 146:148] = True                         # the 4 px hole is noise: filled
    return m


# ----------------------------------------------------------------------------- pure pieces

def test_carve_keeps_ids_merges_specks_back_and_respects_holes():
    labels = np.zeros((40, 60), np.int32)
    labels[:, 30:] = 1
    labels[5:15, 5:15] = 2                              # wholly inside the part: keeps its id, joins it
    mask = np.zeros((40, 60), bool)
    mask[2:30, 2:40] = True                              # cuts regions 0 and 1
    mask[20:26, 10:16] = False                           # a 36 px hole: stays region 0
    mask[8:10, 20:22] = False                            # a 4 px hole: a speck of region 0, joins the part
    mask[35:37, 50:52] = True                            # a 4 px island: not carved
    mask[2:30, 39] = True
    labels[2:30, 40] = 1                                 # region 1 would keep a 1 px column beside the part ...
    labels[2:30, 41:] = 1
    c = userparts.carve(labels, mask)
    assert c is not None and c.labels.dtype == np.int32 and c.labels.min() == 0
    assert c.covered == [2] and c.new_id == 3 and c.part_ids == [2, 3]
    assert set(np.unique(c.labels).tolist()) == {0, 1, 2, 3}          # nothing emptied, one id added
    assert (c.labels[5:15, 5:15] == 2).all()
    assert (c.labels[20:26, 10:16] == 0).all()                       # the real hole is respected
    assert (c.labels[8:10, 20:22] == 3).all()                        # the noise hole joined the part
    assert (c.labels[35:37, 50:52] == 1).all()                       # the island merged back
    assert c.mask.sum() == int(np.isin(c.labels, c.part_ids).sum())
    untouched = ~(mask | c.mask)
    assert (c.labels[untouched] == labels[untouched]).all()          # ids stable elsewhere
    assert userparts.carve(labels, np.zeros_like(mask)) is None
    tiny = np.zeros_like(mask)
    tiny[0:2, 0:3] = True
    assert userparts.carve(labels, tiny) is None                     # under a few px: nothing to carve


def test_parse_prompt_checks_coordinates_labels_and_intent():
    p = interactive.parse_prompt({"points": [[10, 12, 1], [199.6, 0, 0]], "box": None, "multimask": True}, W, H)
    assert p.points == ((10.0, 12.0, 1), (199.0, 0.0, 0)) and p.steps == 2       # clamped onto the last pixel
    assert interactive.parse_prompt({}, W, H).empty
    for bad in ({"points": [[-5, 10, 1]]}, {"points": [[10, 400, 1]]}, {"points": [[10, 10, 2]]},
                {"points": [[10, 10, 0]]}, {"points": "x"}, {"box": [1, 1, 1.5, 9]}, {"multimask": "yes", "points": [[1, 1]]},
                {"points": [[float("nan"), 1, 1]]}, {"points": [[1, 1, 1]], "pick": 5}):
        with pytest.raises(interactive.PromptError):
            interactive.parse_prompt(bad, W, H)
    b = interactive.parse_prompt({"box": [150, 90, 110, 60], "points": [[130, 75, 0]]}, W, H)
    assert b.box == (110.0, 60.0, 150.0, 90.0) and not b.empty                     # reordered, negative + box is fine
    assert interactive.parse_prompt({"points": [[5, 5, 0], [9, 9, 1]]}, W, H).ordered_points()[0] == (9.0, 9.0, 1)


def test_crop_choice_is_grid_snapped_and_skips_large_parts():
    m = np.zeros((1024, 1536), bool)
    m[700:760, 200:250] = True
    prompt = interactive.Prompt(points=((222.0, 735.0, 1),))
    c = interactive.choose_crop(m, prompt, 1536, 1024)
    assert c is not None and all(v % interactive.CROP_GRID == 0 for v in c[:2])
    assert c[0] <= 200 and c[1] <= 700 and c[2] >= 250 and c[3] >= 760 and c[2] - c[0] >= interactive.CROP_MIN_SIDE
    big = np.zeros_like(m)
    big[100:900, 100:1000] = True
    assert interactive.choose_crop(big, prompt, 1536, 1024) is None


# ----------------------------------------------------------------------------- /segment

def test_segment_caches_the_embedding_and_refines_on_a_crop(client, env):
    job = env["make_job"]()
    r = client.post(f"/api/jobs/{job.id}/segment", json={"points": [[*CALIPER, 1]], "box": None, "multimask": True})
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["embed"] == "computed" and a["size"] == [W, H] and a["steps"] == 1 and a["pick"] == 1
    assert (_decode(a["mask"]) == _caliper_expected()).all()
    assert a["mask"]["area"] == int(_caliper_expected().sum()) and a["mask"]["refined"] is True
    assert len(a["alternatives"]) == 2 and {x["index"] for x in a["alternatives"]} == {0, 2}
    crop = a["crop"]
    assert crop is not None and crop[0] > 0 and crop[2] - crop[0] >= interactive.CROP_MIN_SIDE - 16
    shapes = _set_images()
    assert shapes[0] == (H, W) and shapes[1] == (crop[3] - crop[1], crop[2] - crop[0])
    # the crop saw the click in its own frame
    crop_calls = [e for e in FakePredictor.log if e[0] == "predict" and e[6] != (H, W)]
    assert crop_calls[0][1] == [[CALIPER[0] - crop[0], CALIPER[1] - crop[1]]]
    # a second prompt on the same part: no new embedding of the image, the crop's is reused
    FakePredictor.log.clear()
    r2 = client.post(f"/api/jobs/{job.id}/segment", json={"points": [[*CALIPER, 1], [140, 88, 1]], "multimask": True,
                                                          "pick": a["pick"], "crop": crop})
    assert r2.status_code == 200 and r2.json()["embed"] == "cached"
    assert _set_images() == []
    steps = [e for e in FakePredictor.log if e[0] == "predict" and e[6] == (H, W)]
    assert [e[5] for e in steps] == [True, False] and steps[1][4] is True     # the replay: mask input from step 1
    warm = client.post(f"/api/jobs/{job.id}/segment", json={})
    assert warm.status_code == 200 and warm.json()["warm"] is True and warm.json()["embed"] == "cached"


def test_negative_points_cut_a_piece_out(client, env):
    job = env["make_job"]()
    r = client.post(f"/api/jobs/{job.id}/segment", json={"points": [[*CALIPER, 1], [*BOLT, 0]], "multimask": True})
    assert r.status_code == 200, r.text
    m = _decode(r.json()["mask"])
    assert (m == _caliper_expected(with_bolt=False)).all() and not m[BOLT[1], BOLT[0]]
    assert r.json()["alternatives"] == []                                # two steps: one answer


def test_a_box_prompt_picks_the_candidate_that_fills_it(client, env):
    job = env["make_job"]()
    r = client.post(f"/api/jobs/{job.id}/segment", json={"points": [], "box": [114, 64, 158, 98], "multimask": True})
    assert r.status_code == 200, r.text
    assert (_decode(r.json()["mask"]) == _caliper_expected()).all()


def test_bad_prompts_and_states_are_clean_errors(client, env):
    job = env["make_job"]()
    for body in ({"points": [[500, 10, 1]]}, {"points": [[10, 10, 0]]}, {"box": [0, 0, 1, 1]}):
        r = client.post(f"/api/jobs/{job.id}/segment", json=body)
        assert r.status_code == 400 and r.json()["error"] == "bad_request", body
    assert client.post("/api/jobs/000000000000/segment", json={}).status_code == 404
    job.set_status("analyzing")
    assert client.post(f"/api/jobs/{job.id}/segment", json={"points": [[*CALIPER, 1]]}).status_code == 409


def test_segment_answers_503_while_the_gpu_is_busy(client, env, monkeypatch):
    job = env["make_job"]()
    monkeypatch.setattr(pipeline, "PROMPT_GPU_WAIT_S", 0.05)
    held, done = threading.Event(), threading.Event()

    def hold():
        with pipeline.gpu_lock:
            held.set()
            done.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    try:
        held.wait(5)
        r = client.post(f"/api/jobs/{job.id}/segment", json={"points": [[*CALIPER, 1]]})
        assert r.status_code == 503 and r.headers["Retry-After"] == str(pipeline.PROMPT_RETRY_AFTER_S)
        assert "busy" in r.json()["detail"]
        assert client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).status_code == 503
        assert client.post(f"/api/jobs/{job.id}/find", json={"text": "caliper"}).status_code == 503
    finally:
        done.set()
        t.join()
    assert client.post(f"/api/jobs/{job.id}/segment", json={"points": [[*CALIPER, 1]]}).status_code == 200


def test_the_embedding_cache_is_an_lru_freed_on_idle_unload_and_delete(client, env, monkeypatch):
    a, b, c, d = (env["make_job"](f"j{i}.png") for i in range(4))
    monkeypatch.setattr(pipeline, "_last_activity", 0.0)
    for j in (a, b, c):
        assert client.post(f"/api/jobs/{j.id}/segment", json={}).json()["embed"] == "computed"
    assert sam_masks.SamMasker.prompt_keys() == [a.id, b.id, c.id]
    assert pipeline._last_activity > 0.0                                # a prompt counts for the idle unload
    assert client.post(f"/api/jobs/{a.id}/segment", json={}).json()["embed"] == "cached"     # a is the newest now
    assert client.post(f"/api/jobs/{d.id}/segment", json={}).json()["embed"] == "computed"
    assert sam_masks.SamMasker.prompt_keys() == [c.id, a.id, d.id]                            # b was the oldest
    assert client.post(f"/api/jobs/{b.id}/segment", json={}).json()["embed"] == "computed"
    assert client.delete(f"/api/jobs/{a.id}").status_code == 200
    assert a.id not in sam_masks.SamMasker.prompt_keys()
    pipeline._release_idle_models()                                     # the idle unload drops SAM 2 ...
    assert sam_masks.SamMasker.prompt_keys() == [] and sam_masks.SamMasker._model is None
    assert client.post(f"/api/jobs/{b.id}/segment", json={}).json()["embed"] == "computed"   # ... and it comes back


# ----------------------------------------------------------------------------- /groups/from_mask

def _state(job):
    labels = np.load(job.path("labels.npy"))
    with open(job.path("regions.json")) as f:
        regions = json.load(f)
    return labels, regions


def test_from_mask_carves_a_part_group_and_keeps_everything_else(client, env):
    job = env["make_job"]()
    labels0, regions0 = _state(job)
    islands0 = np.load(job.path("islands.npy"))
    frame = next(g for g in job.groups() if 1 in g.region_ids)
    pipeline.save_state(job, {"mapping": {str(frame.id): "#1b2a57"}})
    r = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]], "multimask": True})
    assert r.status_code == 200, r.text
    out = r.json()
    labels, regions = _state(job)
    # a complete int32 partition, contiguous ids, none emptied, one added, every other pixel as it was
    assert labels.dtype == np.int32 and labels.min() == 0 and int(labels.max()) + 1 == len(regions) == len(regions0) + 1
    assert set(np.unique(labels).tolist()) == set(range(len(regions)))
    part = _caliper_expected()
    new_id = len(regions0)
    assert (labels[part] == new_id).all() and (labels[~part] == labels0[~part]).all()
    assert (labels[84:89, 140:145] == 1).all()                         # the hole keeps the frame
    rec = regions[new_id]
    assert rec["source"] == "user" and rec["part_kind"] == "user_1" and rec["part_label"] == "Part 1"
    assert rec["area"] == int(part.sum()) and rec["backdrop"] is False
    g = next(g for g in out["groups"] if g["id"] == out["created_group"])
    assert g["name"] == "Part 1" and g["part"] == "user_1" and g["region_ids"] == [new_id]
    assert not g["is_background"] and not g["locked"] and out["created_part"]["area"] == int(part.sum())
    # the mapping followed the frame, the new part is unpainted
    frame2 = next(x for x in out["groups"] if 1 in x["region_ids"])
    assert out["mapping"] == {str(frame2["id"]): "#1b2a57"}
    # the id maps the viewer decodes, the masks, the seed and the registry are all up to date
    rid = pipeline.decode_region_ids(imageio.load_image(job.path("ids", "regions.png")))
    assert (rid == labels).all()
    gm = np.load(job.path("group_map.npy"))
    assert (imageio.load_image(job.path("ids", "groups.png"))[..., 0] == gm).all() and (gm[part] == g["id"]).all()
    assert (np.load(job.path("islands.npy")) == islands0).all() and np.load(job.path("protect.npy")).shape == (H, W)
    seed = pipeline._load_seed(job, labels.shape)
    home = int(seed.origin[new_id])
    assert home >= 0 and seed.part_tags[home]["kind"] == "user_1" and (seed.labels[part] == home).all()
    assert pipeline._load_user_parts(job) == {"user_1": {"label": "Part 1", "regions": [new_id],
                                                         "donors": {"1": int(part.sum())}, "new_id": new_id}}
    assert pipeline._read_user_file(job)["parts"]["user_1"]["was"] == {}
    with np.load(job.path("parts", "user_1.npz")) as z:                 # where each carved pixel came from
        x0, y0, x1, y1 = z["bbox"]
        assert (z["donors"] == np.where(part[y0:y1, x0:x1], 1, -1)).all()
    assert not any(f.endswith(".tmp") or ".tmp." in f for f in os.listdir(job.dir))   # atomic writes leave nothing


def test_from_mask_names_the_part_and_numbers_the_next(client, env):
    job = env["make_job"]()
    r = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]], "name": "  Far caliper "})
    assert r.status_code == 200
    assert next(g for g in r.json()["groups"] if g["id"] == r.json()["created_group"])["name"] == "Far caliper"
    r2 = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*SLIVER, 1]]})
    g2 = next(g for g in r2.json()["groups"] if g["id"] == r2.json()["created_group"])
    assert g2["name"] == "Part 2" and g2["part"] == "user_2"
    assert client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*SLIVER, 1]], "name": "x" * 49}).status_code == 400
    assert client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": []}).status_code == 400


def test_a_user_part_survives_auto_regroup_and_the_junk_pruning(client, env):
    """The sliver is 40 px of the frame's own colour (2 dE off): without its tag the clustering
    would merge it into the frame, and as a group of its own the pruning would fold it."""
    job = env["make_job"]()
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*SLIVER, 1]]}).json()
    rid = next(g for g in out["groups"] if g["id"] == out["created_group"])["region_ids"]
    for body in ({"delta_e": job.options.delta_e}, {"max_groups": 2}, {"delta_e": 40}):
        r = client.post(f"/api/jobs/{job.id}/regroup", json=body)
        assert r.status_code == 200, r.text
        parts = [g for g in r.json()["groups"] if g["part"] == "user_1"]
        assert len(parts) == 1 and parts[0]["region_ids"] == rid and parts[0]["name"] == "Part 1", body
        assert not parts[0]["is_background"]
    seed = pipeline._load_seed(job, np.load(job.path("labels.npy")).shape)
    assert seed.prune is not None                                        # the regroup ran the pruning


def test_the_regroup_safety_net_restores_a_part_the_seed_lost(client, env):
    job = env["make_job"]()
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).json()
    rid = next(g for g in out["groups"] if g["id"] == out["created_group"])["region_ids"]
    arrays = pipeline._seed_arrays(job)                                  # a seed without the part's input region
    n_in = int(arrays["labels"].max()) + 1
    pipeline._put_seed_tags(arrays, {k: v for k, v in pipeline._seed_tags(arrays, n_in).items()
                                     if not v["kind"].startswith("user_")}, n_in)
    arrays["origin"][rid[0]] = 1
    pipeline._save_seed_arrays(job, arrays)
    r = client.post(f"/api/jobs/{job.id}/regroup", json={"delta_e": job.options.delta_e})
    assert [g["region_ids"] for g in r.json()["groups"] if g["part"] == "user_1"] == [rid]


def test_deleting_a_user_part_merges_it_into_its_neighbour_for_good(client, env):
    job = env["make_job"]()
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    frame = next(g for g in out["groups"] if 1 in g["region_ids"])
    r = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [frame["id"], part["id"]], "into": frame["id"]})
    assert r.status_code == 200, r.text
    assert not any(g["part"] == "user_1" for g in r.json()["groups"])
    merged = next(g for g in r.json()["groups"] if 1 in g["region_ids"])
    assert set(part["region_ids"]) <= set(merged["region_ids"]) and merged["name"] == frame["name"]
    assert pipeline._load_user_parts(job) == {}
    _, regions = _state(job)
    assert regions[part["region_ids"][0]]["part_kind"] == ""
    r2 = client.post(f"/api/jobs/{job.id}/regroup", json={"delta_e": job.options.delta_e})
    assert not any(g["part"].startswith("user_") for g in r2.json()["groups"])        # gone for good
    # a plain merge into a larger neighbour removes a part too (the Part badge follows the area)
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*SLIVER, 1]]}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    frame = next(g for g in out["groups"] if 1 in g["region_ids"])
    r3 = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [part["id"], frame["id"]]})
    assert not any(g["part"].startswith("user_") for g in r3.json()["groups"])
    assert client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [0, 1], "into": 9}).status_code == 400


def test_a_user_part_merged_into_a_detected_part_becomes_an_instance_of_it(client, env):
    job = env["make_job"]()
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*SLIVER, 1]]}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    spring = next(g for g in out["groups"] if g["part"] == "shock_spring")
    r = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [spring["id"], part["id"]]})
    springs = [g for g in r.json()["groups"] if g["part"] == "shock_spring"]
    assert len(springs) == 1 and springs[0]["part_instances"] == 2 and springs[0]["name"] == "Shock springs"
    r2 = client.post(f"/api/jobs/{job.id}/regroup", json={"delta_e": job.options.delta_e})
    springs = [g for g in r2.json()["groups"] if g["part"] == "shock_spring"]
    assert len(springs) == 1 and set(part["region_ids"]) <= set(springs[0]["region_ids"])


def test_moving_regions_in_and_out_of_a_user_part(client, env):
    job = env["make_job"]()
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    tank = next(g for g in out["groups"] if 3 in g["region_ids"])
    r = client.post(f"/api/jobs/{job.id}/groups/move", json={"region_ids": [3], "group_id": part["id"]})
    grown = next(g for g in r.json()["groups"] if g["part"] == "user_1")
    assert 3 in grown["region_ids"] and not any(g["id"] == tank["id"] and 3 in g["region_ids"] for g in r.json()["groups"])
    assert sorted(pipeline._load_user_parts(job)["user_1"]["regions"]) == sorted(grown["region_ids"])
    r2 = client.post(f"/api/jobs/{job.id}/regroup", json={"delta_e": job.options.delta_e})
    assert 3 in next(g for g in r2.json()["groups"] if g["part"] == "user_1")["region_ids"]


def test_from_mask_on_an_old_job_without_masks_seed_or_part_fields(client, env):
    job = env["make_job"]()
    for name in ("islands.npy", "protect.npy", "regroup.npz", "user_flags.json"):
        if os.path.exists(job.path(name)):
            os.remove(job.path(name))
    with open(job.path("regions.json")) as f:
        regions = json.load(f)
    old = [{k: v for k, v in r.items() if not k.startswith("part_") and k not in ("glint", "chrome")} for r in regions]
    with open(job.path("regions.json"), "w") as f:
        json.dump(old, f)
    with job.lock:
        job.meta["groups"] = [{k: v for k, v in g.items() if not k.startswith("part") and k not in ("minor", "parent", "ref_lab")}
                              for g in job.meta["groups"]]
    pipeline._layers_cache.clear()
    r = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]})
    assert r.status_code == 200, r.text
    assert not os.path.exists(job.path("regroup.npz")) and not os.path.exists(job.path("protect.npy"))
    rid = next(g for g in r.json()["groups"] if g["id"] == r.json()["created_group"])["region_ids"]
    r2 = client.post(f"/api/jobs/{job.id}/regroup", json={"max_groups": 2})
    assert [g["region_ids"] for g in r2.json()["groups"] if g["part"] == "user_1"] == [rid]


# ----------------------------------------------------------------------------- /find

def test_find_prompts_the_detector_boxes_and_commits_one(client, env, monkeypatch):
    from recolor.segmentation import partdetect
    job = env["make_job"]()
    seen = []

    def detect(image, phrases):
        seen.append((image.shape, list(phrases)))
        return [{"box": [116, 66, 156, 96], "phrase": phrases[0], "score": 0.41, "det": "owlv2"},
                {"box": [118, 67, 155, 95], "phrase": phrases[0], "score": 0.33, "det": "owlv2"},     # the same caliper
                {"box": [96, 16, 174, 54], "phrase": phrases[0], "score": 0.12, "det": "owlv2"},      # the tank
                {"box": [0, 0, W, H], "phrase": phrases[0], "score": 0.6, "det": "owlv2"}]            # the whole photo

    monkeypatch.setattr(partdetect, "detect", detect)
    r = client.post(f"/api/jobs/{job.id}/find", json={"text": "  brake   caliper "})
    assert r.status_code == 200, r.text
    f = r.json()
    assert seen == [((H, W, 3), ["brake caliper"])] and f["text"] == "brake caliper" and f["detector"] == "owlv2"
    assert [c["score"] for c in f["candidates"]] == [0.41, 0.12]
    assert (_decode(f["candidates"][0]["mask"]) == _caliper_expected()).all()
    commit = f["candidates"][0]["prompt"]
    assert commit["box"] == [116, 66, 156, 96] and commit["points"] == []
    c = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={**commit, "name": "Brake caliper"})
    g = next(x for x in c.json()["groups"] if x["id"] == c.json()["created_group"])
    assert g["name"] == "Brake caliper" and g["area"] == int(_caliper_expected().sum())
    assert client.post(f"/api/jobs/{job.id}/find", json={"text": ""}).status_code == 400
    assert client.post(f"/api/jobs/{job.id}/find", json={"text": "x" * 61}).status_code == 400


def test_find_falls_back_to_florence_and_reports_nothing_found(client, env, monkeypatch):
    from recolor.segmentation import florence, partdetect
    job = env["make_job"]()
    monkeypatch.setattr(partdetect, "detect", lambda image, phrases: None)
    monkeypatch.setattr(florence, "ground", lambda image, text: [{"box": [116, 66, 156, 96], "phrase": text, "score": None}])
    f = client.post(f"/api/jobs/{job.id}/find", json={"text": "caliper"}).json()
    assert f["detector"] == "florence" and len(f["candidates"]) == 1 and f["candidates"][0]["score"] is None
    monkeypatch.setattr(florence, "ground", lambda image, text: None)
    f = client.post(f"/api/jobs/{job.id}/find", json={"text": "caliper"}).json()
    assert f["detector"] is None and f["candidates"] == []


def test_remove_part_dissolves_even_into_a_detected_part(client, env):
    """The studio's Remove merges a drawn part into the group around it with ``dissolve``: its
    regions lose the user tag whatever that group is, so a part next to a detected one never
    turns into another instance of it."""
    job = env["make_job"]()
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*SLIVER, 1]]}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    spring = next(g for g in out["groups"] if g["part"] == "shock_spring")
    r = client.post(f"/api/jobs/{job.id}/groups/merge",
                    json={"group_ids": [spring["id"], part["id"]], "into": spring["id"], "dissolve": True})
    assert r.status_code == 200, r.text
    springs = [g for g in r.json()["groups"] if g["part"] == "shock_spring"]
    assert len(springs) == 1 and springs[0]["part_instances"] == 1 and springs[0]["name"] == "Shock spring"
    assert set(part["region_ids"]) <= set(springs[0]["region_ids"])
    _, regions = _state(job)
    assert all(regions[i]["part_kind"] == "" for i in part["region_ids"]) and pipeline._load_user_parts(job) == {}
    bad = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [0, 1], "into": 0, "dissolve": "yes"})
    assert bad.status_code == 400


def test_find_says_which_part_group_a_candidate_already_is(client, env, monkeypatch):
    from recolor.segmentation import partdetect
    job = env["make_job"]()
    client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]], "name": "Caliper"})
    monkeypatch.setattr(partdetect, "detect", lambda image, phrases: [
        {"box": [116, 66, 156, 96], "phrase": phrases[0], "score": 0.5, "det": "owlv2"},
        {"box": [96, 16, 174, 54], "phrase": phrases[0], "score": 0.2, "det": "owlv2"}])
    f = client.post(f"/api/jobs/{job.id}/find", json={"text": "caliper"}).json()
    assert len(f["candidates"]) == 2
    m = f["candidates"][0]["matches"]
    assert m is not None and m["name"] == "Caliper" and m["iou"] > 0.9
    assert f["candidates"][1]["matches"] is None                        # the tank is a colour group
    assert f["candidates"][0]["rank"] > f["candidates"][1]["rank"]


def test_remove_part_alone_puts_it_back_into_the_group_it_came_from(client, env):
    """``merge {group_ids: [part], dissolve: true}``: the studio's Remove. The part goes back into
    the group of the region that gave it the most pixels (recorded when it was carved)."""
    job = env["make_job"]()
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    assert pipeline._load_user_parts(job)["user_1"]["donors"] == {"1": part["area"]}
    frame = next(g for g in out["groups"] if 1 in g["region_ids"])
    r = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [part["id"]], "dissolve": True})
    assert r.status_code == 200, r.text
    home = next(g for g in r.json()["groups"] if 1 in g["region_ids"])
    assert set(part["region_ids"]) <= set(home["region_ids"]) and home["name"] == frame["name"]
    assert not any(g["part"].startswith("user_") for g in r.json()["groups"]) and pipeline._load_user_parts(job) == {}
    tank = next(g for g in r.json()["groups"] if 3 in g["region_ids"])
    bad = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [tank["id"]], "dissolve": True})
    assert bad.status_code == 400 and "Select part" in bad.json()["detail"]
    assert client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [tank["id"]]}).status_code == 400


def test_remove_part_gives_a_detected_part_its_pixels_back(client, env):
    """A part drawn over a whole detected part (the tank) and a piece of the frame, then removed:
    the tank's region is the tank again, the piece goes back to the frame's group, and a regroup
    keeps it so."""
    job = env["make_job"](tank_part=True)
    assert any(g.part == "tank" and g.region_ids == [3] for g in job.groups())
    work = _scene()[0]
    tank_px = (work == work[30, 130]).all(-1)
    # a box around the tank: the fake answers with the red blob; one more click on the caliper
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask",
                      json={"points": [[130, 30, 1], [*CALIPER, 1]], "multimask": True}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    assert 3 in part["region_ids"] and not any(g["part"] == "tank" for g in out["groups"])     # it took the tank
    rec = pipeline._read_user_file(job)["parts"]["user_1"]
    assert rec["was"]["3"]["kind"] == "tank" and rec["donors"] == {"1": int(_caliper_expected().sum())}
    r = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [part["id"]], "dissolve": True})
    assert r.status_code == 200, r.text
    tanks = [g for g in r.json()["groups"] if g["part"] == "tank"]
    assert len(tanks) == 1 and tanks[0]["region_ids"] == [3] and tanks[0]["name"] == "Fuel tank"
    frame = next(g for g in r.json()["groups"] if 1 in g["region_ids"])
    new_id = next(i for i in part["region_ids"] if i != 3)
    assert new_id in frame["region_ids"] and tank_px.sum() == tanks[0]["area"]
    r2 = client.post(f"/api/jobs/{job.id}/regroup", json={"delta_e": job.options.delta_e})
    assert [g["region_ids"] for g in r2.json()["groups"] if g["part"] == "tank"] == [[3]]
    assert not any(g["part"].startswith("user_") for g in r2.json()["groups"])


def test_a_part_that_would_swallow_the_whole_image_is_refused(client, env, monkeypatch):
    job = env["make_job"]()

    def everything(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True):
        h, w = self.flat.shape
        n = 3 if multimask_output else 1
        return np.ones((n, h, w), np.float32), np.full(n, 0.9, np.float32), np.zeros((n, 256, 256), np.float32)

    monkeypatch.setattr(FakePredictor, "predict", everything)
    r = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]})
    assert r.status_code == 400 and "whole image" in r.json()["detail"]
    assert int(np.load(job.path("labels.npy")).max()) == 3                # nothing was written


# ----------------------------------------------------------------------------- hardening (round 2)

def test_prompt_numbers_too_large_for_a_float_and_float_labels_are_clean_400s(client, env):
    job = env["make_job"]()
    big = "1" + "0" * 400
    raw = {"x": '{"points": [[' + big + ', 10, 1]]}',
           "box": '{"box": [0, 0, ' + big + ', 10]}',
           "crop": '{"points": [[136, 73, 1]], "crop": [0, 0, ' + big + ', 10]}'}
    for what, body in raw.items():
        for path in ("segment", "groups/from_mask"):
            r = client.post(f"/api/jobs/{job.id}/{path}", content=body, headers={"content-type": "application/json"})
            assert r.status_code == 400 and "finite number" in r.json()["detail"], (what, path, r.text)
    with pytest.raises(interactive.PromptError):
        interactive.parse_prompt({"points": [[3, 3, 1.0]]}, W, H)                 # a float is no label
    assert interactive.parse_prompt({"points": [[3, 3, True]]}, W, H).points[0][2] == 1
    with pytest.raises(interactive.PromptError) as e:
        interactive.parse_prompt({"points": [[1e300, 3, 1]]}, W, H)
    assert len(str(e.value)) < 120                                                # no 300-digit coordinate


def test_json_bodies_nested_too_deeply_or_too_large_are_4xx_on_every_route(client, env):
    job = env["make_job"]()
    deep = '{"points": ' + "[" * 100000 + "]" * 100000 + "}"
    for path in (f"/api/jobs/{job.id}/segment", f"/api/jobs/{job.id}/groups/from_mask", f"/api/jobs/{job.id}/find",
                 f"/api/jobs/{job.id}/groups/merge", f"/api/jobs/{job.id}/render", "/api/jobs"):
        r = client.post(path, content=deep, headers={"content-type": "application/json"})
        assert r.status_code == 400 and r.json()["error"] == "bad_json", (path, r.status_code, r.text[:200])
    huge = '{"points": [], "pad": "' + "x" * (2 << 20) + '"}'
    r = client.post(f"/api/jobs/{job.id}/segment", content=huge, headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_a_commit_waits_for_the_gpu_without_holding_the_edit_lock(client, env, monkeypatch):
    import time as _time
    job = env["make_job"]()
    monkeypatch.setattr(pipeline, "PROMPT_GPU_WAIT_S", 1.0)
    held, done = threading.Event(), threading.Event()

    def hold():
        with pipeline.gpu_lock:
            held.set()
            done.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    held.wait(5)
    res = {}
    c = threading.Thread(target=lambda: res.update(r=client.post(f"/api/jobs/{job.id}/groups/from_mask",
                                                                 json={"points": [[*CALIPER, 1]]})))
    try:
        c.start()
        _time.sleep(0.15)
        t0 = _time.perf_counter()
        r = client.patch(f"/api/jobs/{job.id}/groups/{job.groups()[0].id}", json={"name": "Renamed"})
        waited = _time.perf_counter() - t0
        c.join(5)
    finally:
        done.set()
        t.join()
    assert r.status_code == 200 and waited < 0.6, waited              # the rename never queued behind the commit
    assert res["r"].status_code == 503


def test_the_carve_and_the_writes_run_outside_the_gpu_lock(client, env, monkeypatch):
    job = env["make_job"]()
    seen = []
    real_carve, real_write = userparts.carve, pipeline._write_grouping

    def carve(*a, **k):
        seen.append(("carve", pipeline.gpu_lock._is_owned()))
        return real_carve(*a, **k)

    def write(*a, **k):
        seen.append(("write", pipeline.gpu_lock._is_owned()))
        return real_write(*a, **k)

    monkeypatch.setattr(userparts, "carve", carve)
    monkeypatch.setattr(pipeline, "_write_grouping", write)
    assert client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).status_code == 200
    assert {k for k, _ in seen} == {"carve", "write"} and not any(v for _, v in seen)


# ----------------------------------------------------------------------------- paint and names (round 2)

def _group_of(groups, rid):
    return next(g for g in groups if rid in g["region_ids"])


def test_remove_part_takes_its_paint_with_it(client, env):
    """The part painted red, the frame it came from unpainted: after Remove the frame is still
    unpainted (carried by membership the part's red painted the whole frame)."""
    job = env["make_job"]()
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    spring = next(g for g in out["groups"] if g["part"] == "shock_spring")
    pipeline.save_state(job, {"mapping": {str(part["id"]): "#d62828", str(spring["id"]): "#f5c518"}})
    j = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [part["id"]], "dissolve": True}).json()
    spring2 = next(g for g in j["groups"] if g["part"] == "shock_spring")
    assert j["mapping"] == {str(spring2["id"]): "#f5c518"}                  # the frame unpainted, the spring kept


def test_merging_into_a_group_keeps_that_groups_own_paint(client, env):
    job = env["make_job"]()
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    frame = _group_of(out["groups"], 1)
    tank = _group_of(out["groups"], 3)
    pipeline.save_state(job, {"mapping": {str(part["id"]): "#d62828"}})
    j = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [frame["id"], part["id"]], "into": frame["id"]}).json()
    assert j["mapping"] == {}                                               # the unpainted frame stays unpainted
    # into a painted group: its own paint, even when a painted member has the lower id
    frame, tank = _group_of(j["groups"], 1), _group_of(j["groups"], 3)
    lo, hi = sorted((frame["id"], tank["id"]))
    pipeline.save_state(job, {"mapping": {str(lo): "#111111", str(hi): "#eeeeee"}})
    j2 = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [lo, hi], "into": hi}).json()
    merged = _group_of(j2["groups"], 1)
    assert j2["mapping"] == {str(merged["id"]): "#eeeeee"}


def test_keep_names_restores_every_name_the_rebuild_renumbered():
    from recolor.types import ColorGroup

    def g(i, name, part="", label=""):
        return ColorGroup(id=i, name=name, albedo_lab=(60.0, 0.0, 0.0), albedo_hex="#919191", area=100, area_frac=0.1,
                          region_ids=[i], hue_family="neutral", part=part, part_label=label, part_plural=label + "s",
                          part_instances=1 if part else 0)

    old = [g(0, "Gray"), g(1, "Gray 2"), g(2, "Gray 3"), g(3, "Brake calipers", "brake_caliper", "Brake caliper")]
    new = [g(0, "Gray"), g(1, "Charcoal"), g(2, "Gray 2"), g(3, "Brake caliper", "brake_caliper", "Brake caliper"),
           g(4, "Gray", "user_1", "Gray")]
    userparts.keep_names(old, new, {0: 0, 1: 1, 2: 2, 3: 3}, recompute={3})
    assert [x.name for x in new] == ["Gray", "Gray 2", "Gray 3", "Brake caliper", "Gray 4"]


def test_a_commit_renames_no_group_it_did_not_make(client, env):
    job = env["make_job"]()
    before = {tuple(g.region_ids): g.name for g in job.groups()}
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]], "name": "Caliper"}).json()
    for g in out["groups"]:
        if g["id"] == out["created_group"]:
            continue
        old = next(name for rids, name in before.items() if set(rids) & set(g["region_ids"]))
        assert g["name"] == old


# ----------------------------------------------------------------------------- absorb and replace (round 2)

def _answer_with(monkeypatch, mask):
    """Every prompt answers with ``mask`` (no refinement crop)."""
    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True):
        n = 3 if multimask_output else 1
        return (np.stack([mask] * n).astype(np.float32), np.array([0.9] * n, np.float32)[:n],
                np.zeros((n, 256, 256), np.float32))
    monkeypatch.setattr(FakePredictor, "predict", predict)
    monkeypatch.setattr(interactive, "choose_crop", lambda *a, **k: None)


def test_a_part_covering_most_of_a_detected_part_takes_it_whole_and_remove_gives_it_back(client, env, monkeypatch):
    job = env["make_job"]()
    spring = next(g for g in job.groups() if g.part == "shock_spring")
    labels0 = np.load(job.path("labels.npy"))
    m = np.isin(labels0, spring.region_ids)
    ys, xs = np.nonzero(m)
    m[ys.max() - 1:, :] = False                                   # 2 rows short: 93 % of the spring ...
    m[40:44, 45:50] = True                                        # ... and a bit of the frame beside it
    _answer_with(monkeypatch, m)
    pipeline.save_state(job, {"mapping": {str(spring.id): "#f5c518"}})
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[35, 40, 1]], "name": "Spring"}).json()
    assert out["created_part"]["took_in"] == ["Shock spring"] and out["created_part"]["replaced"] == "Shock spring"
    assert not any(g["part"] == "shock_spring" for g in out["groups"])               # no sliver of it is left
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    assert set(spring.region_ids) <= set(part["region_ids"]) and part["name"] == "Spring"
    assert out["mapping"] == {str(part["id"]): "#f5c518"}                            # it replaces the spring: its paint
    r = client.post(f"/api/jobs/{job.id}/regroup", json={"delta_e": job.options.delta_e}).json()
    assert not any(g["part"] == "shock_spring" for g in r["groups"])                 # nor after a regroup
    part = next(g for g in r["groups"] if g["part"] == "user_1")
    j = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [part["id"]], "dissolve": True}).json()
    back = [g for g in j["groups"] if g["part"] == "shock_spring"]
    assert len(back) == 1 and set(spring.region_ids) <= set(back[0]["region_ids"])
    assert j["mapping"].get(str(back[0]["id"])) == "#f5c518"                         # with the paint it had


def test_selecting_a_user_part_again_replaces_it_and_remove_brings_it_back(client, env):
    job = env["make_job"]()
    first = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]], "name": "Far caliper"}).json()
    old = next(g for g in first["groups"] if g["id"] == first["created_group"])
    pipeline.save_state(job, {"mapping": {str(old["id"]): "#d62828"}})
    again = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).json()
    new = next(g for g in again["groups"] if g["id"] == again["created_group"])
    assert again["created_part"]["replaced"] == "Far caliper" and new["name"] == "Far caliper" and new["part"] == "user_2"
    assert not any(g["part"] == "user_1" for g in again["groups"])
    assert again["mapping"] == {str(new["id"]): "#d62828"}                          # the paint goes on
    reg = pipeline._load_user_parts(job)
    assert reg["user_1"]["regions"] == [] and reg["user_1"]["inside"] == "user_2"   # it waits inside the new one
    assert sorted(reg["user_2"]["regions"]) == sorted(new["region_ids"])
    j = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [new["id"]], "dissolve": True}).json()
    back = [g for g in j["groups"] if g["part"] == "user_1"]
    assert len(back) == 1 and back[0]["name"] == "Far caliper" and back[0]["region_ids"] == old["region_ids"]
    assert j["mapping"] == {str(back[0]["id"]): "#d62828"}
    assert set(pipeline._load_user_parts(job)) == {"user_1"}


def test_a_part_drawn_over_an_earlier_one_keeps_the_registry_consistent(client, env):
    """The bolt drawn first, then the caliper around it: the bolt's registry entry waits inside the
    caliper (it listed region 4, tagged user_2 by then), a regroup keeps one group per part, and
    Remove of the caliper brings the bolt back as its own part."""
    job = env["make_job"]()
    client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*BOLT, 1]], "name": "Bolt"})
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]], "name": "Caliper"}).json()
    assert out["created_part"]["took_in"] == ["Bolt"] and out["created_part"]["replaced"] is None
    _, regions = _state(job)
    tagged = {k: sorted(r["id"] for r in regions if r["part_kind"] == k) for k in ("user_1", "user_2")}
    reg = pipeline._load_user_parts(job)
    assert tagged["user_1"] == [] and reg["user_1"]["regions"] == [] and sorted(reg["user_2"]["regions"]) == tagged["user_2"]
    r = client.post(f"/api/jobs/{job.id}/regroup", json={"delta_e": job.options.delta_e}).json()
    assert [g["name"] for g in r["groups"] if g["part"].startswith("user_")] == ["Caliper"]
    cal = next(g for g in r["groups"] if g["part"] == "user_2")
    j = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [cal["id"]], "dissolve": True}).json()
    assert [g["name"] for g in j["groups"] if g["part"].startswith("user_")] == ["Bolt"]
    assert set(pipeline._load_user_parts(job)) == {"user_1"}


# ----------------------------------------------------------------------------- Find (round 2)

def test_find_offers_the_part_groups_it_names_first_and_asks_the_detector_with_synonyms(client, env, monkeypatch):
    from recolor.segmentation import partdetect
    job = env["make_job"]()
    client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]], "name": "Far caliper"})
    seen = []

    def detect(image, phrases):
        seen.append(list(phrases))
        return [{"box": [116, 66, 156, 96], "phrase": phrases[1], "score": 0.3, "det": "owlv2"},       # the same caliper
                {"box": [96, 16, 174, 54], "phrase": phrases[0], "score": 0.2, "det": "owlv2"}]        # the tank

    monkeypatch.setattr(partdetect, "detect", detect)
    f = client.post(f"/api/jobs/{job.id}/find", json={"text": "calipers"}).json()
    assert seen == [["calipers", "brake caliper"]] and f["phrases"] == seen[0]
    first = f["candidates"][0]
    assert first["existing"] is True and first["prompt"] is None and first["matches"]["name"] == "Far caliper"
    assert first["mask"]["area"] == int(_caliper_expected().sum())
    assert len(f["candidates"]) == 2 and f["candidates"][1]["existing"] is False     # the detector's caliper is the same mask
    springs = client.post(f"/api/jobs/{job.id}/find", json={"text": "spring"}).json()
    assert springs["candidates"][0]["existing"] and springs["candidates"][0]["matches"]["name"] == "Shock spring"
    assert seen[-1][:3] == ["spring", "coil spring", "shock absorber"]
    assert pipeline._find_phrases("mirror")[:2] == ["mirror", "rear view mirror"]


def test_absorb_takes_a_part_instance_it_covers_almost_whole_and_nothing_else():
    from recolor.types import Region

    def reg(i, kind="", inst=0):
        return Region(id=i, area=1, bbox=(0, 0, 1, 1), albedo_lab=(50.0, 0.0, 0.0), albedo_hex="#777777", group_id=0,
                      touches_border=False, source="sam", part_kind=kind, part_label=kind, part_plural=kind,
                      part_instance=inst if kind else -1)

    labels = np.zeros((40, 60), np.int32)
    labels[5:15, 5:15] = 1                                     # a caliper instance, covered 90 %
    labels[5:15, 40:50] = 2                                    # the far instance of the same kind, untouched
    labels[25:35, 5:15] = 3                                    # a colour region, covered 90 %
    regions = [reg(0), reg(1, "brake_caliper", 0), reg(2, "brake_caliper", 1), reg(3)]
    m = np.zeros_like(labels, bool)
    m[5:14, 5:15] = True
    m[25:34, 5:15] = True
    grown, taken = userparts.absorb_parts(m, labels, regions)
    assert taken == [1] and grown[5:15, 5:15].all() and not grown[5:15, 40:50].any()
    assert not grown[34, 5:15].any()                           # the colour region is the carve's to cut
    few = np.zeros_like(m)
    few[5:12, 5:15] = True                                     # 70 %: cut, not taken
    assert userparts.absorb_parts(few, labels, regions)[1] == []


def test_opening_the_tool_warms_what_a_commit_needs(client, env, monkeypatch):
    import time as _time
    from recolor.segmentation import refine
    job = env["make_job"]()
    refine._PHOTO_LAB.clear()
    pipeline._SPEC_Q.clear()
    assert client.post(f"/api/jobs/{job.id}/segment", json={}).json()["warm"] is True
    for _ in range(100):
        if len(refine._PHOTO_LAB) and len(pipeline._SPEC_Q) and job.id not in pipeline._warming:
            break
        _time.sleep(0.02)
    assert len(refine._PHOTO_LAB) == 1 and len(pipeline._SPEC_Q) == 1
    calls = []
    from recolor.segmentation import materials
    real = materials.spec_reference
    monkeypatch.setattr(materials, "spec_reference", lambda r: calls.append(1) or real(r))
    assert client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).status_code == 200
    assert calls == []                                                   # the commit took the cached reference


# ----------------------------------------------------------------------------- round 3: flags, colour groups, instances

def _gid(groups, rid):
    return next(g for g in groups if rid in g["region_ids"])


def test_a_commit_or_remove_keeps_every_other_groups_background_flag(client, env, monkeypatch):
    """The spring flagged background (and locked) by hand: an unrelated commit, a Remove and a merge
    that dissolves another user part keep the flag (a commit cleared it on every part group)."""
    job = env["make_job"]()
    spring = next(g for g in job.groups() if g.part == "shock_spring")
    r = client.patch(f"/api/jobs/{job.id}/groups/{spring.id}", json={"is_background": True, "locked": True})
    assert r.status_code == 200 and _gid(r.json()["groups"], 2)["is_background"] is True

    def spring_flags():
        g = next(g for g in job.groups() if 2 in g.region_ids)
        return g.is_background, g.locked

    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).json()
    assert spring_flags() == (True, True)
    sliver = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*SLIVER, 1]]}).json()
    assert spring_flags() == (True, True)
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    part = _gid(job.snapshot()["groups"], part["region_ids"][0])
    assert client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [part["id"]], "dissolve": True}).status_code == 200
    assert spring_flags() == (True, True)
    # a plain merge of the sliver part into the frame (it is dissolved: the user parts follow, a rebuild)
    groups = job.snapshot()["groups"]
    sp = next(g for g in groups if g["part"] == "user_2")
    frame = _gid(groups, 1)
    assert client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [sp["id"], frame["id"]]}).status_code == 200
    assert spring_flags() == (True, True)
    assert not any(g.part.startswith("user_") for g in job.groups())
    # the new part itself is never background, whatever it was cut from
    back = next(g for g in job.groups() if 0 in g.region_ids)
    client.patch(f"/api/jobs/{job.id}/groups/{back.id}", json={"is_background": True})
    m = np.zeros((H, W), bool)
    m[2:10, 2:40] = True                                              # a strip of the backdrop
    _answer_with(monkeypatch, m)
    j = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[5, 5, 1]]}).json()
    new = next(g for g in j["groups"] if g["id"] == j["created_group"])
    assert new["is_background"] is False and _gid(j["groups"], 0)["is_background"] is True
    assert spring_flags() == (True, True)


def test_a_colour_group_taken_whole_is_replaced_and_remove_gives_it_back(client, env, monkeypatch):
    """The red tank is a colour group of one region, painted and locked. Selected exactly, the new
    part replaces it (its paint and name); Remove gives the tank back as its own group, with its
    name, lock and paint (it went into the frame's group, and its paint was lost)."""
    job = env["make_job"]()
    labels0 = np.load(job.path("labels.npy"))
    tank = next(g for g in job.groups() if 3 in g.region_ids)
    frame = next(g for g in job.groups() if 1 in g.region_ids)
    client.patch(f"/api/jobs/{job.id}/groups/{tank.id}", json={"name": "Tank", "locked": True})
    pipeline.save_state(job, {"mapping": {str(tank.id): "#0000ff", str(frame.id): "#123456"}})
    _answer_with(monkeypatch, labels0 == 3)
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[120, 30, 1]]}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    assert out["created_part"]["replaced"] == "Tank" and out["created_part"]["took_in"] == ["Tank"]
    assert part["name"] == "Tank" and part["region_ids"] == [3] and part["part"] == "user_1"
    assert out["mapping"] == {str(part["id"]): "#0000ff", str(_gid(out["groups"], 1)["id"]): "#123456"}
    rec = pipeline._load_user_parts(job)["user_1"]
    assert rec["groups"] == [{"regions": [3], "name": "Tank", "locked": True, "is_background": False, "part": "",
                              "paint": "#0000ff"}]
    pipeline.save_state(job, {"mapping": {**out["mapping"], str(part["id"]): "#ff00ff"}})     # repainted as a part
    j = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [part["id"]], "dissolve": True}).json()
    t, f = _gid(j["groups"], 3), _gid(j["groups"], 1)
    assert t["id"] != f["id"] and t["region_ids"] == [3] and t["name"] == "Tank" and t["locked"] is True and not t["part"]
    assert j["mapping"] == {str(t["id"]): "#0000ff", str(f["id"]): "#123456"}             # its own paint of then
    assert pipeline._load_user_parts(job) == {}
    assert j["removed_part"]["name"] == "Tank" and j["removed_part"]["restored"] == ["Tank"]


def test_a_colour_group_taken_with_a_piece_of_its_neighbour_comes_back_alone(client, env, monkeypatch):
    job = env["make_job"]()
    labels0 = np.load(job.path("labels.npy"))
    names0 = sorted(g.name for g in job.groups())
    m = labels0 == 3
    m[50:56, 100:170] = True                                          # a strip of the frame below the tank
    _answer_with(monkeypatch, m)
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[120, 30, 1]], "name": "Tank"}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    assert 3 in part["region_ids"] and len(part["region_ids"]) == 2
    j = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [part["id"]], "dissolve": True}).json()
    t, f = _gid(j["groups"], 3), _gid(j["groups"], 1)
    new_id = next(i for i in part["region_ids"] if i != 3)
    assert t["region_ids"] == [3] and new_id in f["region_ids"] and sorted(g["name"] for g in j["groups"]) == names0


def test_remove_gives_split_instances_back_each_with_its_paint(client, env, monkeypatch):
    """Two springs split into instances and painted differently, both taken in whole by one part:
    Remove gives back two groups, each with its name and its own paint (the bookkeeping was by part
    kind, so they came back as one group with one paint)."""
    job = env["make_job"](two_springs=True)
    springs = next(g for g in job.groups() if g.part == "shock_spring")
    assert springs.part_instances == 2
    r = client.post(f"/api/jobs/{job.id}/groups/split", json={"group_id": springs.id, "mode": "instances"}).json()
    halves = sorted((g for g in r["groups"] if g["part"] == "shock_spring"), key=lambda g: g["region_ids"])
    assert [g["region_ids"] for g in halves] == [[2], [4]]
    names = [g["name"] for g in halves]
    pipeline.save_state(job, {"mapping": {str(halves[0]["id"]): "#ff0000", str(halves[1]["id"]): "#00ff00"}})
    labels0 = np.load(job.path("labels.npy"))
    m = np.isin(labels0, [2, 4])
    m[30:92, 45:50] = True                                            # and a strip of the frame beside them
    _answer_with(monkeypatch, m)
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[35, 40, 1]], "name": "Springs"}).json()
    assert sorted(out["created_part"]["took_in"]) == sorted(names) and out["created_part"]["replaced"] is None
    assert not any(g["part"] == "shock_spring" for g in out["groups"])
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    j = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [part["id"]], "dissolve": True}).json()
    back = sorted((g for g in j["groups"] if g["part"] == "shock_spring"), key=lambda g: g["region_ids"])
    assert [g["region_ids"] for g in back] == [[2], [4]] and [g["name"] for g in back] == names
    assert j["mapping"] == {str(back[0]["id"]): "#ff0000", str(back[1]["id"]): "#00ff00"}
    assert sorted(j["removed_part"]["restored"]) == sorted(names)


def test_a_region_of_a_group_taken_in_part_goes_back_to_that_group(client, env, monkeypatch):
    """The frame's group gets a second region (the sliver's, moved in by hand); a part covering that
    region whole and a bit of the tank: Remove puts the region back in the frame's group, not in the
    tank's (the main donor's)."""
    job = env["make_job"]()
    labels0 = np.load(job.path("labels.npy"))
    # make the sliver a region of its own in the frame's group (a split cuts it off the frame)
    frame = next(g for g in job.groups() if 1 in g.region_ids)
    m0 = np.zeros((H, W), bool)
    m0[95:99, 60:70] = True
    _answer_with(monkeypatch, m0)
    first = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[64, 96, 1]]}).json()
    p1 = next(g for g in first["groups"] if g["id"] == first["created_group"])
    frame = _gid(first["groups"], 1)
    moved = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [frame["id"], p1["id"]], "into": frame["id"]}).json()
    frame = _gid(moved["groups"], 1)
    sliver_rid = p1["region_ids"][0]
    assert sliver_rid in frame["region_ids"] and not any(g["part"].startswith("user_") for g in moved["groups"])
    labels1 = np.load(job.path("labels.npy"))
    m = labels1 == sliver_rid
    m[40:50, 150:170] = True                                          # and a corner of the tank (the main donor)
    _answer_with(monkeypatch, m)
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[64, 96, 1]]}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    rec = pipeline._load_user_parts(job)[part["part"]]
    assert rec["homes"] == {str(sliver_rid): 1} and set(rec["donors"]) == {"3"}
    j = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [part["id"]], "dissolve": True}).json()
    assert _gid(j["groups"], sliver_rid)["id"] == _gid(j["groups"], 1)["id"]
    assert _gid(j["groups"], 3)["id"] != _gid(j["groups"], 1)["id"]


def test_absorb_measures_an_instance_over_all_its_regions():
    from recolor.types import Region

    def reg(i, kind="", inst=0):
        return Region(id=i, area=1, bbox=(0, 0, 1, 1), albedo_lab=(50.0, 0.0, 0.0), albedo_hex="#777777", group_id=0,
                      touches_border=False, source="sam", part_kind=kind, part_label=kind, part_plural=kind,
                      part_instance=inst if kind else -1)

    labels = np.zeros((40, 60), np.int32)
    labels[0:30, 0:40] = 1                                     # a drawn part: one big region ...
    labels[32:34, 50:53] = 2                                   # ... and a small one far from it
    regions = [reg(0), reg(1, "user_1"), reg(2, "user_1")]
    m = labels == 1                                            # 1200 of the part's 1206 px: 99.5 %
    grown, taken = userparts.absorb_parts(m, labels, regions)
    assert taken == [2] and grown[32:34, 50:53].all()          # the instance is taken whole, small region too
    m2 = np.zeros_like(m)
    m2[0:20, 0:40] = True                                      # 800 of 1206: cut
    assert userparts.absorb_parts(m2, labels, regions)[1] == []


def test_take_all_grows_the_part_over_a_group_under_the_selection(client, env, monkeypatch):
    job = env["make_job"]()
    spring = next(g for g in job.groups() if g.part == "shock_spring")
    labels0 = np.load(job.path("labels.npy"))
    m = np.isin(labels0, spring.region_ids)
    m[45:, :] = False                                           # half of the spring: cut, not taken
    _answer_with(monkeypatch, m)
    pipeline.save_state(job, {"mapping": {str(spring.id): "#f5c518"}})
    cut = client.post(f"/api/jobs/{job.id}/segment", json={"points": [[35, 40, 1]]}).json()
    assert cut["mask"]["area"] == int(m.sum())
    bad = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[35, 40, 1]], "take": [_gid(job.snapshot()["groups"], 3)["id"]]})
    assert bad.status_code == 400 and "not under the selection" in bad.json()["detail"]
    for take in ([True], ["1"], [1.5], list(range(9)), "x", [99]):
        r = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[35, 40, 1]], "take": take})
        assert r.status_code == 400, take
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[35, 40, 1]], "take": [spring.id]}).json()
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    assert set(spring.region_ids) <= set(part["region_ids"]) and not any(g["part"] == "shock_spring" for g in out["groups"])
    assert out["created_part"]["replaced"] == "Shock spring" and part["name"] == "Shock spring"
    assert out["mapping"] == {str(part["id"]): "#f5c518"}


def test_regroup_and_job_numbers_are_bounded_like_a_new_job(client, env):
    job = env["make_job"]()
    for body in ({"max_groups": 10 ** 400}, {"max_groups": 1.7}, {"max_groups": 257}, {"max_groups": 0},
                 {"max_groups": "abc"}, {"max_groups": True}, {"delta_e": 10 ** 400}, {"delta_e": "hot"}):
        raw = json.dumps(body)
        r = client.post(f"/api/jobs/{job.id}/regroup", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 400, (raw[:60], r.status_code, r.text[:200])
    for raw in ('{"max_groups": NaN}', '{"delta_e": Infinity}', '{"delta_e": 1e400}'):
        r = client.post(f"/api/jobs/{job.id}/regroup", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 400 and r.json()["error"] == "bad_json", raw
    r = client.post(f"/api/jobs/{job.id}/render", content='{"options": {"texture": ' + "9" * 400 + '}}',
                    headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert client.post(f"/api/jobs/{job.id}/regroup", json={"max_groups": 2.0}).json()["options"]["max_groups"] == 2
    with job.lock:                                               # an older server stored an unusable option
        job.meta["options"] = {**job.meta["options"], "max_groups": 10 ** 400}
    r = client.post(f"/api/jobs/{job.id}/regroup", json={})
    assert r.status_code == 200 and r.json()["options"]["max_groups"] is None
    with pytest.raises(pipeline.PipelineError):
        pipeline.parse_analysis_options({"max_groups": 1.7})
    assert pipeline.parse_analysis_options({"max_groups": "12", "delta_e": "8"}).max_groups == 12


def test_a_chunked_body_over_the_limit_is_413_before_it_is_read_whole(client, env):
    """A body without Content-Length is read chunk by chunk and nothing past the limit is kept (it
    was read whole into memory first); the rest is read and thrown away up to MAX_DRAIN_BYTES, so a
    client still sending gets the 413 instead of a reset connection, and beyond that the reading
    stops. The TestClient buffers a streamed body itself, so the reader is driven here with the ASGI
    messages a server would send."""
    import anyio
    from starlette.requests import Request
    from recolor.server import app as app_mod

    def run(n_chunks):
        consumed = []

        async def receive():
            consumed.append(1)
            return {"type": "http.request", "body": b" " * 65536, "more_body": len(consumed) < n_chunks}

        req = Request({"type": "http", "method": "POST", "path": "/", "headers": []}, receive)
        with pytest.raises(app_mod.ApiError) as e:
            anyio.run(app_mod._read_limited, req)
        assert e.value.status == 413
        return len(consumed)

    assert run(64) == 64                                                     # 4 MB: read to the end, kept 1 MB at most
    assert run(4096) == app_mod.MAX_DRAIN_BYTES // 65536 + 1                 # 256 MB: given up past the drain limit
    job = env["make_job"]()
    r = client.post(f"/api/jobs/{job.id}/segment", content=iter([b" " * 65536] * 20), headers={"content-type": "application/json"})
    assert r.status_code == 413
    ok = client.post(f"/api/jobs/{job.id}/segment", content=iter([b'{"points": [[', b'136, 73, 1]]}']),
                     headers={"content-type": "application/json"})
    assert ok.status_code == 200


def test_find_puts_a_candidate_that_is_a_part_of_another_kind_last(client, env, monkeypatch):
    """Find "caliper" on a photo whose tank is a detected part: the detector's best box is the tank,
    which is the Fuel tank group; it comes after the caliper, flagged as another kind's."""
    from recolor.segmentation import partdetect
    job = env["make_job"](tank_part=True)
    monkeypatch.setattr(partdetect, "detect", lambda image, phrases: [
        {"box": [96, 16, 174, 54], "phrase": phrases[0], "score": 0.6, "det": "owlv2"},
        {"box": [116, 66, 156, 96], "phrase": phrases[0], "score": 0.3, "det": "owlv2"}])
    f = client.post(f"/api/jobs/{job.id}/find", json={"text": "caliper"}).json()
    assert [c["score"] for c in f["candidates"]] == [0.3, 0.6]
    assert f["candidates"][0]["matches"] is None
    assert f["candidates"][1]["matches"]["name"] == "Fuel tank" and f["candidates"][1]["matches"]["named"] is False
    t = client.post(f"/api/jobs/{job.id}/find", json={"text": "tank"}).json()
    assert t["candidates"][0]["existing"] and t["candidates"][0]["matches"]["named"] is True


def test_names_part_reads_the_vocabulary_heads_and_synonyms():
    from recolor.types import ColorGroup

    def part(kind, label, plural):
        return ColorGroup(id=0, name=plural, albedo_lab=(50.0, 0.0, 0.0), albedo_hex="#777777", area=10, area_frac=0.1,
                          region_ids=[0], hue_family="neutral", part=kind, part_label=label, part_plural=plural,
                          part_instances=2)

    exhaust, seat, spring = part("exhaust", "Exhaust", "Exhausts"), part("seat", "Seat", "Seats"), \
        part("shock_spring", "Shock spring", "Shock springs")
    w = pipeline._find_words
    assert pipeline._names_part(exhaust, w("muffler")) and pipeline._names_part(exhaust, w("exhaust pipe"))
    assert not pipeline._names_part(exhaust, w("spring"))
    assert pipeline._names_part(seat, w("saddle")) and not pipeline._names_part(seat, w("motorcycle"))
    assert pipeline._names_part(spring, w("shock absorber")) and pipeline._names_part(spring, w("coil spring"))
    assert pipeline._names_part(spring, w("springs")) and not pipeline._names_part(spring, w("wheel rim"))


def test_a_prompt_whose_client_went_away_stops_waiting_for_the_gpu(env, monkeypatch):
    import time as _time
    monkeypatch.setattr(pipeline, "PROMPT_GPU_WAIT_S", 1.0)
    held, done = threading.Event(), threading.Event()

    def hold():
        with pipeline.gpu_lock:
            held.set()
            done.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    held.wait(5)
    try:
        gone = {"v": False}
        threading.Timer(0.25, lambda: gone.update(v=True)).start()
        t0 = _time.perf_counter()
        with pytest.raises(pipeline.PromptCancelled):
            with pipeline._prompt_gpu("Select part", lambda: gone["v"]):
                pass
        assert 0.2 < _time.perf_counter() - t0 < 0.6
        t0 = _time.perf_counter()
        with pytest.raises(pipeline.PipelineError) as e:
            with pipeline._prompt_gpu("Select part", lambda: False):
                pass
        assert e.value.status == 503 and 0.9 < _time.perf_counter() - t0 < 1.4
    finally:
        done.set()
        t.join()
    ran = []
    with pipeline._prompt_gpu("Select part", lambda: ran.append(1) or True):   # uncontended: no check, it runs
        pass
    assert ran == []


def test_a_deleted_jobs_embedding_is_never_kept(client, env):
    job = env["make_job"]()
    with job.lock:
        job.deleted = True                                       # deleted while this prompt ran
    with pytest.raises(pipeline.PipelineError) as e:
        pipeline._prompt_session(job)
    assert e.value.status == 404 and job.id not in sam_masks.SamMasker.prompt_keys()


def test_remove_gives_every_carved_pixel_back_to_its_own_region(client, env, monkeypatch):
    """Part A (the caliper, cut from the frame), then part B over 80 % of A and a corner of the tank:
    Remove B gives the tank its pixels back and A all of its own, still a part (B's cut went whole to
    A, the region that gave the most, and as the colour region it mostly was it dissolved A)."""
    job = env["make_job"]()
    a = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]], "name": "Caliper"}).json()
    ga = next(g for g in a["groups"] if g["id"] == a["created_group"])
    labels1 = np.load(job.path("labels.npy"))
    A = np.isin(labels1, ga["region_ids"])
    tank_px = int((labels1 == 3).sum())
    ys, xs = np.nonzero(A)
    m = A.copy()
    m[:, : int(np.percentile(xs, 20))] = False                         # 80 % of A ...
    m[40:50, 150:170] = True                                            # ... and a corner of the tank
    _answer_with(monkeypatch, m)
    b = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[140, 80, 1]], "name": "Cut"}).json()
    gb = next(g for g in b["groups"] if g["id"] == b["created_group"])
    assert b["created_part"]["took_in"] == [] and _gid(b["groups"], ga["region_ids"][0])["part"] == "user_1"
    j = client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [gb["id"]], "dissolve": True}).json()
    labels2 = np.load(job.path("labels.npy"))
    ga2 = next(g for g in j["groups"] if g["part"] == "user_1")
    assert ga2["name"] == "Caliper" and (np.isin(labels2, ga2["region_ids"]) == A).all()      # A whole again
    assert int((labels2 == 3).sum()) == tank_px and _gid(j["groups"], 3)["area"] == _gid(a["groups"], 3)["area"]
    assert not any(g["part"] == "user_2" for g in j["groups"])
    rid = pipeline.decode_region_ids(imageio.load_image(job.path("ids", "regions.png")))
    assert (rid == labels2).all() and set(np.unique(labels2).tolist()) == set(range(int(labels2.max()) + 1))
    _, regions = _state(job)
    assert [r["area"] for r in regions] == np.bincount(labels2.ravel()).tolist()
    seed = pipeline._load_seed(job, labels2.shape)
    twin = [i for i in ga2["region_ids"] if i not in ga["region_ids"]]
    assert len(twin) == 1 and seed.origin[twin[0]] == seed.origin[ga["region_ids"][0]]
    assert not os.path.exists(job.path("parts", "user_2.npz")) and os.path.exists(job.path("parts", "user_1.npz"))
    r = client.post(f"/api/jobs/{job.id}/regroup", json={"delta_e": job.options.delta_e}).json()
    assert [sorted(g["region_ids"]) for g in r["groups"] if g["part"] == "user_1"] == [sorted(ga2["region_ids"])]


def test_a_region_the_mask_covers_whole_joins_whole_even_its_specks():
    """A group told to join (Take all) whose region has a 1 px speck apart from the rest: the speck,
    an island of the mask, was dropped, so the region was cut and the speck stayed behind as it."""
    labels = np.zeros((40, 60), np.int32)
    labels[5:15, 5:15] = 1
    labels[30, 50] = 1                                         # a 1 px speck of region 1
    labels[20:30, 20:30] = 2
    mask = labels == 1
    mask[20:22, 20:30] = True                                  # and a strip of region 2
    c = userparts.carve(labels, mask)
    assert c.covered == [1] and c.new_id == 3 and c.donors == {2: 20}
    assert (c.labels[labels == 1] == 1).all()
    stray = np.zeros_like(mask)
    stray[5:15, 5:15] = True
    stray[35, 5] = True                                        # an island of the mask on region 0: dropped
    c2 = userparts.carve(labels, stray)
    assert c2.labels[35, 5] == 0


def test_an_edit_leaves_the_display_layers_to_be_drawn_when_asked(client, env):
    """A commit writes the label map and the id PNGs the studio reads at once; the display layers
    (Regions, Groups and the edges) are drawn when a tab asks for them, from the new maps."""
    job = env["make_job"]()
    assert os.path.exists(job.path("layers", "regions.png")) and os.path.exists(job.path("layers", "groups.png"))
    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).json()
    for name in ("regions", "edges", "groups"):
        assert not os.path.exists(job.path("layers", f"{name}.png")), name
    labels = np.load(job.path("labels.npy"))
    gm = np.load(job.path("group_map.npy"))
    for name, want in (("regions", pipeline.regions_display(labels)), ("groups", pipeline.groups_display(gm, job.groups())),
                       ("edges", pipeline.edges_display(imageio.load_image(job.path("work.png")), labels))):
        r = client.get(f"/api/jobs/{job.id}/layers/{name}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png", name
        import io
        from PIL import Image
        assert (np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB")) == want).all(), name
        assert os.path.exists(job.path("layers", f"{name}.png"))
    part = next(g for g in out["groups"] if g["id"] == out["created_group"])
    client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [part["id"]], "dissolve": True})
    assert not os.path.exists(job.path("layers", "groups.png"))
    assert client.get(f"/api/jobs/{job.id}/layers/groups?w=64").status_code == 200


def test_the_layers_an_edit_leaves_in_the_cache_are_the_files_it_wrote(client, env, monkeypatch):
    job = env["make_job"](two_springs=True)

    def same():
        cached = pipeline.load_layers(job)
        assert job.id in pipeline._layers_cache
        for name in ("labels", "group_map"):
            assert (cached[name] == np.load(job.path(f"{name}.npy"))).all(), name
        assert (cached["protect"] == np.load(job.path("protect.npy")).astype(bool)).all()
        assert (cached["islands"] == np.load(job.path("islands.npy")).astype(bool)).all()

    out = client.post(f"/api/jobs/{job.id}/groups/from_mask", json={"points": [[*CALIPER, 1]]}).json()
    same()
    springs = next(g for g in out["groups"] if g["part"] == "shock_spring")
    j = client.post(f"/api/jobs/{job.id}/groups/split", json={"group_id": springs["id"], "mode": "instances"}).json()
    same()
    tank = _gid(j["groups"], 3)
    client.post(f"/api/jobs/{job.id}/groups/split", json={"group_id": tank["id"], "k": 2})
    same()
    part = next(g for g in job.snapshot()["groups"] if g["part"] == "user_1")
    client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [part["id"]], "dissolve": True})
    same()
    client.post(f"/api/jobs/{job.id}/regroup", json={"delta_e": job.options.delta_e})
    same()
    g0, g1 = job.groups()[0].id, job.groups()[1].id
    client.post(f"/api/jobs/{job.id}/groups/merge", json={"group_ids": [g0, g1]})
    same()
