"""Detected parts (recolor/segmentation/smallparts.py: vocabulary, detector gates, SAM
masks, stamping; partdetect.py: the OWLv2 wrapper; florence.caption) on synthetic images
with the detector, SAM and the models replaced by stand-ins: no model, no network."""
from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from recolor import imageio
from recolor.segmentation import florence, partdetect, smallparts
from recolor.segmentation.labelops import region_areas
from recolor.segmentation.smallparts import Kind, PartGates, PartMask


def _lab(h, w, base, paint=()):
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = base
    for sl, col in paint:
        lab[sl] = col
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    return lab, albedo, imageio.to_uint8(imageio.linear_to_srgb(albedo * 0.9))


def _box_of(m):
    ys, xs = np.nonzero(m)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _cand(mask, crop, score=0.95, clipped=False):
    x0, y0, x1, y1 = crop
    return {"mask": mask[y0:y1, x0:x1].copy(), "x0": x0, "y0": y0, "score": score, "clipped": clipped}


SPRING = Kind("shock_spring", "Shock spring", "Shock springs", ("coil spring", "shock absorber"), 0.2, max_instances=2)
GRIP = Kind("grip", "Grip", "Grips", ("handlebar grip",), 0.1, max_instances=2)


# ------------------------------------------------------------------ vocabulary

def test_the_caption_picks_the_class_whose_word_comes_first():
    assert smallparts.object_class("A red motorcycle is parked in a dark room.") == "motorcycle"
    assert smallparts.object_class("A man riding a bicycle next to a car") == "bicycle"
    assert smallparts.object_class("A model of a gundam with a gun and shield") == "figure"
    assert smallparts.object_class("A pair of white sneakers") == "sneaker"
    assert smallparts.object_class("A large room filled with lots of different colored cars.") == "car"
    assert smallparts.object_class("a teapot") == "generic" and smallparts.object_class(None) == "generic"
    moto = smallparts.kinds_for("motorcycle")
    assert all(k.tier == "accessory" for k in moto) and {"shock_spring", "grip", "rim", "tyre"} <= {k.key for k in moto}
    assert "tank" not in {k.key for k in moto} and "tank" in {k.key for k in smallparts.kinds_for("motorcycle", ("panel",))}
    assert smallparts.kinds_for("unknown class") == smallparts.kinds_for("generic")


# ------------------------------------------------------------------ detector gates -> SAM jobs

def test_part_jobs_keep_the_confident_boxes_once_per_part():
    dets = [
        {"box": [10, 10, 40, 30], "phrase": "coil spring", "score": 0.48, "det": "owlv2"},
        {"box": [11, 10, 41, 31], "phrase": "shock absorber", "score": 0.40, "det": "owlv2"},   # the same spring
        {"box": [15, 12, 30, 25], "phrase": "coil spring", "score": 0.35, "det": "owlv2"},      # inside the first box
        {"box": [60, 10, 80, 30], "phrase": "coil spring", "score": 0.25, "det": "owlv2"},      # below the gate
        {"box": [60, 40, 80, 60], "phrase": "handlebar grip", "score": 0.9, "det": "gdino"},    # an unknown detector
        {"box": [0, 0, 200, 100], "phrase": "handlebar grip", "score": 0.9, "det": "owlv2"},    # far too big for a grip
        {"box": [100, 60, 120, 80], "phrase": "handlebar grip", "score": 0.41, "det": "owlv2"},
        {"box": [100, 60, 120, 80], "phrase": "front fender", "score": 0.9, "det": "owlv2"},    # not in the vocabulary
    ]
    jobs = smallparts.part_jobs(dets, (SPRING, GRIP), (100, 200), object_px=2000)
    assert [(j["kind"], j["score"]) for j in jobs] == [("shock_spring", 0.48), ("grip", 0.41)]
    spring = jobs[0]
    assert spring["crop"][0] < 10 and spring["crop"][2] > 40 and spring["crop"][1] >= 0      # grown by 8 % + 8 px


def test_wheel_boxes_are_pooled_and_the_prompts_per_kind_capped():
    rim = Kind("rim", "Wheel rim", "Wheel rims", ("wheel rim",), 0.5, wheel="rim")
    tyre = Kind("tyre", "Tyre", "Tyres", ("tire",), 0.5, wheel="tyre")
    dets = [{"box": [0, 0, 50, 50], "phrase": "tire", "score": 0.8, "det": "owlv2"},
            {"box": [2, 2, 50, 50], "phrase": "wheel rim", "score": 0.7, "det": "owlv2"}]
    jobs = smallparts.part_jobs(dets, (rim, tyre), (100, 200), object_px=10000)
    assert [j["kind"] for j in jobs] == ["wheel"]                     # one wheel prompt for tyre and rim
    many = [{"box": [i * 12, 0, i * 12 + 10, 10], "phrase": "coil spring", "score": 0.9 - i * 0.01, "det": "owlv2"}
            for i in range(15)]
    assert len(smallparts.part_jobs(many, (SPRING,), (100, 200), 5000, PartGates(max_jobs_per_kind=5))) == 5


# ------------------------------------------------------------------ SAM masks -> gated parts

def _spring_scene():
    h, w = 100, 160
    spring = np.zeros((h, w), bool)
    spring[20:35, 30:42] = True                                       # 180 px: within 2 % of the 80 x 140 object
    lab, albedo, photo = _lab(h, w, (30.0, 0.0, 0.0), [(spring, (80.0, 5.0, 70.0))])
    fg = np.zeros((h, w), np.float32)
    fg[10:90, 10:150] = 1.0
    return spring, lab, albedo, photo, fg


def test_select_parts_applies_the_sam_and_shape_gates():
    spring, lab, albedo, photo, fg = _spring_scene()
    job = {"kind": "shock_spring", "box": _box_of(spring), "score": 0.5, "phrase": "coil spring", "det": "owlv2",
           "votes": 1, "crop": (20, 10, 52, 45)}
    blob = np.zeros_like(spring)
    blob[5:95, 5:155] = True                                          # SAM answered with the whole bike
    cands = [[_cand(spring, job["crop"], score=0.80),                  # below the SAM gate
              _cand(blob, job["crop"], score=0.99),                    # does not agree with the box
              _cand(spring, job["crop"], score=0.95, clipped=True),    # clipped by the crop
              _cand(spring, job["crop"], score=0.93)]]                 # the one
    log_out: list = []
    parts = smallparts.select_parts(photo, lab, fg, [job], cands, (SPRING,), log_out=log_out)
    assert len(parts) == 1 and parts[0].kind == "shock_spring" and parts[0].area == int(spring.sum())
    assert parts[0].sam_score == pytest.approx(0.93) and log_out[0]["result"].startswith("accepted")
    off = np.zeros_like(fg)                                            # the same part off the matte
    assert smallparts.select_parts(photo, lab, off, [job], cands, (SPRING,)) == []
    tiny = Kind("shock_spring", "Shock spring", "Shock springs", ("coil spring",), 0.01)     # at most 112 px
    assert smallparts.select_parts(photo, lab, fg, [job], cands, (tiny,)) == []   # larger than its kind allows


def test_duplicates_and_the_instance_cap():
    h, w = 60, 200
    masks = []
    for x in (10, 60, 110, 160):
        m = np.zeros((h, w), bool)
        m[10:40, x:x + 20] = True
        masks.append(m)
    parts = [PartMask("grip", "Grip", "Grips", masks[i], 0.9 - 0.1 * i, 0.9, _box_of(masks[i]), "handlebar grip", "owlv2")
             for i in range(4)]
    dup = PartMask("grip", "Grip", "Grips", masks[0].copy(), 0.3, 0.9, _box_of(masks[0]), "handlebar grip", "owlv2")
    other = PartMask("shock_spring", "Shock spring", "Shock springs", masks[1].copy(), 0.2, 0.9, _box_of(masks[1]),
                     "coil spring", "owlv2")
    kept = smallparts.dedup_parts(parts + [dup, other], {"grip": GRIP, "shock_spring": SPRING})
    assert [p.score for p in kept] == [0.9, pytest.approx(0.8)]        # max 2 grips; the duplicate and the look-alike go


def test_find_kind_parts_asks_every_tier_and_keeps_the_accessories():
    spring, lab, albedo, photo, fg = _spring_scene()
    asked = []

    def detector(image, phrases):
        asked.append(list(phrases))
        return [{"box": _box_of(spring), "phrase": "coil spring", "score": 0.6, "det": "owlv2"},
                {"box": [80, 20, 140, 60], "phrase": "fuel tank", "score": 0.9, "det": "owlv2"}]

    def prompter(image, jobs):
        return [[_cand(spring, j["crop"], 0.95)] for j in jobs]

    parts, rep = smallparts.find_kind_parts(photo, albedo, fg, "A red motorcycle in a studio", detector, prompter)
    assert "fuel tank" in asked[0] and "coil spring" in asked[0]      # the panel tier competes for the boxes ...
    assert [p.kind for p in parts] == ["shock_spring"] and rep["class"] == "motorcycle"   # ... but only accessories become parts
    assert smallparts.find_kind_parts(photo, albedo, fg, "a motorcycle", lambda i, p: None, prompter)[0] == []
    # no time left: nothing is prompted, nothing is found
    parts, _ = smallparts.find_kind_parts(photo, albedo, fg, "a motorcycle", detector, prompter, budget_s=0.0)
    assert parts == []


# ------------------------------------------------------------------ stamping

def _check(labels, info):
    assert labels.dtype == np.int32 and labels.min() == 0
    assert int(labels.max()) + 1 == len(info) and [d["id"] for d in info] == list(range(len(info)))
    assert (region_areas(labels, len(info)) > 0).all()


def test_a_part_cuts_its_host_and_keeps_its_neighbours_and_lettering():
    h, w = 100, 160
    labels = np.zeros((h, w), np.int32)                  # 0: the frame (a large host)
    labels[40:60, 90:120] = 1                            # 1: a neighbour the mask only nicks
    labels[24:30, 32:40] = 2                             # 2: lettering on the spring
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (70.0, 5.0, 60.0)
    lab[labels == 1] = (40.0, 0.0, 0.0)
    lab[labels == 2] = (95.0, 0.0, 0.0)
    info = [{"id": 0, "source": "sam"}, {"id": 1, "source": "sam"}, {"id": 2, "source": "text"}]
    mask = np.zeros((h, w), bool)
    mask[20:60, 30:92] = True                            # the spring, overlapping 2 columns of region 1
    part = PartMask("shock_spring", "Shock spring", "Shock springs", mask, 0.5, 0.95, _box_of(mask), "coil spring", "owlv2")
    out, out_info, rep = smallparts.stamp_parts(labels, info, [part], lab)
    _check(out, out_info)
    kinds = [d for d in out_info if d.get("part_kind")]
    assert len(kinds) == 1 and kinds[0]["source"] == smallparts.SOURCE_KIND and kinds[0]["part_instance"] == 0
    assert kinds[0]["part_label"] == "Shock spring" and kinds[0]["part_plural"] == "Shock springs"
    rid = kinds[0]["id"]
    assert (out[mask & (labels == 0)] == rid).all()                    # cut out of the frame
    assert (out[labels == 1] == out[45, 100]).all() and int(out[45, 100]) != rid   # the nicked neighbour stays whole
    assert (out[labels == 2] != rid).all()                             # the lettering keeps its pixels
    assert rep["stamped"][0]["kind"] == "shock_spring"


def test_a_region_that_already_is_the_part_is_adopted_whole():
    h, w = 80, 120
    labels = np.zeros((h, w), np.int32)
    labels[20:50, 30:60] = 1
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (50.0, 0.0, 0.0)
    info = [{"id": 0, "source": "sam"}, {"id": 1, "source": "small", "exempt": True}]
    mask = np.zeros((h, w), bool)
    mask[21:50, 30:60] = True                                          # IoU 0.97 with region 1
    part = PartMask("grip", "Grip", "Grips", mask, 0.4, 0.9, _box_of(mask), "handlebar grip", "owlv2")
    out, out_info, rep = smallparts.stamp_parts(labels, info, [part], lab)
    assert np.array_equal(out, labels)                                 # pixels unchanged
    assert out_info[1]["part_kind"] == "grip" and out_info[1]["adopted_from"] == "small"
    assert out_info[1]["source"] == smallparts.SOURCE_KIND and rep["adopted"][0]["region"] == 1


def test_a_distinct_sub_part_stays_and_the_remnant_ring_joins_the_part():
    h, w = 100, 100
    labels = np.zeros((h, w), np.int32)
    labels[20:60, 18:42] = 1                             # the rim's left half in SAM's automatic masks, 2 px wider ...
    labels[20:60, 42:62] = 3                             # ... and its right half, 2 px wider on the other side
    labels[35:45, 25:35] = 2                             # a red caliper on the rim
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (20.0, 0.0, 0.0)
    lab[(labels == 1) | (labels == 3)] = (70.0, 2.0, 2.0)
    lab[labels == 2] = (45.0, 60.0, 40.0)
    info = [{"id": i, "source": "sam"} for i in range(4)]
    mask = np.zeros((h, w), bool)
    mask[20:60, 20:60] = True                            # the box prompt's mask: neither half is the part (IoU < 0.6)
    part = PartMask("rim", "Wheel rim", "Wheel rims", mask, 0.6, 0.95, _box_of(mask), "wheel rim", "owlv2")
    out, out_info, rep = smallparts.stamp_parts(labels, info, [part], lab)
    _check(out, out_info)
    rid = next(d["id"] for d in out_info if d.get("part_kind") == "rim")
    assert (out[35:45, 25:35] != rid).all() and len(np.unique(out[35:45, 25:35])) == 1   # the caliper is kept
    assert rep["kept_subparts"][0]["why"] == "sub-part"
    assert (out[20:60, 18:20] == rid).all() and (out[20:60, 60:62] == rid).all()        # the rings are the rim's
    assert rep["remnants_merged"] == 2 and len(out_info) == 3          # background, caliper, rim


def test_a_region_reaching_beyond_the_part_is_cut_and_its_cored_rest_stays_apart():
    """The BMW's front-wheel region held the fork stanchion in front of the tyre: adopted whole
    (IoU 0.79), the tyre part painted the fork. A region reaching more than a tenth beyond a
    wheel part's mask is cut to it, and its rest, a piece with a core, stays a region of its own;
    another kind (a spring, whose region is often the better outline) adopts it whole."""
    h, w = 100, 160
    labels = np.zeros((h, w), np.int32)
    labels[10:90, 20:100] = 1                            # the tyre region (6400 px) ...
    labels[10:90, 100:118] = 1                           # ... with the fork stanchion (1440 px) beside it
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (80.0, 0.0, 0.0)
    lab[labels == 1] = (21.0, 0.0, 3.0)
    info = [{"id": 0, "source": "sam"}, {"id": 1, "source": "wheel"}]
    mask = np.zeros((h, w), bool)
    mask[10:90, 20:100] = True                           # SAM's tyre: IoU 0.82 with region 1, 82 % of it
    part = PartMask("tyre", "Tyre", "Tyres", mask, 0.4, 0.9, _box_of(mask), "tire", "owlv2")
    out, out_info, rep = smallparts.stamp_parts(labels, info, [part], lab)
    _check(out, out_info)
    assert rep["adopted"] == []
    rid = next(d["id"] for d in out_info if d.get("part_kind") == "tyre")
    assert (out[mask] == rid).all() and not (out[10:90, 100:118] == rid).any()          # the fork stays apart
    assert rep["remnants_merged"] == 0
    spring = PartMask("shock_spring", "Shock spring", "Shock springs", mask, 0.4, 0.9, _box_of(mask), "coil spring", "owlv2")
    out, out_info, rep = smallparts.stamp_parts(labels, info, [spring], lab)
    assert rep["adopted"] and np.array_equal(out, labels)


def test_an_exact_mask_adopts_every_piece_of_its_part():
    """The Corvette's caliper seen above and below a spoke: two regions of the partition, one
    mask from them (the wheel look's painted route). Both go into the part."""
    h, w = 80, 80
    labels = np.zeros((h, w), np.int32)
    labels[10:30, 20:40] = 1                             # the upper piece (400 px)
    labels[40:48, 22:36] = 2                             # the lower piece (112 px), below the spoke
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (35.0, 0.0, -5.0)
    lab[labels == 1] = (78.0, -11.0, 69.0)
    lab[labels == 2] = (67.0, -5.0, 63.0)                # a shade darker: dE 9
    info = [{"id": i, "source": "sam"} for i in range(3)]
    mask = (labels == 1) | (labels == 2)
    exact = PartMask("brake_caliper", "Brake caliper", "Brake calipers", mask, 1.0, 1.0, _box_of(mask), "brake caliper",
                     "wheel-colour", exact=True)
    out, out_info, rep = smallparts.stamp_parts(labels, info, [exact], lab)
    rid = next(d["id"] for d in out_info if d.get("part_kind"))
    assert (out[mask] == rid).all() and rep["coadopted"][0]["px"] == 112
    sam = PartMask("brake_caliper", "Brake caliper", "Brake calipers", mask, 1.0, 1.0, _box_of(mask), "brake caliper",
                   "owlv2")
    out, out_info, rep = smallparts.stamp_parts(labels, info, [sam], lab)
    rid = next(d["id"] for d in out_info if d.get("part_kind"))
    assert not (out[labels == 2] == rid).any() and "coadopted" not in rep              # a SAM mask adopts one


def test_a_mirror_takes_the_stalk_the_detector_called_something_else():
    h, w = 80, 120
    head = np.zeros((h, w), bool)
    head[20:45, 40:80] = True                            # the mirror head (1000 px)
    stalk = np.zeros((h, w), bool)
    stalk[45:55, 55:75] = True                           # the stalk under it (200 px), called a spoiler
    grip = np.zeros((h, w), bool)
    grip[45:50, 80:84] = True                            # a grip beside it: never folded
    far = np.zeros((h, w), bool)
    far[60:70, 0:12] = True                              # a badge far away
    parts = [PartMask(k, k.title(), k.title() + "s", m, 0.4, 0.9, _box_of(m), k, "owlv2")
             for k, m in (("mirror", head), ("spoiler", stalk), ("grip", grip), ("badge", far))]
    log = []
    out = smallparts.attach_parts(parts, log_out=log)
    assert [p.kind for p in out] == ["mirror", "grip", "badge"]
    assert out[0].area == 1200 and log[0]["result"] == "attached to mirror"


def test_two_instances_of_a_kind_are_numbered_largest_first():
    h, w = 60, 160
    labels = np.zeros((h, w), np.int32)
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (40.0, 0.0, 0.0)
    a = np.zeros((h, w), bool)
    a[10:30, 10:30] = True
    b = np.zeros((h, w), bool)
    b[10:40, 100:130] = True
    parts = [PartMask("mirror", "Mirror", "Mirrors", m, 0.5, 0.9, _box_of(m), "side mirror", "owlv2") for m in (a, b)]
    out, out_info, _ = smallparts.stamp_parts(labels, [{"id": 0, "source": "sam"}], parts, lab)
    tags = {d["part_instance"]: d["id"] for d in out_info if d.get("part_kind")}
    assert set(tags) == {0, 1} and int(out[20, 110]) == tags[0] and int(out[20, 20]) == tags[1]
    assert smallparts.part_info(out_info)[tags[0]] == {"kind": "mirror", "label": "Mirror", "plural": "Mirrors", "instance": 0}


# ------------------------------------------------------------------ the model wrappers

class _Model:
    def to(self, device):
        return self

    def eval(self):
        return self


def _fake_transformers(calls, present):
    def from_pretrained(name):
        def f(model_id, **kw):
            calls.append((name, dict(kw)))
            if not kw.get("local_files_only"):
                raise AssertionError("tried the network")
            if not present["ok"]:
                raise OSError("not in the cache")
            return _Model()
        return staticmethod(f)

    fake = types.ModuleType("transformers")
    fake.Owlv2Processor = type("P", (), {"from_pretrained": from_pretrained("processor")})
    fake.Owlv2ForObjectDetection = type("O", (), {"from_pretrained": from_pretrained("owlv2")})
    return fake


@pytest.fixture
def fresh_owl(monkeypatch):
    monkeypatch.setattr(partdetect, "_snapshot_cached", lambda: True)    # the fake transformers decide
    monkeypatch.setattr(partdetect, "_runner", None)
    monkeypatch.setattr(partdetect, "_state", "cold")
    monkeypatch.setattr(partdetect, "_weights_missing", False)
    monkeypatch.setattr(partdetect, "_unavailable_reason", None)


def test_owlv2_weights_come_from_the_local_cache_only(monkeypatch, caplog, fresh_owl):
    calls, present = [], {"ok": False}
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls, present))
    img = np.zeros((40, 60, 3), np.uint8)
    with caplog.at_level(logging.WARNING, logger=partdetect.log.name):
        assert partdetect.detect(img, ["coil spring"]) is None and partdetect.status() == "unavailable"
        assert partdetect.detect(img, ["coil spring"]) is None
    assert sum("unavailable" in r.getMessage() for r in caplog.records) == 1
    assert "setup.sh" in partdetect._unavailable_reason
    assert all(kw["local_files_only"] and kw["revision"] == partdetect.MODEL_REVISION for _, kw in calls)
    assert partdetect.detect(img, []) == []                            # nothing to ask, nothing loaded
    present["ok"] = True
    assert partdetect._get_runner() is not None and partdetect.status() == "ready" and partdetect.is_loaded()
    partdetect.release()
    assert not partdetect.is_loaded() and partdetect.status() == "cold"


class OutOfMemoryError(RuntimeError):
    pass


def test_owlv2_boxes_are_mapped_back_to_the_image_and_errors_fall_back(monkeypatch, fresh_owl):
    import torch

    class Proc:
        def __call__(self, text, images, return_tensors):
            return {"input_ids": torch.zeros((1, 2), dtype=torch.long), "pixel_values": torch.zeros((1, 3, 4, 4))}

    class Model:
        def __call__(self, **kw):
            # two boxes: one "grip" at the crop's centre, one below the score floor
            logits = torch.tensor([[[2.0, -9.0], [-9.0, -6.0]]])
            boxes = torch.tensor([[[0.25, 0.25, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]]])
            return types.SimpleNamespace(logits=logits, pred_boxes=boxes)

    owl = partdetect._Owl.__new__(partdetect._Owl)
    owl.torch, owl.device, owl.dtype, owl.processor, owl.model = torch, torch.device("cpu"), torch.float32, Proc(), Model()
    monkeypatch.setattr(partdetect, "_runner", owl)
    monkeypatch.setattr(partdetect, "_state", "ready")
    img = np.zeros((100, 200, 3), np.uint8)
    out = partdetect.detect(img, ["handlebar grip", "coil spring"])
    full = [d for d in out if d["src"] == "full"]
    assert len(out) == 5 and all(d["phrase"] == "handlebar grip" and d["det"] == partdetect.NAME for d in out)
    assert full[0]["box"] == pytest.approx([40.0, 40.0, 60.0, 60.0])   # relative to the padded 200 px square
    assert all(0.8 < d["score"] < 0.9 for d in out)

    def oom(**kw):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")

    owl.model = oom
    assert partdetect.detect(img, ["handlebar grip"]) is None
    assert not partdetect.is_loaded() and partdetect.status() == "cold"


def test_the_caption_comes_from_florence_or_is_none(monkeypatch):
    class Runner:
        def run(self, images, task, text=None):
            assert task == "<CAPTION>" and len(images) == 1
            return ["A red motorcycle is parked in a studio.</s>"]

    monkeypatch.setattr(florence, "_runner", Runner())
    monkeypatch.setattr(florence, "_state", "ready")
    assert florence.caption(np.zeros((20, 30, 3), np.uint8)) == "A red motorcycle is parked in a studio."
    broken = Runner()
    broken.run = lambda images, task, text=None: (_ for _ in ()).throw(RuntimeError("bad"))
    monkeypatch.setattr(florence, "_runner", broken)
    assert florence.caption(np.zeros((20, 30, 3), np.uint8)) is None
    with pytest.raises(ValueError):
        florence.caption(np.zeros((20, 30), np.uint8))


# ------------------------------------------------------------------ the wheel second look (brake calipers)

CALIPER = Kind("brake_caliper", "Brake caliper", "Brake calipers", ("brake caliper",), 0.08, max_instances=4)


def _wheel_scene(caliper_colour=(40.0, 10.0, -50.0), own_region=True, band=None):
    """A wheel on a grey backdrop: a dark tyre ring (1), a grey rim (2) and, off the hub, a small
    caliper (3, 12 x 18 px; its own region when ``own_region``, else part of the rim). ``band``
    paints a strip of that colour just outside the wheel (a fender edge)."""
    h, w = 220, 220
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - 110, xx - 110)
    labels = np.zeros((h, w), np.int32)
    labels[r < 90] = 1
    labels[r < 68] = 2
    cal = np.zeros((h, w), bool)
    cal[95:113, 150:162] = True                                      # radius ~0.52 of the wheel
    if own_region:
        labels[cal] = 3
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (70.0, 0.0, 0.0)
    lab[labels == 1] = (15.0, 0.0, 0.0)
    lab[labels == 2] = (55.0, 1.0, 2.0)
    lab[cal] = caliper_colour
    if band is not None:
        labels[5:15, 60:160] = 4
        lab[5:15, 60:160] = band
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    photo = imageio.to_uint8(imageio.linear_to_srgb(albedo * 0.9))
    fg = (r < 90).astype(np.float32)
    wheel = {"box": [20.0, 20.0, 200.0, 200.0], "mask": r < 90, "split": True}
    info = [{"id": i, "source": "sam"} for i in range(int(labels.max()) + 1)]
    return photo, lab, labels, info, fg, wheel, cal


def _zoom(truths, calls=None):
    """A stand-in for partdetect.detect_in: every truth box (box, phrase, score_wide, score_zoomed)
    that lies inside a window is returned for it, with the zoomed score in a small window."""
    def zoom(image, crops, phrases):
        if calls is not None:
            calls.append(list(crops))
        out = []
        for k, c in enumerate(crops):
            small = (c[2] - c[0]) * (c[3] - c[1]) < 0.3 * image.shape[0] * image.shape[1]
            for box, phrase, s_wide, s_zoom in truths:
                if box[0] >= c[0] and box[1] >= c[1] and box[2] <= c[2] and box[3] <= c[3]:
                    s = s_zoom if small else s_wide
                    out.append({"box": list(box), "phrase": phrase, "score": s, "scores": [s], "src": f"crop{k}",
                                "det": "owlv2"})
        return out
    return zoom


def test_a_painted_caliper_is_the_one_colour_in_the_wheel():
    photo, lab, labels, info, fg, wheel, cal = _wheel_scene()
    out = smallparts.find_calipers(photo, lab, fg, labels, info, [wheel], _zoom([]), None, CALIPER, int(fg.sum()))
    assert len(out) == 1 and out[0].kind == "brake_caliper" and out[0].det == "wheel-colour"
    assert np.array_equal(out[0].mask, cal) and out[0].label == "Brake caliper"


def test_a_piece_of_a_fender_seen_inside_the_wheel_is_no_caliper():
    # the caliper's blue continues just outside the wheel: a fender's edge, not a caliper
    photo, lab, labels, info, fg, wheel, cal = _wheel_scene(band=(40.0, 10.0, -50.0))
    log: list = []
    out = smallparts.find_calipers(photo, lab, fg, labels, info, [wheel], _zoom([]), None, CALIPER, int(fg.sum()),
                                   log_out=log)
    assert out == [] and log[-1]["result"] == "no caliper"


def test_a_caliper_owlv2_names_on_a_tight_crop_wins_over_a_painted_piece():
    # a rim-coloured caliper (not painted) that OWLv2 names at 0.30 on its tight crop (0.12 wide)
    photo, lab, labels, info, fg, wheel, cal = _wheel_scene(caliper_colour=(58.0, 2.0, 6.0))
    box = (150.0, 95.0, 162.0, 113.0)
    out = smallparts.find_calipers(photo, lab, fg, labels, info, [wheel], _zoom([(box, "brake caliper", 0.12, 0.30)]),
                                   None, CALIPER, int(fg.sum()))
    assert len(out) == 1 and out[0].det == "owlv2-zoom" and out[0].score == pytest.approx(0.30)
    assert np.array_equal(out[0].mask, cal)
    # the same box named a lug nut, or a caliper below the gate: nothing
    for truth in ((box, "lug nut", 0.3, 0.9), (box, "brake caliper", 0.12, 0.18)):
        assert smallparts.find_calipers(photo, lab, fg, labels, info, [wheel], _zoom([truth]), None, CALIPER,
                                        int(fg.sum())) == []


def test_a_caliper_inside_the_rim_region_gets_its_mask_from_sam():
    # the black caliper is part of the black rim's region: OWLv2's box, then SAM
    photo, lab, labels, info, fg, wheel, cal = _wheel_scene(caliper_colour=(56.0, 1.0, 2.0), own_region=False)
    box = (150.0, 95.0, 162.0, 113.0)
    seen = []

    def prompter(image, jobs):
        seen.extend(jobs)
        return [[_cand(cal, j["crop"], 0.93)] for j in jobs]

    out = smallparts.find_calipers(photo, lab, fg, labels, info, [wheel], _zoom([(box, "brake caliper", 0.15, 0.33)]),
                                   prompter, CALIPER, int(fg.sum()))
    assert len(seen) == 1 and len(out) == 1 and out[0].sam_score == pytest.approx(0.93)
    assert np.array_equal(out[0].mask, cal)


def test_no_detector_no_caliper_and_find_kind_parts_runs_the_look():
    photo, lab, labels, info, fg, wheel, cal = _wheel_scene()
    assert smallparts.find_calipers(photo, lab, fg, labels, info, [wheel], lambda i, c, p: None, None, CALIPER,
                                    int(fg.sum())) == []
    albedo = np.clip(imageio.lab_to_linear(lab), 0.0, 1.0).astype(np.float32)
    rim = Kind("rim", "Wheel rim", "Wheel rims", ("wheel rim",), 0.9, wheel="rim")
    tyre = Kind("tyre", "Tyre", "Tyres", ("tire",), 0.9, wheel="tyre")

    def detector(image, phrases):
        return [{"box": wheel["box"], "phrase": "tire", "score": 0.6, "det": "owlv2"}]

    def prompter(image, jobs):
        return [[_cand(wheel["mask"], j["crop"], 0.95)] for j in jobs]

    import unittest.mock as um
    with um.patch.object(smallparts, "kinds_for", lambda cls, tiers=("accessory",): (rim, tyre, CALIPER)):
        parts, rep = smallparts.find_kind_parts(photo, albedo, fg, "a car", detector, prompter, labels=labels,
                                                info=info, zoom=_zoom([]))
        assert rep.get("calipers") == 1 and any(p.kind == "brake_caliper" for p in parts)
        parts, rep = smallparts.find_kind_parts(photo, albedo, fg, "a car", detector, prompter, labels=labels,
                                                info=info, zoom=_zoom([]), look=None)
        assert "calipers" not in rep and not any(p.kind == "brake_caliper" for p in parts)


def test_owlv2_detect_in_scores_every_phrase_on_each_window(monkeypatch, fresh_owl):
    import torch

    class Proc:
        def __call__(self, text, images, return_tensors):
            return {"input_ids": torch.zeros((1, 2), dtype=torch.long), "pixel_values": torch.zeros((1, 3, 4, 4))}

    class Model:
        def __call__(self, **kw):
            logits = torch.tensor([[[2.0, -1.0], [-9.0, -9.0]]])
            boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]])
            return types.SimpleNamespace(logits=logits, pred_boxes=boxes)

    owl = partdetect._Owl.__new__(partdetect._Owl)
    owl.torch, owl.device, owl.dtype, owl.processor, owl.model = torch, torch.device("cpu"), torch.float32, Proc(), Model()
    monkeypatch.setattr(partdetect, "_runner", owl)
    monkeypatch.setattr(partdetect, "_state", "ready")
    img = np.zeros((100, 200, 3), np.uint8)
    out = partdetect.detect_in(img, [(100, 0, 200, 100), (0, 0, 50, 50)], ["brake caliper", "lug nut"])
    assert [d["src"] for d in out] == ["crop0", "crop1"]
    assert out[0]["box"] == pytest.approx([140.0, 40.0, 160.0, 60.0])   # mapped back from the window
    assert out[0]["phrase"] == "brake caliper" and len(out[0]["scores"]) == 2 and out[0]["scores"][1] < out[0]["score"]
    assert partdetect.detect_in(img, [], ["brake caliper"]) == []


@pytest.mark.parametrize("mod", ["partdetect", "florence", "foreground", "matting"])
def test_a_missing_snapshot_never_calls_from_pretrained(monkeypatch, mod):
    """transformers asks the network for its error message when from_pretrained does not find
    local files; a wrapper whose snapshot is missing must not call it at all."""
    import importlib
    m = importlib.import_module(f"recolor.segmentation.{mod}")
    called = []
    fake = types.ModuleType("transformers")

    def boom(*a, **kw):
        called.append(a)
        raise AssertionError("from_pretrained was called")

    for name in ("Owlv2Processor", "Owlv2ForObjectDetection", "AutoProcessor", "Florence2ForConditionalGeneration",
                 "AutoModelForImageSegmentation", "VitMatteImageProcessor", "VitMatteForImageMatting"):
        setattr(fake, name, type(name, (), {"from_pretrained": staticmethod(boom)}))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(m, "_snapshot_cached", lambda: False)
    monkeypatch.setattr(m, "_runner", None)
    monkeypatch.setattr(m, "_state", "cold")
    monkeypatch.setattr(m, "_weights_missing", False)
    assert m._get_runner() is None and m.status() == "unavailable" and called == []
    assert "setup.sh" in m._unavailable_reason
