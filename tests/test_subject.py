"""The matte keeps one subject (recolor/segmentation/subject.py, pipeline._subject_matte): another
object of the photo's class that the matte took in goes to the backdrop, a piece of the subject
that SAM's silhouette missed stays. SAM is a stand-in that answers from hand-drawn masks. No
models."""
from __future__ import annotations

import types

import numpy as np

from recolor import imageio, pipeline
from recolor.segmentation import grouping, subject
from recolor.types import AnalysisOptions

H, W = 240, 400


def _scene():
    """A car (the subject, 80x140) and, behind it, a second car whose door panel (30x40) the matte
    took in; the rest of the second car is off the matte."""
    car = np.zeros((H, W), bool)
    car[60:220, 80:360] = True
    other = np.zeros((H, W), bool)
    other[10:80, 0:120] = True                                       # the car behind ...
    other &= ~car
    fg = np.zeros((H, W), np.float32)
    fg[car] = 1.0
    fg[10:60, 40:120] = 1.0                                          # ... its door panel on the matte
    return car, other, fg


def _prompter(car, other, piece=None):
    """SAM: the box gives the subject's silhouette; a point gives the object it lies on (the
    second car, or a piece of the subject when ``piece`` holds the point)."""
    calls = []

    def prompt(image, jobs):
        calls.append(jobs)
        out = []
        for j in jobs:
            x0, y0, x1, y1 = j["crop"]
            if "box" in j:
                m = car
            else:
                px, py = j["points"][0]
                m = piece if piece is not None and piece[py, px] else (other if other[py, px] else car)
            out.append([{"mask": m[y0:y1, x0:x1], "x0": x0, "y0": y0, "score": 0.95, "clipped": False}])
        return out

    prompt.calls = calls
    return prompt


def _image():
    return np.full((H, W, 3), 128, np.uint8)


def test_another_object_the_matte_took_in_goes_to_the_backdrop():
    car, other, fg = _scene()
    rep = {}
    out = subject.other_objects(_image(), fg, _prompter(car, other), report=rep)
    panel = (fg >= 0.5) & ~car
    assert out.sum() >= 0.8 * panel.sum() and not (out & car).any()
    assert rep["subject_cover"] > 0.85 and rep["components"][0]["other"]


def test_a_piece_of_the_subject_its_silhouette_missed_stays():
    """The Corvette's bumper corner behind a sign post: outside SAM's silhouette, but its own SAM
    object reaches into the subject."""
    car, other, fg = _scene()
    corner = np.zeros((H, W), bool)
    corner[10:60, 40:120] = True
    piece = corner | car                                             # the point's object is the whole car
    out = subject.other_objects(_image(), fg, _prompter(car, other, piece=piece))
    assert not out.any()


def test_a_detected_part_and_a_weak_silhouette_keep_the_matte():
    car, other, fg = _scene()
    keep = np.zeros((H, W), bool)
    keep[10:60, 40:120] = True                                        # a detected part there
    assert not subject.other_objects(_image(), fg, _prompter(car, other), keep=[keep]).any()
    half = car.copy()
    half[:, 220:] = False                                            # SAM's answer covers half the matte
    assert not subject.other_objects(_image(), fg, _prompter(half, other)).any()


def test_a_region_mostly_in_the_other_object_goes_with_it_whole():
    """The 5 px seam kept along the subject's silhouette stays with the subject only when its
    region is the subject's: a region of the other object goes whole."""
    car, other, fg = _scene()
    labels = np.zeros((H, W), np.int32)
    labels[car] = 1
    panel = (fg >= 0.5) & ~car
    labels[panel] = 2                                                # the panel is a region of its own
    others = subject.other_objects(_image(), fg, _prompter(car, other))
    assert (labels[others] == 2).all() and others.sum() < panel.sum()      # less the seam
    whole = subject.whole_regions(labels, others)
    assert np.array_equal(whole, labels == 2)                               # the seam goes too
    keep = np.zeros((H, W), bool)
    keep[10:20, 40:60] = True
    assert not (subject.whole_regions(labels, others, keep=keep) & keep).any()


def test_the_regions_stage_gives_the_other_object_to_the_backdrop(monkeypatch):
    """pipeline._subject_matte on a partition: the panel region is cut off the car's region and,
    with the matte set to backdrop there, the backdrop decisions flag it; a pair of sneakers (a
    class of two subjects) is left alone."""
    car, other, fg = _scene()
    labels = np.zeros((H, W), np.int32)                              # 0 the backdrop
    labels[car] = 1                                                  # 1 the car
    labels[(fg >= 0.5) & ~car] = 1                                   # ... with the panel in its region
    info = [{"id": 0, "source": "sam"}, {"id": 1, "source": "sam"}]
    albedo = np.full((H, W, 3), 0.3, np.float32)
    albedo[labels == 1] = imageio.lab_to_linear(np.array([[[46.0, 56.0, 51.0]]], np.float32))[0, 0]
    prompt = _prompter(car, other)
    monkeypatch.setattr(pipeline, "_sam_masks", lambda: types.SimpleNamespace(
        SamMasker=lambda: types.SimpleNamespace(prompt_boxes=prompt)))
    job = types.SimpleNamespace(id="subj00000000", options=AnalysisOptions())
    ctx = {"work": _image(), "albedo": albedo, "caption": "an orange car in a showroom", "part_masks": []}
    out, out_info, fg2 = pipeline._subject_matte(job, ctx, labels, info, fg)
    panel = (fg >= 0.5) & ~car
    assert len(out_info) == 3 and out_info[2]["source"] == "split"
    cut = out == 2
    assert cut.sum() >= 0.9 * panel.sum() and not (cut & car).any()   # the panel, less a 5 px seam at the car
    assert float(fg2[cut].max()) == 0.0 and float(fg2[car].min()) == 1.0
    kinds = grouping.backdrop_decisions(out, albedo, out_info, fg2)
    assert kinds[2] > 0 and kinds[1] == 0
    ctx["caption"] = "a pair of white sneakers"
    out3, info3, fg3 = pipeline._subject_matte(job, ctx, labels, info, fg)
    assert out3 is labels and fg3 is fg
