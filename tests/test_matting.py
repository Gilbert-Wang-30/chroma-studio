"""Boundary snap (recolor/segmentation/matting.py) with ViTMatte mocked: trimaps, the
add-only label move, the model lifecycle and the guided-filter fallback. No model, no
network, no transformers import."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from recolor import imageio
from recolor.segmentation import matting
from recolor.types import ColorGroup

H, W = 60, 90


def _group(gid: int, lab, bg: bool = False) -> ColorGroup:
    lab = tuple(float(v) for v in lab)
    return ColorGroup(id=gid, name=f"g{gid}", albedo_lab=lab, albedo_hex=imageio.lab_to_hex(lab), area=1,
                      area_frac=0.1, region_ids=[gid], hue_family="red", is_background=bg)


def _scene():
    """A red part (group 1) whose label stops 3 px short of its real edge at x = 40, a grey
    part (group 2) and a white decal island inside the red, on a grey backdrop (group 0)."""
    labels = np.zeros((H, W), np.int32)
    labels[10:50, 10:37] = 1                     # label: x 10..36, the paint really reaches x 39
    labels[10:50, 50:80] = 2
    labels[25:35, 20:28] = 3                     # decal island (its own region, grey group)
    group_map = np.array([0, 1, 2, 2], np.int32)[labels]
    true = np.zeros((H, W), bool)
    true[10:50, 10:40] = True
    image = np.full((H, W, 3), 120, np.uint8)
    image[true] = (200, 30, 20)
    image[25:35, 20:28] = (245, 245, 245)
    groups = [_group(0, (50.0, 0.0, 0.0), bg=True), _group(1, (45.0, 60.0, 45.0)), _group(2, (60.0, 1.0, 1.0))]
    return image, labels, group_map, groups, true


class _FakeRunner:
    """Alpha = 1 on the true paint, 0 elsewhere; records every trimap it is given."""

    def __init__(self, true: np.ndarray) -> None:
        self.true = true
        self.trimaps: list[np.ndarray] = []

    def matte(self, image, trimap):
        self.trimaps.append(trimap.copy())
        a = self.true.astype(np.float32)
        a[trimap == 255] = 1.0
        a[trimap == 0] = 0.0
        return a


@pytest.fixture
def fake_model(monkeypatch):
    image, labels, group_map, groups, true = _scene()
    runner = _FakeRunner(true)
    monkeypatch.setattr(matting, "_runner", runner)
    monkeypatch.setattr(matting, "_state", "ready")
    return runner


def test_trimap_band_and_thin_parts():
    hard = np.zeros((H, W), bool)
    hard[10:50, 10:40] = True
    hard[5, 60:85] = True                        # a 1-px cable: no core survives a 6-px erosion
    tri = matting.make_trimap(hard, 6)
    assert set(np.unique(tri)) <= {0, 128, 255}
    assert (tri[20:40, 20:30] == 255).all()      # deep inside: sure paint
    assert tri[30, 12] == 128 and tri[30, 43] == 128 and tri[30, 55] == 0
    assert (tri[5, 60:85] == 255).any()          # the cable keeps a foreground anchor


def test_matte_move_grows_groups_only_where_alpha_says_so():
    image, labels, gm, groups, true = _scene()
    tri = matting.make_trimap(gm == 1)
    alpha = {1: true.astype(np.float32)}
    band = {1: tri == 128}
    unprotected = matting.matte_move_labels(labels, gm, alpha, band)
    assert (unprotected[25:35, 20:28] == 1).all()                   # unprotected, the decal hole is claimed
    protect = labels == 3
    protect[10:50, 38:40] = True                 # never claimed
    out = matting.matte_move_labels(labels, gm, alpha, band, protect=protect)
    grown = (out == 1) & (labels != 1)
    assert grown.sum() == 40 * 1                 # column 37 only: 38..39 and the decal are protected
    assert (out[10:50, 37] == 1).all() and not (out[protect] == 1).any()
    assert ((labels == 1) <= (out == 1)).all()   # nothing ever leaves a group


def test_snap_labels_with_vitmatte_moves_the_paint_edge(fake_model):
    image, labels, gm, groups, true = _scene()
    islands = labels == 3
    out, method = matting.snap_labels(image, labels, gm, groups, protect=islands)
    assert method == "vitmatte"
    assert len(fake_model.trimaps) == 1          # only the chromatic, non-background group is matted
    assert (out[10:50, 37:40] == 1).all()        # the label now reaches the real edge
    assert out.dtype == np.int32 and out.min() >= 0
    assert np.array_equal(out[islands], labels[islands])
    assert np.array_equal(out[:, 45:], labels[:, 45:])


def test_snap_labels_without_chromatic_groups_is_a_no_op(fake_model):
    image, labels, gm, groups, true = _scene()
    grey = [_group(g.id, (50.0, 1.0, 1.0), bg=g.is_background) for g in groups]
    out, method = matting.snap_labels(image, labels, gm, grey)
    assert method == "none" and np.array_equal(out, labels) and fake_model.trimaps == []


def test_falls_back_to_the_guided_filter_when_vitmatte_is_unavailable(monkeypatch, caplog):
    def broken(*a, **k):
        raise ImportError("No module named 'transformers'")

    monkeypatch.setattr(matting, "_runner", None)
    monkeypatch.setattr(matting, "_state", "cold")
    monkeypatch.setattr(matting, "_ViTMatte", broken)
    image, labels, gm, groups, true = _scene()
    islands = labels == 3
    with caplog.at_level(logging.WARNING, logger="recolor.segmentation.matting"):
        out, method = matting.snap_labels(image, labels, gm, groups, protect=islands)
        out2, method2 = matting.snap_labels(image, labels, gm, groups, protect=islands)
    assert method == method2 == "guided"
    assert matting.status() == "unavailable" and not matting.is_loaded()
    assert sum("ViTMatte unavailable" in r.getMessage() for r in caplog.records) == 1   # warned once
    assert out.dtype == np.int32 and out.shape == labels.shape and out.min() >= 0
    assert np.array_equal(out[islands], labels[islands])                                 # islands kept
    assert not ((out == 3) & ~islands).any()


def test_release_frees_and_status_reports(monkeypatch):
    monkeypatch.setattr(matting, "_runner", _FakeRunner(np.zeros((2, 2), bool)))
    monkeypatch.setattr(matting, "_state", "ready")
    assert matting.is_loaded() and matting.status() == "ready"
    matting.release()
    assert not matting.is_loaded() and matting.status() == "cold"
    matting.release()                            # idempotent
    assert matting.status() == "cold"


def test_a_full_gpu_falls_back_for_that_job_and_retries_later(monkeypatch, fake_model):
    """On the shared card ViTMatte may not fit: the job snaps with the guided filter, the model
    is dropped, and the next job tries again (a CUDA OOM never makes it 'unavailable')."""
    class OutOfMemoryError(RuntimeError):
        pass

    def oom(image, trimap):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")

    monkeypatch.setattr(fake_model, "matte", oom)
    image, labels, gm, groups, true = _scene()
    out, method = matting.snap_labels(image, labels, gm, groups, protect=labels == 3)
    assert method == "guided" and not matting.is_loaded() and matting.status() == "cold"

    def load_oom():
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 100.00 MiB")

    monkeypatch.setattr(matting, "_ViTMatte", load_oom)
    out, method = matting.snap_labels(image, labels, gm, groups, protect=labels == 3)
    assert method == "guided" and matting.status() == "cold"



def test_a_model_error_falls_back_for_that_job_and_keeps_the_model(monkeypatch, fake_model):
    """Any other failure inside ViTMatte (a panorama a few pixels tall made the processor
    concatenate the wrong axes) must not fail the analysis: that job uses the fallback, the
    model stays loaded for the next one."""
    def broken(image, trimap):
        raise RuntimeError("Sizes of tensors must match except in dimension 1")

    monkeypatch.setattr(fake_model, "matte", broken)
    image, labels, gm, groups, true = _scene()
    out, method = matting.snap_labels(image, labels, gm, groups, protect=labels == 3)
    assert method == "guided" and out.shape == labels.shape and out.min() >= 0
    assert matting.is_loaded() and matting.status() == "ready"


def test_matte_tells_the_processor_the_channel_axis():
    import torch

    class Processor:
        def __init__(self):
            self.kwargs = None

        def __call__(self, images, trimaps, return_tensors, **kw):
            self.kwargs = kw
            h, w = trimaps.shape
            return {"pixel_values": torch.zeros((1, 4, h, w))}

    class Out:
        def __init__(self, a):
            self.alphas = a

    class Model:
        def __call__(self, pixel_values):
            return Out(torch.full((1, 1) + tuple(pixel_values.shape[2:]), 0.5))

    vm = object.__new__(matting._ViTMatte)
    vm.torch, vm.device, vm.dtype = torch, torch.device("cpu"), torch.float32
    vm.processor, vm.model = Processor(), Model()
    tri = np.full((3, 40), 128, np.uint8)
    tri[:, :5] = 255
    a = vm.matte(np.zeros((3, 40, 3), np.uint8), tri)
    assert vm.processor.kwargs == {"input_data_format": "channels_last"}
    assert a.shape == (3, 40) and (a[:, :5] == 1.0).all() and np.allclose(a[:, 5:], 0.5)


def test_the_weights_are_only_read_from_the_local_cache(monkeypatch, caplog):
    """The analysis must never download ViTMatte (it ran under the GPU lock and could block
    every GPU user on an offline machine): missing weights make it 'unavailable', the next
    job looks at the local cache again (setup.sh needs no server restart), warned once."""
    import sys
    import types
    calls = []
    present = {"ok": False}

    class _Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def half(self):
            return self

    def from_pretrained(cls_name):
        def f(model_id, **kw):
            calls.append((cls_name, dict(kw)))
            if not kw.get("local_files_only"):
                raise AssertionError("tried the network")
            if not present["ok"]:
                raise OSError("We couldn't connect to 'https://huggingface.co' ... not in the cache")
            return _Model()
        return staticmethod(f)

    fake = types.ModuleType("transformers")
    fake.VitMatteImageProcessor = type("P", (), {"from_pretrained": from_pretrained("processor")})
    fake.VitMatteForImageMatting = type("M", (), {"from_pretrained": from_pretrained("model")})
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(matting, "_snapshot_cached", lambda: True)          # the fake transformers decide
    monkeypatch.setattr(matting, "_runner", None)
    monkeypatch.setattr(matting, "_state", "cold")
    monkeypatch.setattr(matting, "_weights_missing", False)
    image, labels, gm, groups, true = _scene()
    with caplog.at_level(logging.WARNING, logger="recolor.segmentation.matting"):
        _, method = matting.snap_labels(image, labels, gm, groups, protect=labels == 3)
        assert method == "guided" and matting.status() == "unavailable"
        _, method = matting.snap_labels(image, labels, gm, groups, protect=labels == 3)
        assert method == "guided"
    assert sum("ViTMatte unavailable" in r.getMessage() for r in caplog.records) == 1
    assert "setup.sh" in matting._unavailable_reason
    assert len([c for c in calls if c[0] == "processor"]) == 2        # looked again on the second job
    assert all(kw["local_files_only"] and kw["revision"] == matting.MODEL_REVISION for _, kw in calls)
    present["ok"] = True                                             # setup.sh ran meanwhile
    assert matting._get_runner() is not None and matting.status() == "ready"
    monkeypatch.setattr(matting, "_runner", None)
