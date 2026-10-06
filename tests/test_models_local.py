"""The two model wrappers this round added (recolor/segmentation/florence.py and
foreground.py) with transformers mocked: weights only from the local cache, the fallbacks
on missing weights, a full GPU and a model error. No model, no network."""
from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from recolor.segmentation import florence, foreground


class _Model:
    def to(self, device):
        return self

    def eval(self):
        return self

    def half(self):
        return self


def _fake_transformers(calls, present):
    def from_pretrained(name):
        def f(model_id, **kw):
            calls.append((name, dict(kw)))
            if not kw.get("local_files_only"):
                raise AssertionError("tried the network")
            if not present["ok"]:
                raise OSError("We couldn't connect to 'https://huggingface.co' ... not in the cache")
            return _Model()
        return staticmethod(f)

    fake = types.ModuleType("transformers")
    fake.AutoProcessor = type("P", (), {"from_pretrained": from_pretrained("processor")})
    fake.Florence2ForConditionalGeneration = type("F", (), {"from_pretrained": from_pretrained("florence")})
    fake.AutoModelForImageSegmentation = type("S", (), {"from_pretrained": from_pretrained("birefnet")})
    return fake


@pytest.fixture
def fresh(monkeypatch):
    for mod in (florence, foreground):
        monkeypatch.setattr(mod, "_snapshot_cached", lambda: True)       # the fake transformers decide
        monkeypatch.setattr(mod, "_runner", None)
        monkeypatch.setattr(mod, "_state", "cold")
        monkeypatch.setattr(mod, "_weights_missing", False)
        monkeypatch.setattr(mod, "_unavailable_reason", None)


@pytest.mark.parametrize("mod,call,kind", [(florence, lambda m, img: m.analyse(img), "florence"),
                                           (foreground, lambda m, img: m.fg_prob(img), "birefnet")])
def test_weights_are_only_read_from_the_local_cache(monkeypatch, caplog, fresh, mod, call, kind):
    calls, present = [], {"ok": False}
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls, present))
    img = np.zeros((40, 60, 3), np.uint8)
    with caplog.at_level(logging.WARNING, logger=mod.log.name):
        assert call(mod, img) is None and mod.status() == "unavailable"   # no weights: the stage runs without it
        assert call(mod, img) is None
    assert sum("unavailable" in r.getMessage() for r in caplog.records) == 1     # warned once
    assert "setup.sh" in mod._unavailable_reason
    assert len([c for c in calls if c[0] in (kind, "processor")]) == 2   # looked at the local cache again on the next job
    assert all(kw["local_files_only"] and kw["revision"] == mod.MODEL_REVISION for _, kw in calls)
    present["ok"] = True                                                # setup.sh ran meanwhile
    assert mod._get_runner() is not None and mod.status() == "ready" and mod.is_loaded()
    mod.release()
    assert not mod.is_loaded() and mod.status() == "cold"


class OutOfMemoryError(RuntimeError):
    pass


def test_florence_parses_ocr_and_grounding_and_falls_back_on_errors(monkeypatch, fresh):
    class Runner:
        def __init__(self):
            self.tasks = []

        def run(self, images, task, text=None):
            self.tasks.append((task, text, len(images)))
            if task == "<OCR_WITH_REGION>":
                return [{"quad_boxes": [[1, 2, 9, 2, 9, 6, 1, 6]], "labels": ["DUCATI</s>"]}] + \
                       [{"quad_boxes": [], "labels": []} for _ in images[1:]]
            return [{"bboxes": [[3, 4, 30, 20]], "labels": ["tire"]}]

    runner = Runner()
    monkeypatch.setattr(florence, "_runner", runner)
    monkeypatch.setattr(florence, "_state", "ready")
    out = florence.analyse(np.zeros((80, 120, 3), np.uint8))
    assert out["ocr"] == [{"quad": [1.0, 2.0, 9.0, 2.0, 9.0, 6.0, 1.0, 6.0], "text": "DUCATI", "src": "full"}]
    assert out["grounding"] == [{"box": [3.0, 4.0, 30.0, 20.0], "label": "tire"}]
    assert [t[0] for t in runner.tasks] == ["<OCR_WITH_REGION>", "<CAPTION_TO_PHRASE_GROUNDING>"]
    assert runner.tasks[0][2] == 5 and runner.tasks[1][1] == florence.CAPTION      # the image and its four tiles
    assert set(out["seconds"]) == {"ocr", "grounding"}

    def oom(images, task, text=None):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")

    monkeypatch.setattr(runner, "run", oom)
    assert florence.analyse(np.zeros((80, 120, 3), np.uint8)) is None
    assert not florence.is_loaded() and florence.status() == "cold"      # released, retried on the next job

    broken = Runner()
    broken.run = lambda images, task, text=None: (_ for _ in ()).throw(RuntimeError("bad tensor"))
    monkeypatch.setattr(florence, "_runner", broken)
    monkeypatch.setattr(florence, "_state", "ready")
    assert florence.analyse(np.zeros((80, 120, 3), np.uint8)) is None
    assert florence.is_loaded() and florence.status() == "ready"         # a model error keeps the model


def test_foreground_matte_shape_and_fallbacks(monkeypatch, fresh):
    class Runner:
        def matte(self, image):
            h, w = image.shape[:2]
            fg = np.zeros((h, w), np.float32)
            fg[:, w // 2:] = 1.0
            return fg

    monkeypatch.setattr(foreground, "_runner", Runner())
    monkeypatch.setattr(foreground, "_state", "ready")
    fg = foreground.fg_prob(np.zeros((30, 50, 3), np.uint8))
    assert fg.shape == (30, 50) and fg.dtype == np.float32 and fg[:, 25:].all() and not fg[:, :25].any()
    with pytest.raises(ValueError):
        foreground.fg_prob(np.zeros((30, 50), np.uint8))

    def oom(image):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")

    monkeypatch.setattr(foreground._runner, "matte", oom)
    assert foreground.fg_prob(np.zeros((30, 50, 3), np.uint8)) is None
    assert not foreground.is_loaded() and foreground.status() == "cold"
