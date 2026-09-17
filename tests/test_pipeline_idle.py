"""Idle-unload: SAM 2 and the intrinsic model must load lazily and drop again after a
period with no analysis or full-resolution export running, but never mid-use.

`recolor.intrinsic` and `recolor.segmentation.sam_masks` are replaced with tiny fakes
that track their own loaded/released state (independent of tests/test_server.py's
fakes, which focus on analysis output rather than the load/unload lifecycle). No
models, no network, no GPU, no real sleeping: `_idle_tick()` is the watchdog's single
per-check decision, called directly instead of through the background thread's loop.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest

from recolor import config, pipeline


class _FakeModel:
    """A sibling's is_loaded()/release() pair, with call counts for assertions."""

    def __init__(self) -> None:
        self.loaded = False
        self.release_calls = 0

    def is_loaded(self, method: str = "careaga") -> bool:
        return self.loaded

    def release(self, method: str = "careaga") -> None:
        self.release_calls += 1
        self.loaded = False


@pytest.fixture
def fakes(monkeypatch):
    intrinsic, sam = _FakeModel(), _FakeModel()

    intrinsic_mod = types.ModuleType("recolor.intrinsic")
    intrinsic_mod.is_loaded = intrinsic.is_loaded
    intrinsic_mod.release = intrinsic.release

    sam_mod = types.ModuleType("recolor.segmentation.sam_masks")
    sam_mod.is_loaded = lambda: sam.loaded
    sam_mod.release = lambda: sam.release()

    monkeypatch.setitem(sys.modules, "recolor.intrinsic", intrinsic_mod)
    monkeypatch.setitem(sys.modules, "recolor.segmentation.sam_masks", sam_mod)
    monkeypatch.setattr(pipeline, "_models", {"sam2": "cold", "intrinsic": "cold"})
    monkeypatch.setattr(pipeline, "_current_job_id", None)
    monkeypatch.setattr(config, "IDLE_UNLOAD_S", 60.0)
    pipeline._last_activity = pipeline.time.monotonic()   # fresh clock for every test
    return intrinsic, sam


def _idle_since(seconds_ago: float) -> None:
    pipeline._last_activity = pipeline.time.monotonic() - seconds_ago


def test_touch_activity_resets_the_clock():
    pipeline._last_activity = 0.0
    pipeline._touch_activity()
    assert pipeline._last_activity > 0.0


def test_idle_tick_does_nothing_before_the_timeout(fakes):
    intrinsic, sam = fakes
    intrinsic.loaded = sam.loaded = True
    _idle_since(1.0)   # well under the 60 s fixture timeout
    pipeline._idle_tick()
    assert intrinsic.release_calls == 0 and sam.release_calls == 0
    assert intrinsic.loaded and sam.loaded


def test_idle_tick_releases_both_models_once_idle(fakes):
    intrinsic, sam = fakes
    intrinsic.loaded = sam.loaded = True
    _idle_since(3600.0)
    pipeline._idle_tick()
    assert intrinsic.release_calls == 1 and sam.release_calls == 1
    assert not intrinsic.loaded and not sam.loaded
    assert pipeline._models == {"sam2": "cold", "intrinsic": "cold"}


def test_idle_tick_is_a_no_op_when_nothing_is_loaded(fakes):
    intrinsic, sam = fakes
    _idle_since(3600.0)
    pipeline._idle_tick()   # must not explode or touch _models when there is nothing to drop
    assert intrinsic.release_calls == 0 and sam.release_calls == 0


def test_idle_tick_disabled_when_the_timeout_is_zero(fakes, monkeypatch):
    intrinsic, sam = fakes
    intrinsic.loaded = sam.loaded = True
    monkeypatch.setattr(config, "IDLE_UNLOAD_S", 0.0)
    _idle_since(3600.0)
    pipeline._idle_tick()
    assert intrinsic.release_calls == 0 and sam.release_calls == 0


def test_idle_tick_never_fires_while_a_job_is_running(fakes, monkeypatch):
    intrinsic, sam = fakes
    intrinsic.loaded = sam.loaded = True
    _idle_since(3600.0)
    monkeypatch.setattr(pipeline, "_current_job_id", "abc123")
    pipeline._idle_tick()
    assert intrinsic.release_calls == 0 and sam.release_calls == 0


def test_idle_tick_never_fires_while_jobs_are_queued(fakes, monkeypatch):
    intrinsic, sam = fakes
    intrinsic.loaded = sam.loaded = True
    _idle_since(3600.0)
    monkeypatch.setattr(pipeline, "queue_length", lambda: 1)
    pipeline._idle_tick()
    assert intrinsic.release_calls == 0 and sam.release_calls == 0


def test_idle_tick_never_fires_while_another_thread_holds_the_gpu_lock(fakes):
    """gpu_lock is an RLock, so the watchdog's own thread reacquiring it proves
    nothing; a real stage or export holds it from a *different* thread."""
    intrinsic, sam = fakes
    intrinsic.loaded = sam.loaded = True
    _idle_since(3600.0)
    holder_ready = threading.Event()
    release_holder = threading.Event()

    def hold() -> None:
        with pipeline.gpu_lock:
            holder_ready.set()
            release_holder.wait(timeout=5)

    t = threading.Thread(target=hold)
    t.start()
    try:
        assert holder_ready.wait(timeout=5)
        pipeline._idle_tick()
    finally:
        release_holder.set()
        t.join(timeout=5)
    assert intrinsic.release_calls == 0 and sam.release_calls == 0


def test_release_idle_models_survives_a_broken_release(fakes, monkeypatch):
    """A release() that raises must not take the whole check down, and must not
    falsely mark that model as unloaded."""
    intrinsic, _sam = fakes
    intrinsic.loaded = True
    pipeline._models["intrinsic"] = "ready"

    def boom(method: str = "careaga") -> None:
        raise RuntimeError("disk is gone")

    monkeypatch.setattr(sys.modules["recolor.intrinsic"], "release", boom)
    pipeline._release_idle_models()   # must not raise
    assert pipeline._models["intrinsic"] == "ready"
