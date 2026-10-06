"""The working-resolution reference a full-resolution export takes from the preview (``pipeline._working_reference``),
with the real engine on a tiny synthetic white-paint scene: the white paint's glints (rule 7e) and every group's
neutral-source weights, measured on the working-resolution layers. No models, no network, no job on disk.

* The preview's cached renderer is reused only when it holds the export's own snapshot (the edit generation it was
  built at, ``pipeline._renderer_gen``, is the export's); otherwise a renderer is built for the reference and freed.
* No glints when the mapping repaints no neutral source (nothing to keep), but always the weights.
* ``(None, None)`` when the card runs out of memory, so the export measures its own instead of failing.
"""
from __future__ import annotations

import types
from collections import OrderedDict

import numpy as np
import pytest

from recolor import engine, pipeline
from recolor.types import RenderOptions

from test_engine_neutral import NAVY, RED, WHITE, glint_scene


@pytest.fixture
def ref(monkeypatch):
    """(job, layers, groups, built): an empty renderer cache, and an engine whose ``Renderer`` records every
    renderer it builds (the real one underneath)."""
    monkeypatch.setattr(pipeline, "_renderer_cache", OrderedDict())
    monkeypatch.setattr(pipeline, "_renderer_gen", {})
    built: list = []

    class Recording(engine.Renderer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            built.append(self)

    monkeypatch.setattr(pipeline, "_engine", lambda: types.SimpleNamespace(Renderer=Recording))
    albedo, shading, residual, gm, groups = glint_scene()
    layers = {"albedo": albedo, "shading": shading, "residual": residual, "group_map": gm,
              "islands": None, "protect": None}
    job = types.SimpleNamespace(id="ref-job")
    yield job, layers, groups, built
    for r in built:
        r.free()


def _cache(job, layers, groups, gen: int) -> engine.Renderer:
    r = engine.Renderer(layers["albedo"], layers["shading"], layers["residual"], layers["group_map"], groups)
    pipeline._renderer_cache[job.id] = r
    pipeline._renderer_gen[job.id] = gen
    return r


def test_the_previews_renderer_is_reused_for_its_own_snapshot(ref):
    job, layers, groups, built = ref
    cached = _cache(job, layers, groups, gen=4)
    try:
        cached.render({WHITE: NAVY}, RenderOptions())          # the preview has found its glints
        glints, neutral = pipeline._working_reference(job, layers, groups, 4, {WHITE: NAVY}, RenderOptions())
        assert built == []                                      # nothing built: the preview's own renderer
        assert np.array_equal(glints, cached.white_glints()) and np.array_equal(neutral, cached.neutral_weights())
        assert glints.max() > 0.8 and neutral[WHITE, 0] == 1.0
        assert not cached.freed                                 # still the preview's
    finally:
        cached.free()


def test_another_snapshot_gets_a_renderer_of_its_own(ref):
    job, layers, groups, built = ref
    cached = _cache(job, layers, groups, gen=4)
    try:
        for gen in (6, None):                                   # an edit since, or no known generation
            glints, neutral = pipeline._working_reference(job, layers, groups, gen, {WHITE: NAVY}, RenderOptions())
            assert built and built[-1].freed                    # built for the reference and freed before the export
            assert glints.shape == layers["group_map"].shape and neutral.shape == (3, 2)
        assert len(built) == 2 and not cached.freed
    finally:
        cached.free()


def test_no_glints_without_a_neutral_source(ref):
    """A mapping that repaints no white paint (a saturated source only, or the white mapped to its own swatch
    colour, which renders as the saturated-paint rules do) needs no glints; the weights still come along."""
    job, layers, groups, built = ref
    swatch = next(g for g in groups if g.id == WHITE).albedo_hex
    for mapping in ({RED: NAVY}, {WHITE: swatch}):
        glints, neutral = pipeline._working_reference(job, layers, groups, None, mapping, RenderOptions())
        assert glints is None and neutral is not None and neutral[WHITE, 0] == 1.0
    cached = _cache(job, layers, groups, gen=2)
    try:
        glints, neutral = pipeline._working_reference(job, layers, groups, 2, {RED: NAVY}, RenderOptions())
        assert glints is None and np.array_equal(neutral, cached.neutral_weights())
    finally:
        cached.free()


def test_out_of_memory_leaves_the_export_to_measure_its_own(ref, monkeypatch):
    job, layers, groups, built = ref
    cached = _cache(job, layers, groups, gen=3)

    def oom(*a, **kw):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    try:
        cached.neutral_weights = oom
        assert pipeline._working_reference(job, layers, groups, 3, {WHITE: NAVY}, RenderOptions()) == (None, None)
        del cached.neutral_weights
        cached.white_glints = oom                               # ... and during the glint search
        assert pipeline._working_reference(job, layers, groups, 3, {WHITE: NAVY}, RenderOptions()) == (None, None)
        del cached.white_glints
        assert pipeline._working_reference(job, layers, groups, 3, {WHITE: NAVY}, RenderOptions())[0] is not None
    finally:
        cached.free()

    class Full(engine.Renderer):                                # no room for a renderer of its own
        def __init__(self, *a, **kw):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(pipeline, "_engine", lambda: types.SimpleNamespace(Renderer=Full))
    assert pipeline._working_reference(job, layers, groups, None, {WHITE: NAVY}, RenderOptions()) == (None, None)
