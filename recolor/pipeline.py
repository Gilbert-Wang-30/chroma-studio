"""Analysis orchestration, the single GPU worker, model warm-up, renders and exports.

The sibling modules (`recolor.intrinsic`, `recolor.segmentation.*`, `recolor.engine`,
`recolor.palette`, `recolor.mapping`) are imported lazily by name inside the functions
that need them, exactly against the signatures in docs/ARCHITECTURE.md §3, so this
module imports cleanly while they are still being written and tests can substitute
them through `sys.modules`.

Concurrency model:
- One daemon worker thread drains a FIFO queue of analysis jobs (the GPU is shared).
- Model warm-up is the first task on that same queue, so it can never race a job for
  the lazy model singletons; `model_status()` reports cold/loading/ready for the UI.
- Exports take `gpu_lock`, which the worker also holds per stage, so a full-resolution
  export slots in between stages instead of colliding with a 14 GB intrinsic pass.
- Preview renders are serialized by a separate lock and never wait for the worker.

Model lifecycle: SAM 2 and the intrinsic model are lazy singletons loaded on first use
(analysis, or a full-resolution export). A background watchdog (started by
`start_worker`, alongside the worker thread) drops them again after
`config.IDLE_UNLOAD_S` seconds with no analysis or full-resolution export running, so
the GPU holds their ~2-3 GB only while the app is actually being used. A startup
`--warmup` just moves the first load earlier; it does not exempt the models from being
unloaded later. Preview renders and working-resolution exports never touch either
model, so tuning colors on an already-analyzed job never keeps them resident.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import queue
import threading
import time
from collections import Counter, OrderedDict
from typing import Any, Callable, Optional

import numpy as np

from . import config, filters, imageio
from .jobs import Job, registry
from .types import STAGES, AnalysisOptions, ColorGroup, Mapping, Region, RenderOptions, mapping_to_json

log = logging.getLogger("recolor.pipeline")

DETAIL_LEVELS = ("fast", "balanced", "max")
INTRINSIC_METHODS = ("auto", "careaga", "heuristic")
GROUP_EDIT_KINDS = ("merge", "split", "move", "regroup", "update")
EXPORT_QUALITIES = ("work", "full")
EXPORT_FORMATS = ("png", "jpg")

# Guided label refinement allocates several [G,H,W] float tensors; above this many
# elements per tensor the export falls back to a plain nearest upsample rather than
# risking a CUDA OOM on a 24-megapixel original with many groups.
_REFINE_MAX_ELEMENTS = 320_000_000

gpu_lock = threading.RLock()
_render_lock = threading.Lock()


class PipelineError(Exception):
    """A user-facing failure with an HTTP status: bad input (400), missing artifact
    (404) or a job in the wrong state (409). Anything else is a real bug."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# =============================================================================
# lazy sibling imports
# =============================================================================

def _mod(name: str):
    return importlib.import_module(name)


def _intrinsic():
    return _mod("recolor.intrinsic")


def _sam_masks():
    return _mod("recolor.segmentation.sam_masks")


def _hierarchy():
    return _mod("recolor.segmentation.hierarchy")


def _grouping():
    return _mod("recolor.segmentation.grouping")


def _engine():
    return _mod("recolor.engine")


# =============================================================================
# worker thread and warm-up
# =============================================================================

_task_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
_worker: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_current_job_id: Optional[str] = None

_models: dict[str, str] = {"sam2": "cold", "intrinsic": "cold"}
_warmup_started = False

# Idle-unload: SAM 2 and the intrinsic model are dropped from the GPU after
# `config.IDLE_UNLOAD_S` seconds with no analysis or full-resolution export running,
# then reloaded lazily (like any other lazy singleton) on the next one.
_last_activity: float = time.monotonic()
_idle_watchdog: Optional[threading.Thread] = None
_idle_watchdog_lock = threading.Lock()
_IDLE_CHECK_PERIOD_S = 5.0


def _touch_activity() -> None:
    """Record that the GPU models were just asked to do something, resetting the
    idle-unload countdown."""
    global _last_activity
    _last_activity = time.monotonic()


def start_worker() -> None:
    """Start the single FIFO GPU worker and the idle-unload watchdog (idempotent)."""
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, name="recolor-worker", daemon=True)
            _worker.start()
    _start_idle_watchdog()


def _start_idle_watchdog() -> None:
    global _idle_watchdog
    with _idle_watchdog_lock:
        if _idle_watchdog is not None and _idle_watchdog.is_alive():
            return
        _idle_watchdog = threading.Thread(target=_idle_watchdog_loop, name="recolor-idle-unload", daemon=True)
        _idle_watchdog.start()


def _idle_watchdog_loop() -> None:
    while True:
        time.sleep(_IDLE_CHECK_PERIOD_S)
        _idle_tick()


def _idle_tick() -> None:
    """One idle-unload check. Split out from the sleep loop so it can be called
    directly (and its timing/locking stubbed) without waiting on a real thread."""
    limit = config.IDLE_UNLOAD_S
    if limit <= 0 or time.monotonic() - _last_activity < limit:
        return
    if _current_job_id is not None or queue_length() > 0:
        return
    # A running stage or a full-resolution export holds gpu_lock for as long as it
    # needs the model; failing to acquire it here just means something is using the
    # GPU right now, so this is the only guard needed against unloading mid-use.
    if not gpu_lock.acquire(blocking=False):
        return
    try:
        _release_idle_models()
    finally:
        gpu_lock.release()


def _release_idle_models() -> None:
    """Drop SAM 2 and the intrinsic model if either is loaded, freeing the CUDA memory
    they hold. Called only with `gpu_lock` held. Errors are logged, not raised: a
    failed release just leaves the model resident until the next check."""
    released = []
    try:
        if _intrinsic().is_loaded("careaga"):
            _intrinsic().release("careaga")
            released.append("intrinsic")
    except Exception:  # noqa: BLE001
        log.exception("idle-unload: failed to release the intrinsic model")
    try:
        sam = _sam_masks()
        if callable(getattr(sam, "is_loaded", None)) and sam.is_loaded() and callable(getattr(sam, "release", None)):
            sam.release()
            released.append("sam2")
    except Exception:  # noqa: BLE001
        log.exception("idle-unload: failed to release SAM 2")
    if released:
        for name in released:
            _models[name] = "cold"
        log.info("released %s after %.0f s idle", " + ".join(released), config.IDLE_UNLOAD_S)


def _worker_loop() -> None:
    global _current_job_id
    while True:
        kind, payload = _task_queue.get()
        try:
            if kind == "warmup":
                _warmup_task()
            elif kind == "job":
                job = registry.get(payload)
                if job is not None:
                    _current_job_id = job.id
                    analyze(job)
        except Exception:  # noqa: BLE001 - the worker must survive anything
            log.exception("worker task %s failed", kind)
        finally:
            _current_job_id = None
            _task_queue.task_done()


def enqueue(job: Job) -> None:
    """Queue a job for analysis (FIFO) and make sure the worker is running."""
    start_worker()
    _task_queue.put(("job", job.id))


def queue_length() -> int:
    """Jobs waiting in the FIFO (not counting the one being analyzed)."""
    return _task_queue.qsize()


def resume_pending() -> int:
    """Re-queue jobs that were `queued`/`analyzing` when the server last stopped.
    Their stages are reset; the original image on disk is enough to restart."""
    n = 0
    for job in reversed(registry.all()):
        if job.status in ("queued", "analyzing"):
            with job.lock:
                for s in STAGES:
                    job.meta["stages"][s] = {"state": "idle", "progress": 0.0, "message": "", "seconds": 0.0}
                job.meta["status"] = "queued"
                job.meta["error"] = None
            job.save()
            enqueue(job)
            n += 1
    return n


def start_warmup() -> None:
    """Load SAM 2 and the intrinsic models on the worker thread so the first job is
    fast. Idempotent; failures are logged and leave the models `cold` (the pipeline
    still loads them lazily on first use)."""
    global _warmup_started
    with _worker_lock:
        if _warmup_started:
            return
        _warmup_started = True
    start_worker()
    _task_queue.put(("warmup", None))


def _warmup_task() -> None:
    _touch_activity()
    _models["intrinsic"] = "loading"
    try:
        with gpu_lock:
            _intrinsic().warmup("careaga")
        _models["intrinsic"] = "ready"
    except Exception as e:  # noqa: BLE001
        log.warning("intrinsic warm-up failed: %s", e)
        _models["intrinsic"] = "cold"
    _models["sam2"] = "loading"
    try:
        with gpu_lock:
            sam = _sam_masks()
            if callable(getattr(sam, "warmup", None)):
                sam.warmup()
            else:
                masker = sam.SamMasker()
                loader = next((getattr(masker, n) for n in ("warmup", "load", "ensure_loaded")
                               if callable(getattr(masker, n, None))), None)
                if loader is not None:
                    loader()
                else:
                    # No explicit loader: a tiny generate loads the model and compiles
                    # the kernels, which is exactly what warm-up is for.
                    probe = np.full((64, 64, 3), 128, np.uint8)
                    probe[16:48, 16:48] = (200, 40, 40)
                    masker.generate(probe, detail="fast")
        _models["sam2"] = "ready"
    except Exception as e:  # noqa: BLE001
        log.warning("SAM 2 warm-up failed: %s", e)
        _models["sam2"] = "cold"
    _free_cuda()


def model_status() -> dict[str, str]:
    """`{sam2, intrinsic}` each 'cold' | 'loading' | 'ready'. Consults the siblings'
    `is_loaded` probes when importable, so a model loaded lazily by a job (or released)
    is reported truthfully rather than from the warm-up's memory."""
    out = dict(_models)
    if out["intrinsic"] != "loading":
        try:
            out["intrinsic"] = "ready" if _intrinsic().is_loaded("careaga") else "cold"
        except Exception:  # noqa: BLE001 - module missing or model unavailable
            pass
    if out["sam2"] != "loading":
        try:
            sam = _sam_masks()
            probe = getattr(sam, "is_loaded", None) or getattr(sam.SamMasker, "is_loaded", None)
            if callable(probe):
                out["sam2"] = "ready" if probe() else "cold"
        except Exception:  # noqa: BLE001
            pass
    return out


def _free_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# =============================================================================
# analysis
# =============================================================================

class _Progress:
    """Adapter handed to sibling stages as `progress`. Tolerates every call shape a
    sibling might use: `(frac, msg)`, `(frac)`, `(msg)`, or keyword arguments."""

    def __init__(self, job: Job, stage: str, lo: float = 0.0, hi: float = 1.0):
        self.job, self.stage, self.lo, self.hi = job, stage, lo, hi

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        frac: Optional[float] = None
        msg: Optional[str] = None
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, str) and msg is None:
                msg = a
            elif isinstance(a, (int, float)) and not isinstance(a, bool) and frac is None:
                frac = float(a)
        p = None if frac is None else self.lo + (self.hi - self.lo) * min(1.0, max(0.0, frac))
        self.job.set_stage(self.stage, "running", progress=p, message=msg)


_OOM_RETRY_DELAY_S = 3.0


def _is_cuda_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e)


def _describe_error(e: BaseException) -> str:
    """A short, human message for the UI; the full traceback goes to the log. CUDA OOM
    texts are several hundred characters of allocator advice, so they are summarized."""
    text = str(e)
    if _is_cuda_oom(e):
        import re
        need = re.search(r"Tried to allocate ([\d.]+ [GM]iB)", text)
        free = re.search(r"of which ([\d.]+ [GM]iB) is free", text)
        parts = ["GPU out of memory"]
        if need and free:
            parts.append(f"(needed {need.group(1)}, {free.group(1)} free)")
        return " ".join(parts) + "; another process is using the GPU - try again in a moment or lower the detail"
    if len(text) > 300:
        text = text[:297] + "..."
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


def _job_alive(job: Job) -> bool:
    return not job.deleted and registry.get(job.id) is job and os.path.isdir(job.dir)


def analyze(job: Job) -> None:
    """Run the whole analysis on the worker thread: ingest → intrinsic → segment →
    regions → groups. Every stage records progress, a human message and its seconds;
    the first exception marks the stage and the job as `error` (message preserved)
    and stops. On success the job is `ready` with all artifacts of §3.6 written."""
    job.set_status("analyzing")
    t0 = time.monotonic()
    ctx: dict[str, Any] = {}
    steps: list[tuple[str, Callable[[Job, dict[str, Any]], None]]] = [
        ("ingest", _stage_ingest),
        ("intrinsic", _stage_intrinsic),
        ("segment", _stage_segment),
        ("regions", _stage_regions),
        ("groups", _stage_groups),
    ]
    for stage, fn in steps:
        if not _job_alive(job):
            log.info("job %s deleted mid-analysis, stopping", job.id)
            job.pending_image = None
            _free_cuda()
            return
        job.set_stage(stage, "running", progress=0.0)
        try:
            try:
                with gpu_lock:
                    _touch_activity()
                    fn(job, ctx)
            except Exception as e:  # noqa: BLE001
                if not _is_cuda_oom(e) or not _job_alive(job):
                    raise
                # The card is shared: free what we can, give the neighbour a moment, retry once.
                log.warning("job %s: stage %s hit CUDA OOM, retrying once", job.id, stage)
                _free_cuda()
                job.set_stage(stage, "running", progress=0.0, message="GPU memory is tight, retrying")
                time.sleep(_OOM_RETRY_DELAY_S)
                with gpu_lock:
                    _touch_activity()
                    fn(job, ctx)
            if not _job_alive(job):
                # Deleted while the stage ran: its files are gone, do not record anything.
                log.info("job %s deleted during stage %s, stopping", job.id, stage)
                job.pending_image = None
                _free_cuda()
                return
            job.set_stage(stage, "done")
        except Exception as e:  # noqa: BLE001 - reported to the user, never crashes the worker
            job.pending_image = None   # ingest already wrote the original; nothing needs the array now
            if not _job_alive(job):
                # A stage that writes into a directory deleted under it fails with ENOENT;
                # that is not an error worth reporting (and there is nowhere to report it).
                log.info("job %s deleted during stage %s (%s), stopping", job.id, stage, type(e).__name__)
                _free_cuda()
                return
            log.exception("job %s: stage %s failed", job.id, stage)
            msg = _describe_error(e)
            job.set_stage(stage, "error", message=msg)
            for later, _ in steps[[s for s, _ in steps].index(stage) + 1:]:
                job.set_stage(later, "skipped")
            job.update(timings={**job.meta["timings"], "total_s": round(time.monotonic() - t0, 3)})
            job.set_status("error", error=f"{stage}: {msg}")
            _free_cuda()
            return
    job.pending_image = None
    job.update(timings={**job.meta["timings"], "total_s": round(time.monotonic() - t0, 3)})
    job.set_status("ready")
    _free_cuda()


def _stage_ingest(job: Job, ctx: dict[str, Any]) -> None:
    job.set_stage("ingest", "running", 0.1, "Reading the image")
    img = job.pending_image
    if img is None:
        img = imageio.load_image(job.path(job.original_file))
    h, w = img.shape[:2]
    ww, wh = imageio.fit_size(w, h, config.WORK_LONG_SIDE)
    pw, ph = imageio.fit_size(w, h, config.PREVIEW_LONG_SIDE)
    job.set_stage("ingest", "running", 0.4, f"Working copy at {ww}×{wh}")
    work = imageio.resize_to(img, (ww, wh))
    imageio.save_image(job.path("work.png"), work)
    if not os.path.exists(job.path("preview.jpg")):
        imageio.save_image(job.path("preview.jpg"), imageio.resize_to(img, (pw, ph)), quality=90)
    job.update(image={"width": w, "height": h, "work_width": ww, "work_height": wh,
                      "preview_width": pw, "preview_height": ph})
    ctx["work"] = work
    job.set_stage("ingest", "running", 1.0, f"{w}×{h} → work {ww}×{wh}, preview {pw}×{ph}")


def _stage_intrinsic(job: Job, ctx: dict[str, Any]) -> None:
    opts = job.options
    label = {"auto": "Intrinsic v2.1", "careaga": "Intrinsic v2.1", "heuristic": "heuristic"}.get(opts.intrinsic, opts.intrinsic)
    job.set_stage("intrinsic", "running", 0.02, f"Separating paint from light · {label}")
    mod = _intrinsic()
    res = mod.decompose(ctx["work"], method=opts.intrinsic, progress=_Progress(job, "intrinsic", 0.05, 0.85))
    if res.albedo.shape[:2] != ctx["work"].shape[:2]:
        raise RuntimeError(f"intrinsic returned {res.albedo.shape[:2]}, expected {ctx['work'].shape[:2]}")
    job.set_stage("intrinsic", "running", 0.9, "Saving albedo, shading and residual")
    imageio.save_f16(job.path("albedo.npy"), res.albedo)
    imageio.save_f16(job.path("shading.npy"), res.shading)
    imageio.save_f16(job.path("residual.npy"), res.residual)
    layers = mod.layers_for_display(res)
    for k in ("albedo", "shading", "residual"):
        imageio.save_image(job.path("layers", f"{k}.jpg"), layers[k], quality=92)
    job.update(intrinsic_method=res.method)
    ctx["albedo"] = res.albedo.astype(np.float32)
    job.set_stage("intrinsic", "running", 1.0, f"Decomposed with {res.method}")


def _stage_segment(job: Job, ctx: dict[str, Any]) -> None:
    detail = job.options.detail
    job.set_stage("segment", "running", 0.02, f"Finding parts with SAM 2 · {detail}")
    masker = _sam_masks().SamMasker()
    masks = masker.generate(ctx["work"], detail=detail, progress=_Progress(job, "segment", 0.05, 0.98))
    ctx["masks"] = masks
    job.set_stage("segment", "running", 1.0, f"{len(masks)} part proposals")


def _stage_regions(job: Job, ctx: dict[str, Any]) -> None:
    job.set_stage("regions", "running", 0.02, "Merging proposals into regions")
    labels, info = _hierarchy().build_regions(ctx["work"], ctx["albedo"], ctx["masks"], detail=job.options.detail,
                                              progress=_Progress(job, "regions", 0.05, 0.9))
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    if labels.shape != ctx["work"].shape[:2]:
        raise RuntimeError(f"regions returned {labels.shape}, expected {ctx['work'].shape[:2]}")
    if labels.min() < 0:
        raise RuntimeError("regions left unassigned pixels (-1)")
    np.save(job.path("labels.npy"), labels)
    imageio.save_image(job.path("ids", "regions.png"), encode_region_ids(labels))
    imageio.save_image(job.path("layers", "regions.png"), regions_display(labels))
    imageio.save_image(job.path("layers", "edges.png"), edges_display(ctx["work"], labels))
    ctx["labels"], ctx["region_info"] = labels, info
    job.set_stage("regions", "running", 1.0, f"{int(labels.max()) + 1} regions")


def _stage_groups(job: Job, ctx: dict[str, Any]) -> None:
    opts = job.options
    job.set_stage("groups", "running", 0.05, "Clustering regions by paint color")
    regions, groups, group_map = _grouping().group_regions(ctx["labels"], ctx["albedo"], ctx["region_info"],
                                                          max_groups=opts.max_groups, delta_e=opts.delta_e)
    _write_grouping(job, regions, groups, group_map)
    job.set_stage("groups", "running", 1.0, f"{len(groups)} color groups from {len(regions)} regions")


def _write_grouping(job: Job, regions: list[Region], groups: list[ColorGroup], group_map: np.ndarray) -> None:
    """Persist regions/groups/group_map (+ id and display PNGs) and publish `groups`.

    The saved `mapping` is keyed by group id, and the grouping module renumbers ids
    (0..G-1 by area) on every merge/split/move/regroup, so the mapping is carried over
    by *region membership* (`remap_mapping`) rather than by id."""
    group_map = np.ascontiguousarray(group_map, dtype=np.int32)
    if group_map.min() < 0:
        raise RuntimeError("group map has unassigned pixels (-1)")
    np.save(job.path("group_map.npy"), group_map)
    imageio.save_image(job.path("ids", "groups.png"), encode_group_ids(group_map))
    imageio.save_image(job.path("layers", "groups.png"), groups_display(group_map, groups))
    with open(job.path("regions.json"), "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in regions], f)
    _invalidate(job.id)
    with job.lock:
        old_groups = [ColorGroup.from_dict(g) for g in job.meta.get("groups") or []]
        job.meta["mapping"] = remap_mapping(job.meta.get("mapping") or {}, old_groups, groups, regions)
    job.set_groups(groups, regions_count=len(regions))


def remap_mapping(mapping: dict[str, Any], old_groups: list[ColorGroup], new_groups: list[ColorGroup],
                  regions: list[Region]) -> dict[str, Any]:
    """Carry a `{gid: hex|null}` mapping across a regrouping by region membership.

    For every old group with a mapping entry, the color moves to the new group that
    holds the (area-weighted) majority of the old group's regions; ties and groups
    whose regions scattered (no new group holds more than half) are dropped. When
    two old groups land on the same new group the larger (first-listed) old group
    wins. Region ids are stable across edits (contract §3.2) so this is exact.
    """
    if not mapping:
        return {}
    area = {r.id: max(int(r.area), 1) for r in regions}
    region_to_new = {rid: g.id for g in new_groups for rid in g.region_ids}
    out: dict[str, Any] = {}
    for old in old_groups:
        key = str(old.id)
        if key not in mapping:
            continue
        weights: Counter[int] = Counter()
        total = 0
        for rid in old.region_ids:
            w = area.get(rid, 1)
            total += w
            gid = region_to_new.get(rid)
            if gid is not None:
                weights[gid] += w
        if not weights or total <= 0:
            continue
        best, w = weights.most_common(1)[0]
        if w * 2 <= total or str(best) in out:
            continue
        out[str(best)] = mapping[key]
    return out


# =============================================================================
# id / display encoders
# =============================================================================

def encode_region_ids(labels: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 with `id = R + 256·G + 65536·B` (exact for ids below 2^24)."""
    ids = labels.astype(np.int64)
    out = np.empty(labels.shape + (3,), np.uint8)
    out[..., 0] = ids & 255
    out[..., 1] = (ids >> 8) & 255
    out[..., 2] = (ids >> 16) & 255
    return out


def decode_region_ids(rgb: np.ndarray) -> np.ndarray:
    """Inverse of `encode_region_ids` -> int32 HxW."""
    r = rgb.astype(np.int32)
    return r[..., 0] + 256 * r[..., 1] + 65536 * r[..., 2]


def encode_group_ids(group_map: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 with R = group id (clamped to 0..255), G = B = 0."""
    out = np.zeros(group_map.shape + (3,), np.uint8)
    out[..., 0] = np.clip(group_map, 0, 255).astype(np.uint8)
    return out


def regions_display(labels: np.ndarray) -> np.ndarray:
    """Each region in a random but deterministic saturated color (uint8 RGB)."""
    n = int(labels.max()) + 1
    rng = np.random.default_rng(1234)
    hsv = np.stack([rng.random(n) * 179, 120 + rng.random(n) * 135, 150 + rng.random(n) * 105], 1).astype(np.uint8)
    import cv2
    lut = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0]
    return lut[np.clip(labels, 0, n - 1)]


def groups_display(group_map: np.ndarray, groups: list[ColorGroup]) -> np.ndarray:
    """Each pixel painted with its group's albedo color (uint8 RGB)."""
    n = max(int(group_map.max()) + 1, len(groups), 1)
    lut = np.full((n, 3), 128, np.uint8)
    for g in groups:
        if 0 <= g.id < n:
            lut[g.id] = imageio.to_uint8(imageio.hex_to_rgb01(g.albedo_hex))
    return lut[np.clip(group_map, 0, n - 1)]


def edges_display(image_rgb_u8: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Region boundaries drawn in cyan over a slightly dimmed copy of the image."""
    edge = np.zeros(labels.shape, bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    out = (image_rgb_u8.astype(np.float32) * 0.8).astype(np.uint8)
    out[edge] = (40, 230, 255)
    return out


# =============================================================================
# cached layers and renderers
# =============================================================================

_layers_cache: "OrderedDict[str, dict[str, np.ndarray]]" = OrderedDict()
_renderer_cache: "OrderedDict[str, Any]" = OrderedDict()
_cache_lock = threading.Lock()
_LAYERS_CACHE_SIZE = 4
_RENDERER_CACHE_SIZE = 3


def _require_ready(job: Job) -> None:
    if job.status != "ready":
        raise PipelineError(409, f"job is {job.status}, not ready")


def load_layers(job: Job) -> dict[str, np.ndarray]:
    """Working-resolution arrays `albedo`, `shading`, `residual` (float32 linear) and
    `labels`, `group_map` (int32). Cached per job (small LRU); invalidated by group
    edits. Raises PipelineError(409) before analysis is complete."""
    _require_ready(job)
    with _cache_lock:
        if job.id in _layers_cache:
            _layers_cache.move_to_end(job.id)
            return _layers_cache[job.id]
    try:
        layers = {
            "albedo": imageio.load_f16(job.path("albedo.npy")),
            "shading": imageio.load_f16(job.path("shading.npy")),
            "residual": imageio.load_f16(job.path("residual.npy")),
            "labels": np.load(job.path("labels.npy")).astype(np.int32),
            "group_map": np.load(job.path("group_map.npy")).astype(np.int32),
        }
    except FileNotFoundError as e:
        raise PipelineError(404, f"missing artifact: {os.path.basename(str(e.filename))}") from e
    with _cache_lock:
        _layers_cache[job.id] = layers
        while len(_layers_cache) > _LAYERS_CACHE_SIZE:
            _layers_cache.popitem(last=False)
    return layers


def get_renderer(job: Job):
    """The engine `Renderer` holding this job's layers on the GPU. Cached per job and
    dropped on regroup/merge/split/move (or when more than a few jobs are active)."""
    _require_ready(job)
    with _cache_lock:
        r = _renderer_cache.get(job.id)
        if r is not None:
            _renderer_cache.move_to_end(job.id)
            return r
    layers = load_layers(job)
    with _render_lock:
        renderer = _engine().Renderer(layers["albedo"], layers["shading"], layers["residual"],
                                      layers["group_map"], job.groups())
    with _cache_lock:
        _renderer_cache[job.id] = renderer
        while len(_renderer_cache) > _RENDERER_CACHE_SIZE:
            _renderer_cache.popitem(last=False)
    return renderer


def _invalidate(job_id: str) -> None:
    with _cache_lock:
        _layers_cache.pop(job_id, None)
        _renderer_cache.pop(job_id, None)


def invalidate(job: Job) -> None:
    """Drop cached arrays and the GPU renderer for this job (after delete/regroup)."""
    _invalidate(job.id)
    _free_cuda()


# =============================================================================
# render / export
# =============================================================================

def _validate_mapping(job: Job, mapping: Any) -> Mapping:
    if mapping is None:
        return {}
    if not isinstance(mapping, dict):
        raise PipelineError(400, "mapping must be an object of group id -> hex color or null")
    valid = {g["id"] for g in job.meta.get("groups") or []}
    out: Mapping = {}
    for k, v in mapping.items():
        try:
            gid = int(k)
        except (TypeError, ValueError):
            raise PipelineError(400, f"mapping key {k!r} is not a group id") from None
        if valid and gid not in valid:
            raise PipelineError(400, f"mapping refers to unknown group {gid}")
        if v in (None, "", False):
            out[gid] = None
            continue
        if not isinstance(v, str):
            raise PipelineError(400, f"mapping value for group {gid} must be a hex color or null")
        try:
            rgb = imageio.hex_to_rgb01(v)
        except ValueError:
            raise PipelineError(400, f"mapping value {v!r} for group {gid} is not a hex color") from None
        out[gid] = imageio.rgb01_to_hex(rgb)
    return out


def _validate_options(options: Any) -> RenderOptions:
    if options is None:
        return RenderOptions()
    if not isinstance(options, dict):
        raise PipelineError(400, "options must be an object")
    o = RenderOptions.from_dict(options)
    if o.mode not in ("shift", "flat"):
        raise PipelineError(400, f"options.mode must be 'shift' or 'flat', not {o.mode!r}")
    for name, lo, hi in (("texture", 0.0, 1.0), ("feather_px", 0.0, 64.0), ("residual_tint", 0.0, 1.0),
                         ("shading_strength", 0.0, 4.0), ("saturation", 0.0, 4.0)):
        try:
            val = float(getattr(o, name))
        except (TypeError, ValueError):
            raise PipelineError(400, f"options.{name} must be a number") from None
        if not (lo <= val <= hi):
            raise PipelineError(400, f"options.{name} must be between {lo} and {hi}")
        setattr(o, name, val)
    o.keep_residual = bool(o.keep_residual)
    o.sharpen_edges = bool(o.sharpen_edges)
    return o


def render_preview(job: Job, mapping: Any, options: Any) -> bytes:
    """JPEG bytes of the recolored image at `config.PREVIEW_LONG_SIDE`. Mapping and
    options are validated (PipelineError 400 on bad values); the job must be ready."""
    m = _validate_mapping(job, mapping)
    o = _validate_options(options)
    renderer = get_renderer(job)
    with _render_lock:
        out = renderer.render_at(config.PREVIEW_LONG_SIDE, m, o)
    return imageio.encode_jpeg(np.ascontiguousarray(out), quality=90)


def _export_name(job: Job, quality: str, fmt: str) -> str:
    stem = os.path.splitext(job.meta["name"])[0] or "image"
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60]
    return f"{stem}_recolor_{quality}_{time.strftime('%Y%m%d-%H%M%S')}.{fmt}"


def export(job: Job, mapping: Any, options: Any, quality: str, fmt: str) -> dict[str, Any]:
    """Render at working (`quality="work"`) or original (`"full"`) resolution and write
    `exports/<name>.<fmt>`. Returns `{file, width, height, ms}`.

    Full resolution re-runs the intrinsic decomposition on the original when it has at
    most `config.FULLRES_INTRINSIC_MAX_PIXELS` pixels (so the identity holds exactly)
    **and** that run uses the same method the preview was tuned with (`intrinsic_method`
    in the job meta); if the model falls back to the heuristic (CUDA OOM on a shared
    card) or the free VRAM is clearly too small to try, the working-res layers are
    guided-upsampled instead, with the residual recomputed so that
    `albedo·shading + residual == original` still holds. The result's `intrinsic` key
    says which path ran: `'careaga'` / `'heuristic'` (full-res decomposition) or
    `'upsampled'`. The group map is nearest-upsampled and snapped to the original's
    edges with the guided filter (skipped only when that would need more GPU memory
    than is sensible on a shared card).
    """
    if quality not in EXPORT_QUALITIES:
        raise PipelineError(400, f"quality must be one of {list(EXPORT_QUALITIES)}")
    if fmt not in EXPORT_FORMATS:
        raise PipelineError(400, f"format must be one of {list(EXPORT_FORMATS)}")
    m = _validate_mapping(job, mapping)
    o = _validate_options(options)
    t0 = time.perf_counter()
    if quality == "work":
        renderer = get_renderer(job)
        with _render_lock:
            out = renderer.render(m, o)
        intrinsic = "work"
    else:
        with gpu_lock:
            _touch_activity()
            out, intrinsic = _render_full(job, m, o)
    name = _export_name(job, quality, fmt)
    os.makedirs(job.path("exports"), exist_ok=True)
    imageio.save_image(job.path("exports", name), np.ascontiguousarray(out), quality=95)
    ms = int((time.perf_counter() - t0) * 1000)
    _free_cuda()
    return {"file": name, "width": int(out.shape[1]), "height": int(out.shape[0]), "ms": ms, "intrinsic": intrinsic}


# Lower bound on the VRAM a full-resolution Careaga pass needs (peak allocations measured
# on the 5090: 4.8 GB at 1.6 MP, 13.8 GB at 6.3 MP, with the model weights already
# resident). Deliberately lenient: it only skips attempts that are hopeless, so a shared
# card is not churned by a 20-second pass that must OOM; a pass that fits by this
# estimate but still OOMs falls back inside the intrinsic module and is then caught by
# the method-consistency check in `_render_full`.
_FULLRES_GB_PER_MP = 1.5
_FULLRES_GB_FLOOR = 1.0
_FULLRES_HEADROOM = 1.0


def _fullres_intrinsic_fits(n_pixels: int) -> bool:
    """False when the free VRAM is clearly too small for a full-res model pass."""
    try:
        import torch
        if not torch.cuda.is_available():
            return True
        free, _ = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001
        return True
    need = (_FULLRES_GB_FLOOR + _FULLRES_GB_PER_MP * n_pixels / 1e6) * _FULLRES_HEADROOM
    fits = free / 2**30 >= need
    if not fits:
        log.info("full-res intrinsic skipped: %.1f GB free, ~%.1f GB needed for %.1f MP",
                 free / 2**30, need, n_pixels / 1e6)
    return fits


def _render_full(job: Job, m: Mapping, o: RenderOptions) -> tuple[np.ndarray, str]:
    """The full-resolution render and which intrinsic path produced its layers
    (`'careaga'` / `'heuristic'` for a full-res decomposition, `'upsampled'`)."""
    layers = load_layers(job)
    original = imageio.load_image(job.path(job.original_file))
    H, W = original.shape[:2]
    groups = job.groups()
    n_groups = max(len(groups), int(layers["group_map"].max()) + 1)

    group_map = filters.upsample_labels(layers["group_map"], (W, H))
    guide = imageio.to_float(original)
    if H * W * min(n_groups, 32) <= _REFINE_MAX_ELEMENTS:
        try:
            group_map = filters.refine_labels_with_guide(group_map, guide, radius=4)
        except RuntimeError as e:  # CUDA OOM: keep the nearest upsample
            log.warning("label refinement skipped (%s)", e)
            _free_cuda()
    else:
        log.info("label refinement skipped: %dx%d with %d groups exceeds the memory budget", W, H, n_groups)

    lin = imageio.srgb_to_linear(guide)
    albedo = shading = residual = None
    path = "upsampled"
    expected = job.meta.get("intrinsic_method") or None
    if H * W <= config.FULLRES_INTRINSIC_MAX_PIXELS and (expected != "careaga" or _fullres_intrinsic_fits(H * W)):
        try:
            res = _intrinsic().decompose(original, method=expected or "auto")
            if res.albedo.shape[:2] != (H, W):
                log.warning("full-res intrinsic returned %s for %dx%d; upsampling working-res layers",
                            res.albedo.shape[:2], H, W)
            elif expected is not None and res.method != expected:
                # The model fell back (typically CUDA OOM on the shared card). Rendering
                # with different layers than the preview was tuned on would change the
                # look, so use the preview's own layers upsampled instead.
                log.warning("full-res intrinsic ran %r but the preview used %r; upsampling working-res layers",
                            res.method, expected)
                _free_cuda()
            else:
                albedo, shading, residual = res.albedo, res.shading, res.residual
                path = str(res.method)
        except Exception as e:  # noqa: BLE001 - fall back to upsampling, never fail the export
            log.warning("full-res intrinsic failed (%s); upsampling working-res layers", e)
            _free_cuda()
    if albedo is None:
        path = "upsampled"
        albedo = np.clip(filters.guided_filter(guide, layers["albedo"], radius=8, eps=1e-3), 0.0, 1.0)
        shading = np.clip(filters.guided_filter(guide, layers["shading"], radius=8, eps=1e-3), 0.0, None)
        residual = (lin - albedo * shading).astype(np.float32)
    out = _engine().render_once(albedo.astype(np.float32), shading.astype(np.float32),
                                residual.astype(np.float32), group_map, groups, m, o)
    return out, path


# =============================================================================
# group edits
# =============================================================================

def _load_regions(job: Job) -> list[Region]:
    try:
        with open(job.path("regions.json"), "r", encoding="utf-8") as f:
            return [Region.from_dict(d) for d in json.load(f)]
    except FileNotFoundError as e:
        raise PipelineError(404, "missing artifact: regions.json") from e


def _int_list(value: Any, what: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise PipelineError(400, f"{what} must be a non-empty list of integers")
    try:
        return [int(v) for v in value]
    except (TypeError, ValueError):
        raise PipelineError(400, f"{what} must be a non-empty list of integers") from None


def _int_value(value: Any, what: str) -> int:
    if isinstance(value, bool) or value is None:
        raise PipelineError(400, f"{what} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise PipelineError(400, f"{what} must be an integer") from None


def apply_group_edit(job: Job, kind: str, payload: dict[str, Any]) -> None:
    """Apply a user edit to the grouping and persist every affected artifact.

    kinds: `merge {group_ids}`, `split {group_id, k}`, `move {region_ids, group_id}`,
    `regroup {max_groups?, delta_e?}`, `update {group_id, name?, locked?, is_background?}`.
    Bad ids raise PipelineError(400); the job must be ready (409 otherwise). Cached
    layers and the renderer are invalidated and a `groups` event is published.
    """
    if kind not in GROUP_EDIT_KINDS:
        raise PipelineError(400, f"unknown edit {kind!r}")
    _require_ready(job)
    payload = payload or {}
    groups = job.groups()
    gids = {g.id for g in groups}

    if kind == "update":
        gid = _int_value(payload.get("group_id"), "group_id")
        target = next((g for g in groups if g.id == gid), None)
        if target is None:
            raise PipelineError(400, f"unknown group {gid}")
        if "name" in payload and payload["name"] is not None:
            name = str(payload["name"]).strip()[:48]
            if not name:
                raise PipelineError(400, "name must not be empty")
            target.name = name
        if "locked" in payload and payload["locked"] is not None:
            target.locked = bool(payload["locked"])
        if "is_background" in payload and payload["is_background"] is not None:
            target.is_background = bool(payload["is_background"])
        job.set_groups(groups)
        with _cache_lock:
            _renderer_cache.pop(job.id, None)   # the renderer keeps its own copy of the groups
        return

    layers = load_layers(job)
    regions = _load_regions(job)
    labels, albedo, group_map = layers["labels"], layers["albedo"], layers["group_map"]
    grouping = _grouping()

    if kind == "merge":
        ids = _int_list(payload.get("group_ids"), "group_ids")
        unknown = [i for i in ids if i not in gids]
        if unknown:
            raise PipelineError(400, f"unknown groups {unknown}")
        if len(set(ids)) < 2:
            raise PipelineError(400, "merge needs at least two distinct groups")
        with gpu_lock:
            regions, groups, group_map = grouping.merge_groups(groups, regions, group_map, labels, ids)
    elif kind == "split":
        gid = _int_value(payload.get("group_id"), "group_id")
        k = _int_value(payload.get("k", 2), "k")
        if gid not in gids:
            raise PipelineError(400, f"unknown group {gid}")
        if not (2 <= k <= 8):
            raise PipelineError(400, "k must be between 2 and 8")
        with gpu_lock:
            regions, groups, group_map = grouping.split_group(groups, regions, group_map, labels, albedo, gid, k)
    elif kind == "move":
        rids = _int_list(payload.get("region_ids"), "region_ids")
        gid = _int_value(payload.get("group_id"), "group_id")
        known = {r.id for r in regions}
        unknown = [i for i in rids if i not in known]
        if unknown:
            raise PipelineError(400, f"unknown regions {unknown[:8]}")
        if gid not in gids:
            raise PipelineError(400, f"unknown group {gid}")
        with gpu_lock:
            regions, groups, group_map = grouping.move_regions(groups, regions, group_map, labels, rids, gid)
    else:  # regroup
        opts = job.options
        max_groups = payload.get("max_groups", opts.max_groups)
        delta_e = payload.get("delta_e", opts.delta_e)
        if max_groups is not None:
            max_groups = _int_value(max_groups, "max_groups")
            if max_groups < 1:
                raise PipelineError(400, "max_groups must be at least 1")
        try:
            delta_e = float(delta_e)
        except (TypeError, ValueError):
            raise PipelineError(400, "delta_e must be a number") from None
        if not (0.5 <= delta_e <= 100.0):
            raise PipelineError(400, "delta_e must be between 0.5 and 100")
        with gpu_lock:
            regions, groups, group_map = grouping.regroup(regions, labels, albedo, max_groups, delta_e)
        job.update(options={**opts.to_dict(), "max_groups": max_groups, "delta_e": delta_e})

    _write_grouping(job, list(regions), list(groups), np.asarray(group_map))


# =============================================================================
# convenience for the API layer
# =============================================================================

def parse_analysis_options(fields: dict[str, Any]) -> AnalysisOptions:
    """Validate `detail`, `intrinsic`, `max_groups`, `delta_e` from a form or JSON
    body (strings accepted for the numbers). PipelineError(400) on bad values."""
    opts = AnalysisOptions()
    detail = fields.get("detail")
    if detail not in (None, ""):
        if detail not in DETAIL_LEVELS:
            raise PipelineError(400, f"detail must be one of {list(DETAIL_LEVELS)}")
        opts.detail = detail
    intrinsic = fields.get("intrinsic")
    if intrinsic not in (None, ""):
        if intrinsic not in INTRINSIC_METHODS:
            raise PipelineError(400, f"intrinsic must be one of {list(INTRINSIC_METHODS)}")
        opts.intrinsic = intrinsic
    mg = fields.get("max_groups")
    if mg not in (None, "", "null", "auto"):
        mg = _int_value(mg, "max_groups")
        if not (1 <= mg <= 256):
            raise PipelineError(400, "max_groups must be between 1 and 256")
        opts.max_groups = mg
    de = fields.get("delta_e")
    if de not in (None, ""):
        try:
            de = float(de)
        except (TypeError, ValueError):
            raise PipelineError(400, "delta_e must be a number") from None
        if not (0.5 <= de <= 100.0):
            raise PipelineError(400, "delta_e must be between 0.5 and 100")
        opts.delta_e = de
    return opts


def validate_state(job: Job, body: dict[str, Any]) -> dict[str, Any]:
    """Validated `{mapping?, render_options?, palette_id?}` for `PUT /state`."""
    out: dict[str, Any] = {}
    if "mapping" in body:
        out["mapping"] = mapping_to_json(_validate_mapping(job, body["mapping"]))
    if "render_options" in body:
        out["render_options"] = _validate_options(body["render_options"]).to_dict()
    if "palette_id" in body:
        pid = body["palette_id"]
        if pid is not None and (not isinstance(pid, str) or not pid.isalnum() or len(pid) > 64):
            raise PipelineError(400, "palette_id must be an alphanumeric id or null")
        out["palette_id"] = pid
    return out


__all__ = [
    "PipelineError", "analyze", "apply_group_edit", "decode_region_ids", "encode_group_ids",
    "encode_region_ids", "enqueue", "export", "get_renderer", "gpu_lock", "invalidate", "load_layers",
    "model_status", "parse_analysis_options", "queue_length", "remap_mapping", "render_preview",
    "resume_pending", "start_warmup", "start_worker", "validate_state",
]
